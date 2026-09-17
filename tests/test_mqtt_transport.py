import json
from types import SimpleNamespace

import paho.mqtt.client as mqtt
import pytest

from g1_mujoco_sim.cli import main
from g1_mujoco_sim.config import MqttConfig, SimulationConfig
from g1_mujoco_sim.mqtt import MqttStatePublisher
from g1_mujoco_sim.simulation import G1Simulation


PAYLOAD = json.dumps({"forces": [{
    "body_name": "torso_link", "direction": [1, 0, 0], "magnitude": 10, "duration": 0.001,
}]}).encode()


@pytest.fixture
def fake_client(monkeypatch):
    class FakeClient:
        instances = []
        subscribe_rc = mqtt.MQTT_ERR_SUCCESS
        reject_subscription = False
        acknowledge = True
        connect_reason = 0

        def __init__(self, **kwargs):
            self.options = kwargs
            self.subscriptions = []
            self.publications = []
            self.stopped = False
            self.disconnected = False
            self.instances.append(self)

        def connect_async(self, *args, **kwargs):
            self.endpoint = (args, kwargs)

        def loop_start(self):
            self.connect()

        def connect(self):
            self.on_connect(self, None, None, self.connect_reason, None)
            if self.subscriptions and self.acknowledge and self.subscribe_rc == mqtt.MQTT_ERR_SUCCESS:
                self.on_subscribe(self, None, len(self.subscriptions),
                                  [SimpleNamespace(is_failure=self.reject_subscription)], None)

        def subscribe(self, topic, qos):
            self.subscriptions.append((topic, qos))
            return self.subscribe_rc, len(self.subscriptions)

        def publish(self, topic, payload, **kwargs):
            self.publications.append((topic, json.loads(payload), kwargs))
            return SimpleNamespace(rc=mqtt.MQTT_ERR_SUCCESS)

        def disconnect(self):
            self.disconnected = True
            self.on_disconnect(self, None, None, 0, None)

        def loop_stop(self):
            self.stopped = True

        def message(self, payload=PAYLOAD, retain=False, topic="forces"):
            self.on_message(self, None, SimpleNamespace(topic=topic, payload=payload, retain=retain))

    monkeypatch.setattr(mqtt, "Client", FakeClient)
    return FakeClient


def test_subscribe_reconnect_filter_and_close(fake_client, caplog):
    batches = []
    transport = MqttStatePublisher(MqttConfig("localhost", force_topic="forces"), on_forces=batches.append)
    client = fake_client.instances[0]
    assert client.options["clean_session"] is True
    assert client.subscriptions == [("forces", 0)]
    client.message(retain=True)
    client.message(topic="state")
    client.message(payload=b"invalid")
    assert not batches
    assert "invalid MQTT force batch" in caplog.text
    client.message()
    assert len(batches) == 1
    client.on_disconnect(client, None, None, 1, None)
    assert not transport._connected.is_set()
    client.connect()
    assert transport._connected.is_set()
    assert client.subscriptions == [("forces", 0), ("forces", 0)]
    assert len(batches) == 1
    transport.close()
    transport.close()
    assert client.stopped and client.disconnected


@pytest.mark.parametrize("failure", ["subscribe", "suback", "timeout", "connection"])
def test_startup_failure_closes_client(fake_client, failure):
    if failure == "subscribe":
        fake_client.subscribe_rc = mqtt.MQTT_ERR_NO_CONN
    elif failure == "suback":
        fake_client.reject_subscription = True
    elif failure == "timeout":
        fake_client.acknowledge = False
    else:
        fake_client.connect_reason = 5
    with pytest.raises(RuntimeError, match="MQTT startup failed"):
        MqttStatePublisher(MqttConfig("localhost", force_topic="forces"),
                           connect_timeout=0.001, on_forces=lambda _: None)
    assert fake_client.instances[0].stopped
    assert fake_client.instances[0].disconnected


@pytest.mark.parametrize("publish, subscribe", [(True, False), (False, True), (True, True)])
def test_run_modes_and_cleanup(fake_client, publish, subscribe):
    config = MqttConfig("localhost", "state" if publish else None,
                        10 if publish else None, "forces" if subscribe else None)
    sim = G1Simulation(SimulationConfig(mqtt=config, headless=True, realtime=False, steps=1))
    sim.run()
    client = fake_client.instances[0]
    assert bool(client.subscriptions) == subscribe
    assert bool(client.publications) == publish
    if publish:
        assert "joint_values" in client.publications[0][1]
    assert client.stopped and client.disconnected
    assert sim.mqtt_publisher is None


def test_message_queues_without_applying_and_disconnect_does_not_cancel(fake_client):
    sim = G1Simulation(SimulationConfig(headless=True, realtime=False, gantry=False))
    transport = MqttStatePublisher(MqttConfig("localhost", force_topic="forces"),
                                   on_forces=sim._enqueue_forces)
    client = fake_client.instances[0]
    client.message()
    assert not sim.data.xfrc_applied.any()
    sim._apply_forces()
    assert sim.data.xfrc_applied[sim.gantry_body_id, 0] == 10
    client.on_disconnect(client, None, None, 1, None)
    sim._apply_forces()
    assert sim.data.xfrc_applied[sim.gantry_body_id, 0] == 10
    sim.data.time = 0.001
    sim._apply_forces()
    assert not sim.data.xfrc_applied.any()
    transport.close()


def test_run_failure_still_closes_mqtt(fake_client, monkeypatch):
    sim = G1Simulation(SimulationConfig(
        mqtt=MqttConfig("localhost", force_topic="forces"), headless=True,
    ))

    def fail(_viewer):
        raise RuntimeError("simulation failed")

    monkeypatch.setattr(sim, "_run_loop", fail)
    with pytest.raises(RuntimeError, match="simulation failed"):
        sim.run()
    assert fake_client.instances[0].stopped and fake_client.instances[0].disconnected


@pytest.mark.parametrize("kwargs", [
    {}, {"topic": "state"}, {"frequency": 5}, {"force_topic": " "},
    {"force_topic": "robot/#"}, {"force_topic": "robot/+"},
    {"topic": "same", "frequency": 5, "force_topic": "same"},
    {"topic": "state", "frequency": float("nan")},
    {"topic": "state", "frequency": float("inf")},
])
def test_invalid_mqtt_config(kwargs):
    with pytest.raises(ValueError):
        MqttConfig("localhost", **kwargs)


@pytest.mark.parametrize("args", [
    ["--mqtt-force-topic", "forces"],
    ["--mqtt-broker", "localhost"],
    ["--mqtt-broker", "localhost", "--mqtt-topic", "state"],
    ["--mqtt-broker", "localhost", "--mqtt-force-topic", "forces", "--mqtt-frequency", "10"],
    ["--mqtt-broker", "localhost", "--mqtt-topic", "state", "--mqtt-frequency", "1001"],
])
def test_cli_rejects_incomplete_config(args):
    with pytest.raises(SystemExit, match="error:"):
        main(args)


def test_cli_allows_subscription_without_publishing(fake_client):
    main(["--headless", "--no-realtime", "--steps", "1",
          "--mqtt-broker", "localhost", "--mqtt-force-topic", "forces"])
    assert fake_client.instances[0].subscriptions == [("forces", 0)]
    assert not fake_client.instances[0].publications
