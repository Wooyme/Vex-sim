import json

import numpy as np
import pytest

from g1_mujoco_sim.config import MqttConfig, SimulationConfig
from g1_mujoco_sim.mqtt import encode_state, parse_broker
from g1_mujoco_sim.simulation import G1Simulation


def test_parse_plain_and_tls_brokers():
    plain = parse_broker("localhost:1884")
    assert (plain.host, plain.port, plain.tls) == ("localhost", 1884, False)

    secure = parse_broker("mqtts://user:p%40ss@example.com")
    assert (secure.host, secure.port, secure.tls) == ("example.com", 8883, True)
    assert (secure.username, secure.password) == ("user", "p@ss")


@pytest.mark.parametrize("broker", ["http://localhost", "mqtt://localhost/path", "mqtt://:1883"])
def test_invalid_broker_is_rejected(broker):
    with pytest.raises(ValueError):
        parse_broker(broker)


def test_state_json_has_documented_mujoco_quaternion_order():
    encoded = encode_state(
        timestamp=123.5,
        simulation_time=1.25,
        joint_names=("a", "b"),
        joint_values=np.array([0.1, -0.2]),
        base_xyz=np.array([1.0, 2.0, 3.0]),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    payload = json.loads(encoded)
    assert payload == {
        "timestamp": 123.5,
        "simulation_time": 1.25,
        "joint_names": ["a", "b"],
        "joint_values": [0.1, -0.2],
        "base_xyz": [1.0, 2.0, 3.0],
        "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def test_wall_clock_scheduler_and_simulation_state():
    config = SimulationConfig(
        mqtt=MqttConfig("localhost", "robot/g1/state", 10.0),
        headless=True,
        realtime=False,
        steps=1,
    )
    sim = G1Simulation(config)

    class FakePublisher:
        def __init__(self):
            self.samples = []

        def publish_state(self, **sample):
            self.samples.append(sample)

    publisher = FakePublisher()
    sim.mqtt_publisher = publisher
    sim._next_mqtt_publish = 10.0

    assert not sim._maybe_publish_mqtt(9.99)
    assert sim._maybe_publish_mqtt(10.0)
    assert not sim._maybe_publish_mqtt(10.05)
    assert sim._maybe_publish_mqtt(10.11)
    assert len(publisher.samples) == 2
    sample = publisher.samples[0]
    assert len(sample["joint_names"]) == len(sample["joint_values"]) == 29
    assert sample["base_xyz"].shape == (3,)
    assert sample["base_quat_wxyz"].shape == (4,)


def test_mqtt_config_rejects_invalid_topic_and_frequency():
    with pytest.raises(ValueError):
        MqttConfig("localhost", "robot/+/state", 10.0)
    with pytest.raises(ValueError):
        MqttConfig("localhost", "robot/state", 0.0)

