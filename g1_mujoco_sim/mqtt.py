"""MQTT transport for state publishing and timed force and point commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import threading
import time
from typing import Callable, Sequence
from urllib.parse import unquote, urlsplit

import numpy as np
import numpy.typing as npt

from .config import MqttConfig
from .forces import ForceCommand, PointCommand, decode_commands

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrokerEndpoint:
    host: str
    port: int
    tls: bool
    username: str | None = None
    password: str | None = None


def parse_broker(value: str) -> BrokerEndpoint:
    """Parse host[:port] or an mqtt/mqtts URL."""
    raw = value.strip()
    if not raw:
        raise ValueError("MQTT broker must not be empty")
    parsed = urlsplit(raw if "://" in raw else f"mqtt://{raw}")
    scheme = parsed.scheme.lower()
    if scheme not in {"mqtt", "mqtts"}:
        raise ValueError("MQTT broker scheme must be mqtt:// or mqtts://")
    if not parsed.hostname:
        raise ValueError(f"Invalid MQTT broker: {value!r}")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("MQTT broker must not contain a path, query, or fragment")
    tls = scheme == "mqtts"
    try:
        port = parsed.port or (8883 if tls else 1883)
    except ValueError as exc:
        raise ValueError(f"Invalid MQTT broker port in {value!r}") from exc
    return BrokerEndpoint(
        host=parsed.hostname,
        port=port,
        tls=tls,
        username=unquote(parsed.username) if parsed.username is not None else None,
        password=unquote(parsed.password) if parsed.password is not None else None,
    )


def encode_state(
    *,
    simulation_time: float,
    joint_names: Sequence[str],
    joint_values: npt.ArrayLike,
    base_xyz: npt.ArrayLike,
    base_quat_wxyz: npt.ArrayLike,
    timestamp: float | None = None,
) -> str:
    """Encode one state sample as compact JSON."""
    payload = {
        "timestamp": time.time() if timestamp is None else timestamp,
        "simulation_time": simulation_time,
        "joint_names": list(joint_names),
        "joint_values": np.asarray(joint_values, dtype=np.float64).tolist(),
        "base_xyz": np.asarray(base_xyz, dtype=np.float64).tolist(),
        "base_quat_wxyz": np.asarray(base_quat_wxyz, dtype=np.float64).tolist(),
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


class MqttStatePublisher:
    """Shared asynchronous MQTT connection with bounded startup waiting."""

    def __init__(
        self, config: MqttConfig, connect_timeout: float = 5.0,
        *, on_forces: Callable[[tuple[ForceCommand | PointCommand, ...]], None] | None = None,
    ):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("MQTT support is not installed; run `uv sync`") from exc

        endpoint = parse_broker(config.broker)
        self._mqtt = mqtt
        self._topic = config.topic
        self._force_topic = config.force_topic
        self._on_forces = on_forces
        if self._force_topic is not None and on_forces is None:
            raise ValueError("A force subscription requires an on_forces callback")
        self._connected = threading.Event()
        self._startup_done = threading.Event()
        self._connect_error: str | None = None
        self._subscription_mid: int | None = None
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"g1-mujoco-sim-{os.getpid()}",
            clean_session=True,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_subscribe = self._on_subscribe
        self._client.on_message = self._on_message
        if endpoint.username is not None:
            self._client.username_pw_set(endpoint.username, endpoint.password)
        if endpoint.tls:
            self._client.tls_set()

        LOG.info("Connecting MQTT client to %s:%d", endpoint.host, endpoint.port)
        try:
            self._client.connect_async(endpoint.host, endpoint.port, keepalive=30)
            self._client.loop_start()
            ready = self._startup_done.wait(connect_timeout)
            if not ready or not self._connected.is_set():
                detail = self._connect_error or "connection/subscription timed out"
                raise RuntimeError(f"MQTT startup failed at {endpoint.host}:{endpoint.port}: {detail}")
        except BaseException:
            self.close()
            raise

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        del userdata, flags, properties
        self._connected.clear()
        self._subscription_mid = None
        if reason_code == 0:
            self._connect_error = None
            LOG.info("MQTT connected; state=%s, forces=%s", self._topic, self._force_topic)
            if self._force_topic is not None:
                result, self._subscription_mid = client.subscribe(self._force_topic, qos=0)
                if result != self._mqtt.MQTT_ERR_SUCCESS:
                    self._fail_startup(f"force subscription failed: {self._mqtt.error_string(result)}")
            else:
                self._connected.set()
                self._startup_done.set()
        else:
            self._fail_startup(f"connection rejected: {reason_code}")

    def _fail_startup(self, reason: str) -> None:
        self._connect_error = reason
        self._connected.clear()
        self._startup_done.set()
        LOG.error("MQTT %s", reason)

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties) -> None:
        del client, userdata, properties
        if mid != self._subscription_mid:
            return
        if not reason_codes or any(code.is_failure for code in reason_codes):
            self._fail_startup(f"force subscription rejected: {reason_codes}")
            return
        self._connected.set()
        self._startup_done.set()
        LOG.info("MQTT subscribed to force topic %s", self._force_topic)

    def _on_message(self, client, userdata, message) -> None:
        del client, userdata
        if message.topic != self._force_topic or message.retain:
            return
        try:
            commands = decode_commands(message.payload)
        except ValueError as exc:
            LOG.warning("Ignoring invalid MQTT force batch: %s", exc)
            return
        if self._on_forces is not None:
            self._on_forces(commands)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties) -> None:
        del client, userdata, disconnect_flags, properties
        self._connected.clear()
        if reason_code != 0:
            LOG.warning("MQTT disconnected unexpectedly: %s", reason_code)

    def publish_state(
        self,
        *,
        simulation_time: float,
        joint_names: Sequence[str],
        joint_values: npt.ArrayLike,
        base_xyz: npt.ArrayLike,
        base_quat_wxyz: npt.ArrayLike,
    ) -> None:
        if self._topic is None:
            return
        payload = encode_state(
            simulation_time=simulation_time,
            joint_names=joint_names,
            joint_values=joint_values,
            base_xyz=base_xyz,
            base_quat_wxyz=base_quat_wxyz,
        )
        result = self._client.publish(self._topic, payload, qos=0, retain=False)
        if result.rc != self._mqtt.MQTT_ERR_SUCCESS:
            LOG.warning("MQTT publish failed: %s", self._mqtt.error_string(result.rc))

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.disconnect()
        finally:
            client.loop_stop()
            self._client = None
