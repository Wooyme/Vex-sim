"""Validated world-frame force commands, independent of MQTT and MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class ForceCommand:
    body_name: str
    force: tuple[float, float, float]
    duration: float


@dataclass(frozen=True)
class PointCommand:
    """Drive a body's center of mass toward a world height and horizontal velocity."""

    body_name: str
    height: float
    velocity: tuple[float, float]
    duration: float


@dataclass(frozen=True)
class ActiveForce:
    body_id: int
    force: tuple[float, float, float]
    end_time: float


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _load_message(payload: bytes | str) -> dict:
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        message = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("force message must be valid UTF-8 JSON") from exc
    if not isinstance(message, dict):
        raise ValueError("force message must be an object with a forces or points array")
    return message


def _body_name(entry: dict) -> str:
    name = entry.get("body_name")
    if not isinstance(name, str) or not name.strip() or "\x00" in name:
        raise ValueError("body_name must be a non-empty body name")
    return name


def _vector(value: object, field: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must contain {size} numbers")
    return tuple(_number(item, field) for item in value)


def _duration(entry: dict) -> float:
    duration = _number(entry.get("duration"), "duration")
    if duration <= 0:
        raise ValueError("duration must be positive")
    return duration


def _decode_force(entry: object) -> ForceCommand:
    if not isinstance(entry, dict):
        raise ValueError("each force must be an object")
    name = _body_name(entry)
    xyz = _vector(entry.get("direction"), "direction", 3)
    # Scale first to normalize safely even for very large/small vectors.
    scale = max(abs(value) for value in xyz)
    if scale == 0:
        raise ValueError("direction must be non-zero")
    scaled = tuple(value / scale for value in xyz)
    norm = math.hypot(*scaled)
    magnitude = _number(entry.get("magnitude"), "magnitude")
    duration = _duration(entry)
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")
    force = tuple(value / norm * magnitude for value in scaled)
    return ForceCommand(name, force, duration)


def _decode_point(entry: object) -> PointCommand:
    if not isinstance(entry, dict):
        raise ValueError("each point must be an object")
    name = _body_name(entry)
    height = _number(entry.get("height"), "height")
    velocity = _vector(entry.get("velocity"), "velocity", 2)
    return PointCommand(name, height, velocity, _duration(entry))


def _entries(message: dict, key: str) -> list:
    entries = message.get(key)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{key} must be a non-empty array")
    return entries


def decode_forces(payload: bytes | str) -> tuple[ForceCommand, ...]:
    """Decode a whole batch or raise ValueError; never accept a partial batch."""
    return tuple(_decode_force(entry) for entry in _entries(_load_message(payload), "forces"))


def decode_commands(payload: bytes | str) -> tuple[ForceCommand | PointCommand, ...]:
    """Decode optional ``forces`` and ``points`` arrays as one all-or-nothing batch."""
    message = _load_message(payload)
    if "forces" not in message and "points" not in message:
        raise ValueError("force message must contain a forces or points array")
    commands: list[ForceCommand | PointCommand] = []
    if "forces" in message:
        commands.extend(_decode_force(entry) for entry in _entries(message, "forces"))
    if "points" in message:
        commands.extend(_decode_point(entry) for entry in _entries(message, "points"))
    return tuple(commands)
