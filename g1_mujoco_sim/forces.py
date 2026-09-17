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


def decode_forces(payload: bytes | str) -> tuple[ForceCommand, ...]:
    """Decode a whole batch or raise ValueError; never accept a partial batch."""
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        message = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("force message must be valid UTF-8 JSON") from exc
    if not isinstance(message, dict):
        raise ValueError("force message must be an object with a forces array")
    entries = message.get("forces")
    if not isinstance(entries, list) or not entries:
        raise ValueError("forces must be a non-empty array")
    commands = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each force must be an object")
        name = entry.get("body_name")
        if not isinstance(name, str) or not name.strip() or "\x00" in name:
            raise ValueError("body_name must be a non-empty body name")
        direction = entry.get("direction")
        if not isinstance(direction, list) or len(direction) != 3:
            raise ValueError("direction must contain three numbers")
        xyz = tuple(_number(value, "direction") for value in direction)
        # Scale first to normalize safely even for very large/small vectors.
        scale = max(abs(value) for value in xyz)
        if scale == 0:
            raise ValueError("direction must be non-zero")
        scaled = tuple(value / scale for value in xyz)
        norm = math.hypot(*scaled)
        magnitude = _number(entry.get("magnitude"), "magnitude")
        duration = _number(entry.get("duration"), "duration")
        if magnitude < 0 or duration <= 0:
            raise ValueError("magnitude must be non-negative and duration must be positive")
        force = tuple(value / norm * magnitude for value in scaled)
        commands.append(ForceCommand(name, force, duration))
    return tuple(commands)
