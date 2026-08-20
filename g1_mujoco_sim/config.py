"""Configuration and G1 constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

DEFAULT_JOINT_POSITIONS = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)

KP = (
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 28.501, 28.501,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
)

KD = (
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 1.814, 1.814,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
)


@dataclass(frozen=True)
class TerrainConfig:
    """Terrain description. ``kind`` is plane, mix, or mesh."""

    kind: str = "plane"
    file: Path | None = None
    seed: int = 42
    size: float = 8.0
    resolution: float = 0.1
    roughness: float = 0.05
    scale: float = 1.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: float = 1.0


@dataclass(frozen=True)
class MqttConfig:
    """MQTT destination and wall-clock publishing frequency."""

    broker: str
    topic: str
    frequency: float

    def __post_init__(self) -> None:
        if not self.broker.strip():
            raise ValueError("MQTT broker must not be empty")
        if not self.topic.strip():
            raise ValueError("MQTT topic must not be empty")
        if "+" in self.topic or "#" in self.topic:
            raise ValueError("MQTT publish topic cannot contain wildcards (+ or #)")
        if self.frequency <= 0:
            raise ValueError("MQTT frequency must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime configuration for one G1 in MuJoCo."""

    terrain: TerrainConfig = TerrainConfig()
    fps: int = 1000
    viewer_fps: int = 60
    headless: bool = False
    realtime: bool = True
    steps: int = 0
    control: str = "stand"
    gantry: bool = True
    gantry_height: float = 3.0
    gantry_length: float = 0.0
    gantry_stiffness: float = 200.0
    gantry_damping: float = 100.0
    gantry_apply_force: float = 0.0
    gantry_apply_force_sign: int = -1
    bridge: bool = False
    bridge_interface: str | None = None
    mqtt: MqttConfig | None = None
