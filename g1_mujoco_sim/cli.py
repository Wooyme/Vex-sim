"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import MqttConfig, SimulationConfig, TerrainConfig
from .simulation import G1Simulation


def _position(value: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("position must be x,y,z") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("position must contain exactly three values")
    return result  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Unitree G1 29-DoF robot in MuJoCo (standalone holosoma run_sim migration)."
    )
    parser.add_argument(
        "compat", nargs="*", help=argparse.SUPPRESS,
    )
    parser.add_argument("--terrain", choices=("plane", "mix", "mesh"), default="plane")
    parser.add_argument("--terrain-file", type=Path, help="OBJ/STL file; implies --terrain mesh")
    parser.add_argument("--terrain-seed", type=int, default=42)
    parser.add_argument("--terrain-size", type=float, default=8.0)
    parser.add_argument("--terrain-resolution", type=float, default=0.1)
    parser.add_argument("--terrain-roughness", type=float, default=0.05)
    parser.add_argument("--terrain-scale", type=float, default=1.0)
    parser.add_argument("--terrain-position", type=_position, default=(0.0, 0.0, 0.0), metavar="X,Y,Z")
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=1000)
    parser.add_argument("--viewer-fps", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="0 runs until the viewer closes/Ctrl+C")
    parser.add_argument("--control", choices=("stand", "zero"), default="stand")
    parser.add_argument("--no-gantry", action="store_true", help="Disable the virtual elastic support")
    parser.add_argument("--gantry-height", type=float, default=3.0)
    parser.add_argument("--gantry-length", type=float, default=0.0)
    parser.add_argument("--gantry-stiffness", type=float, default=200.0)
    parser.add_argument("--gantry-damping", type=float, default=100.0)
    parser.add_argument("--gantry-apply-force", type=float, default=0.0)
    parser.add_argument("--gantry-apply-force-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--bridge", action="store_true", help="Enable the optional Unitree DDS bridge")
    parser.add_argument("--bridge-interface", help="Network interface for Unitree DDS (default: lo/lo0)")
    parser.add_argument(
        "--mqtt-broker",
        help="MQTT broker: host[:port] or mqtt[s]://[user:pass@]host[:port]",
    )
    parser.add_argument("--mqtt-topic", help="MQTT topic used to publish robot state")
    parser.add_argument("--mqtt-frequency", type=float, help="MQTT wall-clock publish rate in Hz")
    parser.add_argument("--mqtt-force-topic", help="MQTT topic used to receive timed world-frame forces")
    return parser


def _compat_terrain(tokens: list[str], selected: str) -> str:
    """Accept the old compositional tokens for easy migration."""
    allowed = {"simulator:mujoco", "robot:g1_29dof"}
    for token in tokens:
        if token == "terrain:terrain_locomotion_plane":
            selected = "plane"
        elif token == "terrain:terrain_locomotion_mix":
            selected = "mix"
        elif token == "terrain:terrain_load_obj":
            selected = "mesh"
        elif token not in allowed:
            raise ValueError(f"Unsupported compatibility argument: {token}")
    return selected


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        terrain_kind = _compat_terrain(args.compat, args.terrain)
        if args.terrain_file is not None:
            terrain_kind = "mesh"
        if args.fps <= 0 or args.viewer_fps <= 0 or args.steps < 0:
            raise ValueError("fps/viewer-fps must be positive and steps must be non-negative")
        mqtt_values = (args.mqtt_topic, args.mqtt_frequency, args.mqtt_force_topic)
        if any(value is not None for value in mqtt_values) and args.mqtt_broker is None:
            raise ValueError("--mqtt-broker is required for MQTT publishing or subscription")
        mqtt_config = None
        if args.mqtt_broker is not None:
            mqtt_config = MqttConfig(
                args.mqtt_broker, args.mqtt_topic, args.mqtt_frequency, args.mqtt_force_topic,
            )
            if mqtt_config.frequency is not None and mqtt_config.frequency > args.fps:
                raise ValueError("MQTT frequency cannot exceed simulation FPS")
        config = SimulationConfig(
            terrain=TerrainConfig(
                kind=terrain_kind, file=args.terrain_file, seed=args.terrain_seed,
                size=args.terrain_size, resolution=args.terrain_resolution,
                roughness=args.terrain_roughness, scale=args.terrain_scale,
                position=args.terrain_position, friction=args.friction,
            ),
            fps=args.fps, viewer_fps=args.viewer_fps, headless=args.headless,
            realtime=not args.no_realtime, steps=args.steps, control=args.control,
            gantry=not args.no_gantry, gantry_height=args.gantry_height,
            gantry_length=args.gantry_length, gantry_stiffness=args.gantry_stiffness,
            gantry_damping=args.gantry_damping,
            gantry_apply_force=args.gantry_apply_force,
            gantry_apply_force_sign=args.gantry_apply_force_sign,
            bridge=args.bridge, bridge_interface=args.bridge_interface,
            mqtt=mqtt_config,
        )
        G1Simulation(config).run()
    except KeyboardInterrupt:
        logging.info("Simulation interrupted")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
