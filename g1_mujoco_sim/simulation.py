"""MuJoCo model lifecycle and direct simulation loop."""

from __future__ import annotations

import logging
from collections import deque
from importlib.resources import files
import math
import threading
import time

import numpy as np

from .bridge import UnitreeBridge
from .config import DEFAULT_JOINT_POSITIONS, JOINT_NAMES, KD, KP, SimulationConfig
from .gantry import VirtualGantry
from .forces import ActiveForce, ForceCommand
from .mqtt import MqttStatePublisher
from .terrain import build_world

LOG = logging.getLogger(__name__)


class G1Simulation:
    def __init__(self, config: SimulationConfig):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is not installed; run `uv sync`") from exc

        self.mujoco = mujoco
        self.config = config
        asset_root = files("g1_mujoco_sim").joinpath("assets/g1")
        robot_xml = asset_root.joinpath("g1_29dof.xml")
        mesh_dir = asset_root.joinpath("meshes")
        world = build_world(robot_xml, mesh_dir, config.terrain, 1.0 / config.fps)
        self.model = mujoco.MjModel.from_xml_string(world.xml)
        self.data = mujoco.MjData(self.model)

        if world.heightfield is not None:
            expected = self.model.hfield_data.size
            values = world.heightfield.ravel()
            if values.size != expected:
                raise RuntimeError(f"Heightfield size mismatch: generated {values.size}, model expects {expected}")
            self.model.hfield_data[:] = values

        if self.model.nu != 29:
            raise RuntimeError(f"Expected the G1 29-DoF model, found {self.model.nu} actuators")
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
        ])
        if np.any(self.joint_ids < 0):
            missing = [name for name, index in zip(JOINT_NAMES, self.joint_ids) if index < 0]
            raise RuntimeError(f"G1 model is missing joints: {missing}")
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.targets = np.asarray(DEFAULT_JOINT_POSITIONS)
        self.kp = np.asarray(KP)
        self.kd = np.asarray(KD)
        self.control_lower = self.model.actuator_ctrlrange[:, 0]
        self.control_upper = self.model.actuator_ctrlrange[:, 1]
        self.gantry_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        self.free_qpos = int(self.model.jnt_qposadr[self.free_joint_id])
        self.virtual_gantry = VirtualGantry(
            enabled=config.gantry,
            stiffness=config.gantry_stiffness,
            damping=config.gantry_damping,
            height=config.gantry_height,
            length=config.gantry_length,
            apply_force=config.gantry_apply_force,
            apply_force_sign=config.gantry_apply_force_sign,
            point=np.array([0.0, 0.0, config.gantry_height]),
        )
        self.viewer = None
        self.show_text_overlay = True
        self.camera_tracking = False
        self.bridge = UnitreeBridge(self.model, self.data, config.bridge_interface) if config.bridge else None
        self.mqtt_publisher: MqttStatePublisher | None = None
        self._mqtt_period = (
            1.0 / config.mqtt.frequency
            if config.mqtt is not None and config.mqtt.frequency is not None else None
        )
        self._next_mqtt_publish: float | None = None
        self._state_lock = threading.RLock()
        self._force_queue: deque[tuple[ForceCommand, ...]] = deque()
        self._active_forces: list[ActiveForce] = []
        self.reset()

    def reset(self) -> None:
        with self._state_lock:
            self._force_queue.clear()
            self._active_forces.clear()
            self.mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.qpos_ids] = self.targets
            # The first joint is the floating base in the bundled model.
            self.data.qpos[self.free_qpos : self.free_qpos + 3] = (0.0, 0.0, 0.82)
            self.data.qpos[self.free_qpos + 3 : self.free_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
            self.mujoco.mj_forward(self.model, self.data)
            self.virtual_gantry.set_position_to_robot(self.data.xpos[self.root_body_id])

    def _enqueue_forces(self, commands: tuple[ForceCommand, ...]) -> None:
        """Called by the MQTT thread; only the simulation thread applies forces."""
        with self._state_lock:
            self._force_queue.append(commands)

    def _apply_forces(self) -> None:
        """Rebuild world-frame forces at body centers of mass before each step."""
        now = float(self.data.time)
        self._active_forces = [
            force for force in self._active_forces
            if now < force.end_time
            and not math.isclose(now, force.end_time, rel_tol=0.0, abs_tol=1e-12)
        ]
        while self._force_queue:
            commands = self._force_queue.popleft()
            batch = []
            for command in commands:
                body_id = self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_BODY, command.body_name,
                )
                if body_id <= 0:
                    LOG.warning("Ignoring MQTT force batch: invalid body_name %r", command.body_name)
                    break
                end_time = now + command.duration
                if not math.isfinite(end_time):
                    LOG.warning("Ignoring MQTT force batch: duration overflows simulation time")
                    break
                batch.append(ActiveForce(body_id, command.force, end_time))
            else:
                self._active_forces.extend(batch)
        self.data.xfrc_applied[:] = 0.0
        self._apply_gantry()
        for force in self._active_forces:
            self.data.xfrc_applied[force.body_id, :3] += force.force

    def _apply_gantry(self) -> None:
        if not self.virtual_gantry.enabled:
            return
        # Holosoma computes the band from the floating root state, then applies
        # that force at the selected attachment body (torso_link for G1).
        position = self.data.xpos[self.root_body_id]
        # cvel stores angular then linear velocity for a body.
        velocity = self.data.cvel[self.root_body_id, 3:6]
        self.data.xfrc_applied[self.gantry_body_id, :3] += self.virtual_gantry.advance(position, velocity)

    def _control(self) -> None:
        if self.bridge is not None:
            self.bridge.step(self.model.opt.timestep)
        elif self.config.control == "stand":
            q = self.data.qpos[self.qpos_ids]
            dq = self.data.qvel[self.qvel_ids]
            torque = self.kp * (self.targets - q) - self.kd * dq
            self.data.ctrl[:] = np.clip(torque, self.control_lower, self.control_upper)
        else:
            self.data.ctrl[:] = 0.0

    def run(self) -> None:
        try:
            if self.config.mqtt is not None:
                self.mqtt_publisher = MqttStatePublisher(self.config.mqtt, on_forces=self._enqueue_forces)
                self._next_mqtt_publish = time.perf_counter()
            if self.config.headless:
                self._run_loop(None)
                return
            try:
                import mujoco.viewer
            except ImportError as exc:
                raise RuntimeError("The installed MuJoCo package has no viewer support") from exc
            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            ) as viewer:
                self.viewer = viewer
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -15.0
                self._update_text_overlay()
                self._run_loop(viewer)
            self.viewer = None
        finally:
            if self.mqtt_publisher is not None:
                self.mqtt_publisher.close()
                self.mqtt_publisher = None

    def _maybe_publish_mqtt(self, now: float | None = None) -> bool:
        """Publish at most one sample and advance the wall-clock deadline."""
        if self.mqtt_publisher is None or self._mqtt_period is None:
            return False
        now = time.perf_counter() if now is None else now
        if self._next_mqtt_publish is None:
            self._next_mqtt_publish = now
        if now < self._next_mqtt_publish:
            return False
        qpos = self.data.qpos
        self.mqtt_publisher.publish_state(
            simulation_time=float(self.data.time),
            joint_names=JOINT_NAMES,
            joint_values=qpos[self.qpos_ids].copy(),
            base_xyz=qpos[self.free_qpos : self.free_qpos + 3].copy(),
            base_quat_wxyz=qpos[self.free_qpos + 3 : self.free_qpos + 7].copy(),
        )
        missed_periods = int((now - self._next_mqtt_publish) // self._mqtt_period)
        self._next_mqtt_publish += (missed_periods + 1) * self._mqtt_period
        return True

    def _key_callback(self, keycode: int) -> None:
        """Handle the same gantry/viewer keys as holosoma's MuJoCo runner."""
        if keycode == 71:  # G: help overlay
            self.show_text_overlay = not self.show_text_overlay
            LOG.info("Text overlay: %s", "ON" if self.show_text_overlay else "OFF")
            self._update_text_overlay()
            return
        if keycode == 89:  # Y: camera tracking
            self.camera_tracking = not self.camera_tracking
            LOG.info("Camera tracking: %s (press 'Y' to toggle)", "ON" if self.camera_tracking else "OFF")
            self._update_text_overlay()
            return
        if keycode == 259:  # Backspace: reset simulation and gantry anchor
            self.reset()
            LOG.info("Simulation reset")
            self._update_text_overlay()
            return
        if self.virtual_gantry.handle_key(keycode):
            self._update_text_overlay()

    def _update_text_overlay(self) -> None:
        if self.viewer is None:
            return
        if not self.show_text_overlay:
            self.viewer.clear_texts()
            return
        gantry = self.virtual_gantry
        text = (
            f"Virtual gantry is {'active' if gantry.enabled else 'inactive'}\n"
            f"Length: {gantry.length:.2f} m | apply force: {gantry.apply_force:.0f} N\n"
            "[7] raise (-0.1 m)   [8] lower (+0.1 m)\n"
            "[9] toggle           [0] adjust force   [-] reverse force step\n"
            f"Camera tracking: {'ON' if self.camera_tracking else 'OFF'}   [Y] toggle\n"
            "[Backspace] reset    [G] hide this menu"
        )
        self.viewer.set_texts((None, None, text, ""))

    def _draw_gantry(self, viewer) -> None:
        """Draw the red anchor and blue elastic line used by holosoma."""
        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            if not self.virtual_gantry.enabled or scene.maxgeom < 2:
                return
            self.mujoco.mjv_initGeom(
                scene.geoms[0], self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.1, 0.0, 0.0]), self.virtual_gantry.point,
                np.eye(3).ravel(), np.array([1.0, 0.0, 0.0, 0.7], dtype=np.float32),
            )
            start = self.virtual_gantry.point
            end = self.data.xpos[self.root_body_id]
            delta = end - start
            length = np.linalg.norm(delta)
            if length >= 1e-6:
                midpoint = (start + end) / 2.0
                direction = delta / length
                z_axis = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(direction, z_axis))) > 0.999:
                    rotation = np.eye(3) if direction[2] > 0 else np.diag([-1.0, -1.0, -1.0])
                else:
                    axis = np.cross(z_axis, direction)
                    axis /= np.linalg.norm(axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                    cross = np.array([
                        [0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ])
                    rotation = (
                        np.cos(angle) * np.eye(3) + np.sin(angle) * cross
                        + (1 - np.cos(angle)) * np.outer(axis, axis)
                    )
                self.mujoco.mjv_initGeom(
                    scene.geoms[1], self.mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.array([0.002, length / 2.0, 0.0]), midpoint,
                    rotation.ravel(), np.array([0.0, 0.0, 1.0, 0.8], dtype=np.float32),
                )
                scene.ngeom = 2
            else:
                scene.ngeom = 1

    def _run_loop(self, viewer) -> None:
        LOG.info(
            "Running G1 at %d Hz on %s terrain%s",
            self.config.fps, self.config.terrain.kind, " with Unitree bridge" if self.bridge else "",
        )
        step = 0
        sync_interval = max(1, round(self.config.fps / self.config.viewer_fps))
        next_step = time.perf_counter()
        while self.config.steps <= 0 or step < self.config.steps:
            if viewer is not None and not viewer.is_running():
                break
            with self._state_lock:
                self._control()
                self._apply_forces()
                self.mujoco.mj_step(self.model, self.data)
                self._maybe_publish_mqtt()
            step += 1
            if viewer is not None and step % sync_interval == 0:
                if self.camera_tracking:
                    viewer.cam.lookat[:] = self.data.xpos[self.root_body_id]
                self._draw_gantry(viewer)
                viewer.sync()
            if self.config.realtime:
                next_step += self.model.opt.timestep
                delay = next_step - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    next_step = time.perf_counter()
        LOG.info("Simulation stopped after %d steps (sim time %.3fs)", step, self.data.time)
