"""Optional Unitree low-level DDS bridge."""

from __future__ import annotations

import sys

import numpy as np


class UnitreeBridge:
    """Small G1-only adapter for the optional ``unitree_interface`` package."""

    def __init__(self, model, data, interface: str | None = None):
        try:
            from unitree_interface import LowState, MessageType, MotorCommand, RobotType, UnitreeInterface
        except ImportError as exc:
            raise RuntimeError(
                "Bridge support requires unitree_interface; install the project's bridge extra"
            ) from exc

        interface = interface or ("lo0" if sys.platform == "darwin" else "lo")
        self.model = model
        self.data = data
        self.interface = UnitreeInterface(interface, RobotType.G1, MessageType.HG)
        self.low_state = LowState(model.nu)
        self.low_cmd = MotorCommand(model.nu)
        self.joint_ids = model.actuator_trnid[:, 0].astype(int)
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = model.jnt_dofadr[self.joint_ids]
        self.torque_limits = np.abs(model.actuator_ctrlrange[:, 1])
        self.previous_velocity = np.zeros(model.nu)

    def step(self, dt: float) -> None:
        q = self.data.qpos[self.qpos_ids].copy()
        dq = self.data.qvel[self.qvel_ids].copy()
        ddq = (dq - self.previous_velocity) / dt
        self.previous_velocity[:] = dq

        self.low_state.motor.q = q.tolist()
        self.low_state.motor.dq = dq.tolist()
        self.low_state.motor.ddq = ddq.tolist()
        self.low_state.motor.tau_est = self.data.qfrc_actuator[self.qvel_ids].tolist()
        self.low_state.imu.quat = self.data.sensor("base_quat").data.copy().tolist()
        self.low_state.imu.omega = self.data.sensor("base_gyro").data.copy().tolist()
        self.low_state.imu.accel = self.data.sensor("base_accel").data.copy().tolist()
        self.low_state.tick = int(self.data.time * 1000)
        self.interface.publish_low_state(self.low_state)

        self.low_cmd = self.interface.read_incoming_command()
        if not self.low_cmd:
            self.data.ctrl[:] = 0.0
            return
        torque = (
            np.asarray(self.low_cmd.tau_ff)
            + np.asarray(self.low_cmd.kp) * (np.asarray(self.low_cmd.q_target) - q)
            + np.asarray(self.low_cmd.kd) * (np.asarray(self.low_cmd.dq_target) - dq)
        )
        self.data.ctrl[:] = np.clip(torque, -self.torque_limits, self.torque_limits)

