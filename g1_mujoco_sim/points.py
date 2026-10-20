"""Height PD and horizontal-velocity PID for a force application point.

Ported from holosoma's batched torch ``PointController`` to a single NumPy
controller; the control law and default gains are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

# Numerical denominator floor inherited from COLA; avoids division by zero
# when normalizing a zero force without changing ordinary force magnitudes.
_NORM_EPS = 1.0e-12


@dataclass(frozen=True)
class PointControllerCfg:
    """Height PD and horizontal-velocity PID parameters in SI units.

    Defaults are the translational settings from COLA's ``BarControllerCfg``.
    A point on an articulated body has no independent mass: these gains need
    tuning for the body's dynamics and contacts, rather than mass rescaling.
    """

    # Height loop: P in N/m, D in N/(m/s), force in N, and reference height
    # slew in m/s. No gravity feedforward is applied.
    height_kp: float = 80.0
    height_kd: float = 3.50
    height_force_limit: float = 300.0
    height_target_rate_limit: float = 0.20

    # Horizontal velocity loop: P in N/(m/s), I in N/m, D in N/(m/s^2).
    # Force limits bound the XY vector norm in N.
    velocity_kp: float = 30.0
    velocity_ki: float = 60.0
    velocity_kd: float = 0.10
    horizontal_force_limit: float = 100.0
    integral_force_limit: float = 15.0

    # Signal conditioning: derivative cutoff in Hz, per-axis velocity
    # reference slew in m/s^2, and integration deadband in m/s.
    derivative_cutoff_hz: float = 20.0
    velocity_target_slew_limit: float = 2.0
    velocity_error_deadband: float = 0.005


def _clamp_vector_norm(values: np.ndarray, limit: float) -> np.ndarray:
    magnitude = float(np.linalg.norm(values))
    return values * min(1.0, limit / max(magnitude, _NORM_EPS))


class PointController:
    """Compute a world-frame force from a point's measured position and velocity.

    The caller supplies the point's measured velocity and applies the force
    at that point; the physics engine handles the motion. This controller neither integrates a physical point
    state nor provides gravity compensation or orientation control.

    ``dt`` is the interval between calls to :meth:`compute`, not necessarily
    the physics timestep. Call :meth:`reset` with measured state before use.
    """

    def __init__(self, cfg: PointControllerCfg, *, dt: float) -> None:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("controller dt must be finite and positive")
        self.cfg = cfg
        self.dt = float(dt)
        self.height_reference = 0.0
        self.height_reference_velocity = 0.0
        self.velocity_reference_xy = np.zeros(2)
        self.velocity_integral_xy = np.zeros(2)
        self.filtered_acceleration_xy = np.zeros(2)
        self.previous_point_velocity_xy = np.zeros(2)
        self.velocity_i_force_xy = np.zeros(2)
        self.force_w = np.zeros(3)

    def reset(self, *, point_position_w: npt.ArrayLike, point_velocity_w: npt.ArrayLike) -> None:
        """Initialize references from measurements and clear integration and derivative history."""
        position = np.asarray(point_position_w, dtype=np.float64)
        velocity = np.asarray(point_velocity_w, dtype=np.float64)
        self.height_reference = float(position[2])
        self.height_reference_velocity = 0.0
        self.velocity_reference_xy = velocity[:2].copy()
        self.velocity_integral_xy = np.zeros(2)
        self.filtered_acceleration_xy = np.zeros(2)
        self.previous_point_velocity_xy = velocity[:2].copy()
        self.velocity_i_force_xy = np.zeros(2)
        self.force_w = np.zeros(3)

    def compute(
        self,
        *,
        requested_height_w: float,
        requested_velocity_xy_w: npt.ArrayLike,
        point_position_w: npt.ArrayLike,
        point_velocity_w: npt.ArrayLike,
    ) -> np.ndarray:
        """Advance references and return a new world-frame ``[3]`` force in N."""
        cfg = self.cfg
        position = np.asarray(point_position_w, dtype=np.float64)
        velocity = np.asarray(point_velocity_w, dtype=np.float64)
        requested_velocity = np.asarray(requested_velocity_xy_w, dtype=np.float64)

        maximum_height_delta = cfg.height_target_rate_limit * self.dt
        height_delta = float(np.clip(
            requested_height_w - self.height_reference, -maximum_height_delta, maximum_height_delta,
        ))
        self.height_reference += height_delta
        self.height_reference_velocity = height_delta / self.dt

        maximum_velocity_delta = cfg.velocity_target_slew_limit * self.dt
        self.velocity_reference_xy = self.velocity_reference_xy + np.clip(
            requested_velocity - self.velocity_reference_xy, -maximum_velocity_delta, maximum_velocity_delta,
        )

        height_p_force = cfg.height_kp * (self.height_reference - position[2])
        height_d_force = cfg.height_kd * (self.height_reference_velocity - velocity[2])
        height_force = float(np.clip(
            height_p_force + height_d_force, -cfg.height_force_limit, cfg.height_force_limit,
        ))

        raw_acceleration_xy = (velocity[:2] - self.previous_point_velocity_xy) / self.dt
        # Exact discrete pole of a first-order low-pass at the supplied call interval.
        alpha = math.exp(-2.0 * math.pi * cfg.derivative_cutoff_hz * self.dt)
        self.filtered_acceleration_xy = alpha * self.filtered_acceleration_xy + (1.0 - alpha) * raw_acceleration_xy
        self.previous_point_velocity_xy = velocity[:2].copy()

        velocity_error_xy = self.velocity_reference_xy - velocity[:2]
        integration_error_xy = np.where(
            np.abs(velocity_error_xy) < cfg.velocity_error_deadband, 0.0, velocity_error_xy,
        )
        velocity_p_force_xy = cfg.velocity_kp * velocity_error_xy
        # Derivative on measurement avoids a derivative kick on target changes.
        velocity_d_force_xy = -cfg.velocity_kd * self.filtered_acceleration_xy
        unsaturated_force_xy = velocity_p_force_xy + cfg.velocity_ki * self.velocity_integral_xy + velocity_d_force_xy

        unsaturated_norm = float(np.linalg.norm(unsaturated_force_xy))
        if unsaturated_norm >= cfg.horizontal_force_limit:
            force_direction = unsaturated_force_xy / max(unsaturated_norm, _NORM_EPS)
            outward_error = float(np.dot(integration_error_xy, force_direction))
            if outward_error > 0.0:
                # Conditional integration removes only the component that deepens saturation.
                integration_error_xy = integration_error_xy - outward_error * force_direction

        if cfg.velocity_ki > 0.0:
            self.velocity_integral_xy = self.velocity_integral_xy + integration_error_xy * self.dt
            self.velocity_i_force_xy = _clamp_vector_norm(
                cfg.velocity_ki * self.velocity_integral_xy, cfg.integral_force_limit,
            )
            self.velocity_integral_xy = self.velocity_i_force_xy / cfg.velocity_ki
        else:
            self.velocity_integral_xy = np.zeros(2)
            self.velocity_i_force_xy = np.zeros(2)
        horizontal_force_xy = _clamp_vector_norm(
            velocity_p_force_xy + self.velocity_i_force_xy + velocity_d_force_xy, cfg.horizontal_force_limit,
        )

        self.force_w = np.array([horizontal_force_xy[0], horizontal_force_xy[1], height_force])
        return self.force_w.copy()
