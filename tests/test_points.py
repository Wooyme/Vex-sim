import json
import math

import numpy as np
import pytest

from g1_mujoco_sim.config import SimulationConfig
from g1_mujoco_sim.forces import ForceCommand, PointCommand, decode_commands
from g1_mujoco_sim.points import PointController, PointControllerCfg
from g1_mujoco_sim.simulation import G1Simulation


def controller(cfg=None, *, dt=1.0 / 200.0, position=(0.0, 0.0, 1.0), velocity=(0.0, 0.0, 0.0)):
    result = PointController(cfg or PointControllerCfg(), dt=dt)
    result.reset(point_position_w=position, point_velocity_w=velocity)
    return result


def compute(ctrl, position, velocity, *, height=None, velocity_xy=None):
    position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    return ctrl.compute(
        requested_height_w=position[2] if height is None else height,
        requested_velocity_xy_w=velocity[:2] if velocity_xy is None else velocity_xy,
        point_position_w=position,
        point_velocity_w=velocity,
    )


def point(**changes):
    return dict(body_name="pelvis", height=3.0, velocity=[0, 0], duration=0.02) | changes


def simulation(**changes):
    return G1Simulation(SimulationConfig(**(
        dict(headless=True, realtime=False, gantry=False, control="zero", fps=1000, steps=1) | changes
    )))


def floating(sim):
    sim.data.qpos[sim.free_qpos + 2] = 3.0  # no ground contact
    sim.mujoco.mj_forward(sim.model, sim.data)
    return sim


@pytest.mark.parametrize("env", [0, 1])
def test_translational_regression_against_cola_without_gravity(env):
    # Holosoma's recorded COLA BarController forces at 200 Hz, one environment per case.
    # The recorded vertical forces correspond to 10x the current default height gains.
    expected_forces = np.array([
        [[-0.885907145345338, 0.489604763563559, 7.41], [0.093302381781779, -1.282209527127116, -10.92]],
        [[-1.647234273998382, 0.895156182665589, 7.78], [0.143078091332794, -2.399312365331174, -11.26]],
        [[-2.344898518811611, 1.257265679207740, 8.11], [0.169632839603870, -3.432531358415480, -11.52]],
        [[-3.617398444705235, 2.203598963136822, -7.2], [0.789799481568412, -5.031197926273646, 3.9]],
        [[-4.882071649041738, 3.145714432694490, -8.55], [1.409357216347247, -6.618428865388984, 5.4]],
        [[-6.148167533542139, 4.089778355694756, -9.94], [2.031389177847380, -8.206556711389517, 6.98]],
    ])[:, env]
    position = np.array([[0.2, -0.1, 1.0], [0.1, 0.4, 0.8]])[env]
    velocity = np.array([[0.2, -0.1, 0.0], [-0.3, 0.4, 0.1]])[env]
    acceleration_step = np.array([[0.03, -0.02, 0.01], [-0.01, 0.04, -0.02]])[env]
    ctrl = controller(PointControllerCfg(height_kp=800.0, height_kd=35.0), position=position, velocity=velocity)
    for step, expected in enumerate(expected_forces):
        velocity = velocity + acceleration_step
        position = position + velocity * ctrl.dt
        height = (1.05, 0.75)[env] if step < 3 else (0.95, 0.85)[env]
        target_velocity = np.array([[0.5, -0.5], [-0.6, 0.7]])[env] * (1 if step < 3 else -1)
        force = compute(ctrl, position, velocity, height=height, velocity_xy=target_velocity)
        np.testing.assert_allclose(force, expected, atol=1e-12, rtol=0)


def test_stationary_target_has_no_feedforward_force():
    ctrl = controller()
    for _ in range(3):
        assert not compute(ctrl, [0, 0, 1], [0, 0, 0]).any()


def test_height_hold_uses_measured_vertical_velocity_and_limits_force():
    cfg = PointControllerCfg(height_kp=100.0, height_kd=10.0, height_force_limit=10.0)
    assert compute(controller(cfg), [0, 0, 0.9], [0, 0, 0.2], height=1.0)[2] == pytest.approx(8.0)
    assert compute(controller(cfg), [0, 0, 1.1], [0, 0, -0.2], height=1.0)[2] == pytest.approx(-8.0)
    assert compute(controller(cfg), [0, 0, 0.0], [0, 0, 0], height=1.0)[2] == 10.0


def test_saturation_blocks_outward_integral_and_recovers_without_windup():
    cfg = PointControllerCfg(velocity_kp=100.0, velocity_ki=10.0, velocity_kd=0.0,
                             horizontal_force_limit=1.0, velocity_target_slew_limit=1000.0)
    ctrl = controller(cfg, dt=0.01)
    for _ in range(20):
        force = compute(ctrl, [0, 0, 1], [0, 0, 0], velocity_xy=[10.0, 0.0])
        assert np.linalg.norm(force[:2]) == pytest.approx(1.0)
    assert not ctrl.velocity_integral_xy.any()
    assert not compute(ctrl, [0, 0, 1], [0, 0, 0]).any()


def test_integral_vector_limit_allows_unwinding():
    cfg = PointControllerCfg(velocity_kp=0.0, velocity_ki=10.0, velocity_kd=0.0, horizontal_force_limit=2.0,
                             integral_force_limit=2.0, velocity_target_slew_limit=1000.0)
    ctrl = controller(cfg, dt=0.1)
    target = np.array([0.6, 0.8])
    for _ in range(10):
        compute(ctrl, [0, 0, 1], [0, 0, 0], velocity_xy=target)
    np.testing.assert_allclose(ctrl.velocity_i_force_xy, target * 2.0)
    compute(ctrl, [0, 0, 1], [0, 0, 0], velocity_xy=-target)
    np.testing.assert_allclose(ctrl.velocity_i_force_xy, target)
    compute(ctrl, [0, 0, 1], [0, 0, 0], velocity_xy=-target)
    np.testing.assert_allclose(ctrl.velocity_i_force_xy, 0.0, atol=1e-12)


def test_deadband_applies_only_to_integration():
    for target, integrates in ((0.004, False), (0.006, True)):
        ctrl = controller()
        assert compute(ctrl, [0, 0, 1], [0, 0, 0], velocity_xy=[target, 0.0])[0] > 0.0
        assert (ctrl.velocity_i_force_xy[0] > 0.0) == integrates


@pytest.mark.parametrize("dt", [0.0, -0.01, math.inf, math.nan])
def test_invalid_call_interval_is_rejected(dt):
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        PointController(PointControllerCfg(), dt=dt)


def test_decode_mixed_batch():
    batch = decode_commands(json.dumps({
        "forces": [{"body_name": "torso_link", "direction": [0, 0, 2], "magnitude": 5, "duration": 1}],
        "points": [point()],
    }))
    assert isinstance(batch[0], ForceCommand) and batch[0].force == (0, 0, 5)
    assert batch[1] == PointCommand("pelvis", 3.0, (0, 0), 0.02)


@pytest.mark.parametrize("message", [
    {}, {"points": []}, {"points": [None]},
    {"points": [point(height=None)]}, {"points": [point(height=float("nan"))]},
    {"points": [point(velocity=[0, 0, 0])]}, {"points": [point(velocity=[True, 0])]},
    {"points": [point(duration=0)]}, {"points": [point(body_name="")]},
    {"points": [point()], "forces": []},
])
def test_invalid_point_message_rejects_whole_batch(message):
    with pytest.raises(ValueError):
        decode_commands(json.dumps(message))


def test_point_force_moves_body_toward_target_at_center_of_mass():
    sim = floating(simulation(steps=200))
    start = sim.data.xpos[sim.root_body_id].copy()
    sim._enqueue_forces(decode_commands(json.dumps({"points": [point(
        height=3.1, velocity=[0.3, 0], duration=1.0,
    )]})))
    sim._run_loop(None)
    sim._apply_forces()
    (active,) = sim._active_points
    force = sim.data.xfrc_applied[sim.root_body_id]
    np.testing.assert_allclose(force[:3], active.controller.force_w)
    assert not force[3:].any()
    assert sim.data.xpos[sim.root_body_id, 0] - start[0] > 0.002
    assert force[2] > 0


def test_repeated_point_command_retargets_existing_controller():
    sim = floating(simulation())
    sim._enqueue_forces(decode_commands(json.dumps({"points": [point(velocity=[0.5, 0])]})))
    sim._apply_forces()
    (first,) = sim._active_points
    controller_state = first.controller
    sim.data.time = 0.01
    sim._enqueue_forces(decode_commands(json.dumps({"points": [point(velocity=[-0.5, 0])]})))
    sim._enqueue_forces(decode_commands(json.dumps({"points": [point(body_name="torso_link", duration=0.05)]})))
    sim._apply_forces()
    assert len(sim._active_points) == 2
    assert sim._active_points[0].controller is controller_state
    np.testing.assert_allclose(sim._active_points[0].velocity, [-0.5, 0])
    assert sim._active_points[0].end_time == pytest.approx(0.03)
    sim.data.time = 0.03
    sim._apply_forces()
    assert [p.body_id for p in sim._active_points] == [sim.gantry_body_id]


def test_invalid_body_discards_mixed_batch_and_reset_clears_points():
    sim = floating(simulation())
    sim._enqueue_forces(decode_commands(json.dumps({
        "forces": [{"body_name": "torso_link", "direction": [1, 0, 0], "magnitude": 5, "duration": 1}],
        "points": [point(), point(body_name="world")],
    })))
    sim._apply_forces()
    assert not sim._active_forces and not sim._active_points
    sim._enqueue_forces(decode_commands(json.dumps({"points": [point()]})))
    sim._apply_forces()
    assert sim._active_points
    sim.reset()
    sim._apply_forces()
    assert not sim._active_points and not sim.data.xfrc_applied.any()
