import json

import numpy as np
import pytest

from g1_mujoco_sim.config import SimulationConfig
from g1_mujoco_sim.forces import decode_forces
from g1_mujoco_sim.simulation import G1Simulation


def force(**changes):
    return dict(body_name="torso_link", direction=[1, 0, 0], magnitude=10, duration=0.02) | changes


def commands(*entries):
    return decode_forces(json.dumps({"forces": entries}))


def simulation(**changes):
    return G1Simulation(SimulationConfig(**(
        dict(headless=True, realtime=False, gantry=False, control="zero", fps=1000, steps=1) | changes
    )))


def test_batch_normalizes_direction_and_accepts_zero_magnitude():
    batch = commands(force(direction=[3, 4, 0]), force(magnitude=0))
    assert len(batch) == 2
    assert batch[0].force == (6, 8, 0)
    assert batch[1].force == (0, 0, 0)
    for scale in (1e-300, 1e300):
        np.testing.assert_allclose(commands(force(direction=[scale, scale, 0]))[0].force,
                                   [10 / np.sqrt(2), 10 / np.sqrt(2), 0])


@pytest.mark.parametrize("payload", [
    b"\xff", "{", "null", "[]", "{}", '{"forces": []}', '{"forces": {}}',
    '{"forces": [null]}', '{"forces": [{}]}',
])
def test_invalid_envelopes(payload):
    with pytest.raises(ValueError):
        decode_forces(payload)


@pytest.mark.parametrize("change", [
    {"body_name": ""}, {"body_name": 5}, {"body_name": "torso_link\x00suffix"},
    {"direction": [0, 0, 0]}, {"direction": [1, 0]}, {"direction": "1,0,0"},
    {"direction": [True, 0, 0]}, {"direction": [float("nan"), 0, 0]},
    {"magnitude": -1}, {"magnitude": "5"}, {"magnitude": True},
    {"magnitude": float("inf")}, {"magnitude": 10 ** 400},
    {"duration": 0}, {"duration": -1}, {"duration": None}, {"duration": float("nan")},
])
def test_invalid_entry_rejects_whole_batch(change):
    with pytest.raises(ValueError):
        commands(force(), force(**change))


def test_overlapping_forces_start_together_and_expire_independently():
    sim = simulation()
    sim.data.time = 2.0
    sim._enqueue_forces(commands(force(duration=0.002), force(body_name="pelvis", duration=0.004)))
    sim._apply_forces()
    torso, pelvis = sim.gantry_body_id, sim.root_body_id
    np.testing.assert_allclose(sim.data.xfrc_applied[[torso, pelvis], 0], [10, 10])
    assert [f.end_time for f in sim._active_forces] == [2.002, 2.004]
    sim.data.time = 2.001
    sim._enqueue_forces(commands(force(direction=[0, 1, 0], magnitude=20, duration=0.003)))
    sim._apply_forces()
    np.testing.assert_allclose(sim.data.xfrc_applied[torso, :3], [10, 20, 0])
    sim.data.time = 2.002
    sim._apply_forces()
    np.testing.assert_allclose(sim.data.xfrc_applied[torso, :3], [0, 20, 0])
    np.testing.assert_allclose(sim.data.xfrc_applied[pelvis, :3], [10, 0, 0])
    sim.data.time = 2.004
    sim._apply_forces()
    assert not sim.data.xfrc_applied.any()
    assert not sim._active_forces


@pytest.mark.parametrize("duration, expected_steps", [(0.0001, 1), (0.002, 2), (0.0025, 3)])
def test_duration_is_quantized_to_simulation_steps(duration, expected_steps):
    sim = simulation()
    sim._enqueue_forces(commands(force(duration=duration)))
    count = 0
    for _ in range(5):
        sim._apply_forces()
        count += bool(sim.data.xfrc_applied[sim.gantry_body_id, 0])
        sim.mujoco.mj_step(sim.model, sim.data)
    assert count == expected_steps


def test_world_force_does_not_rotate_with_body_and_changes_motion():
    pushed, baseline = simulation(), simulation()
    for sim in (pushed, baseline):
        sim.data.qpos[sim.free_qpos + 2] = 3.0  # no ground contact
        sim.data.qpos[sim.free_qpos + 3:sim.free_qpos + 7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        sim.mujoco.mj_forward(sim.model, sim.data)
    pushed._enqueue_forces(commands(force(body_name="pelvis")))
    pushed._run_loop(None)
    baseline._run_loop(None)
    np.testing.assert_allclose(pushed.data.xfrc_applied[pushed.root_body_id], [10, 0, 0, 0, 0, 0])
    # Compare total linear momentum, independent of internal joint motion.
    for sim in (pushed, baseline):
        sim.mujoco.mj_forward(sim.model, sim.data)
        sim.mujoco.mj_subtreeVel(sim.model, sim.data)
    dv = pushed.data.subtree_linvel[pushed.root_body_id] - baseline.data.subtree_linvel[baseline.root_body_id]
    assert dv[0] > 0
    assert abs(dv[1]) < dv[0] * 0.01


def test_gantry_force_is_added_once_and_survives_command_expiry():
    sim = simulation(gantry=True)
    sim._apply_forces()
    gantry = sim.data.xfrc_applied.copy()
    sim._enqueue_forces(commands(force(duration=0.001)))
    for _ in range(2):
        sim._apply_forces()
        expected = gantry.copy()
        expected[sim.gantry_body_id, 0] += 10
        np.testing.assert_allclose(sim.data.xfrc_applied, expected)
    sim.data.time = 0.001
    sim._apply_forces()
    np.testing.assert_allclose(sim.data.xfrc_applied, gantry)


@pytest.mark.parametrize("name", ["not_a_body", "world"])
def test_unknown_or_world_body_discards_batch_without_affecting_existing_force(name):
    sim = simulation()
    sim._enqueue_forces(commands(force(magnitude=3)))
    sim._apply_forces()
    sim._enqueue_forces(commands(force(), force(body_name=name)))
    sim._apply_forces()
    np.testing.assert_allclose(sim.data.xfrc_applied[sim.gantry_body_id, :3], [3, 0, 0])


def test_reset_removes_active_and_queued_forces():
    sim = simulation()
    sim._enqueue_forces(commands(force()))
    sim._apply_forces()
    sim._enqueue_forces(commands(force(body_name="pelvis")))
    sim.reset()
    sim._apply_forces()
    assert not sim._force_queue
    assert not sim._active_forces
    assert not sim.data.xfrc_applied.any()
