from contextlib import nullcontext

import numpy as np

from g1_mujoco_sim.config import SimulationConfig
from g1_mujoco_sim.gantry import KEY_0, KEY_7, KEY_8, KEY_9, KEY_MINUS, VirtualGantry
from g1_mujoco_sim.simulation import G1Simulation


def test_holosoma_keyboard_mapping_updates_all_gantry_state():
    gantry = VirtualGantry()

    assert gantry.handle_key(KEY_7)
    assert gantry.length == -0.1
    assert gantry.handle_key(KEY_8)
    assert gantry.length == 0.0

    assert gantry.handle_key(KEY_9)
    assert not gantry.enabled
    assert gantry.handle_key(KEY_9)
    assert gantry.enabled

    assert gantry.handle_key(KEY_0)
    assert gantry.apply_force == -10.0
    assert gantry.handle_key(KEY_MINUS)
    assert gantry.apply_force_sign == 1
    assert gantry.handle_key(KEY_0)
    assert gantry.apply_force == 0.0


def test_force_adjust_is_clamped_like_holosoma():
    gantry = VirtualGantry(apply_force=-95.0)
    for _ in range(5):
        gantry.handle_key(KEY_0)
    assert gantry.apply_force == -100.0


def test_gantry_spring_damper_equation():
    gantry = VirtualGantry(stiffness=200.0, damping=100.0, length=1.0)
    gantry.point = np.array([0.0, 0.0, 3.0])
    force = gantry.advance(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    np.testing.assert_allclose(force, [0.0, 0.0, 200.0])


def test_simulation_callback_and_gantry_visuals():
    sim = G1Simulation(SimulationConfig(headless=True, steps=1, realtime=False))

    class FakeViewer:
        def __init__(self):
            self.user_scn = sim.mujoco.MjvScene(sim.model, maxgeom=10)
            self.text = None

        def lock(self):
            return nullcontext()

        def set_texts(self, text):
            self.text = text

        def clear_texts(self):
            self.text = None

    viewer = FakeViewer()
    sim.viewer = viewer
    sim._update_text_overlay()
    assert "[7] raise" in viewer.text[2]

    sim._key_callback(KEY_7)
    assert sim.virtual_gantry.length == -0.1
    sim._draw_gantry(viewer)
    assert viewer.user_scn.ngeom == 2

    sim._key_callback(KEY_9)
    sim._draw_gantry(viewer)
    assert viewer.user_scn.ngeom == 0
