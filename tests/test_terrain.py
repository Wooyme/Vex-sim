from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from g1_mujoco_sim.config import TerrainConfig
from g1_mujoco_sim.terrain import build_world


ROBOT_XML = Path(__file__).parents[1] / "g1_mujoco_sim/assets/g1/g1_29dof.xml"
MESH_DIR = ROBOT_XML.parent / "meshes"


def test_plane_is_composed_without_hfield():
    world = build_world(ROBOT_XML, MESH_DIR, TerrainConfig(), 0.001)
    root = ET.fromstring(world.xml)
    assert root.find("worldbody/geom[@name='floor']").attrib["type"] == "plane"
    assert world.heightfield is None


def test_mix_has_normalized_heightfield_and_flat_spawn():
    cfg = TerrainConfig(kind="mix", size=4.0, resolution=0.1, seed=3)
    world = build_world(ROBOT_XML, MESH_DIR, cfg, 0.001)
    assert world.heightfield is not None
    assert world.heightfield.min() >= 0.0
    assert world.heightfield.max() <= 1.0
    assert float(world.heightfield[world.heightfield.shape[0] // 2, world.heightfield.shape[1] // 2]) >= 0.0


def test_mesh_requires_existing_supported_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_world(
            ROBOT_XML, MESH_DIR,
            TerrainConfig(kind="mesh", file=tmp_path / "missing.obj"), 0.001,
        )
