"""Terrain generation and MJCF composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .config import TerrainConfig


@dataclass
class BuiltWorld:
    xml: str
    heightfield: np.ndarray | None = None


def _add_appearance(asset: ET.Element) -> None:
    ET.SubElement(asset, "texture", {
        "name": "terrain_checker", "type": "2d", "builtin": "checker",
        "rgb1": ".2 .3 .4", "rgb2": ".1 .2 .3", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "terrain_grid", "texture": "terrain_checker", "texrepeat": "8 8",
        "reflectance": ".15",
    })
    ET.SubElement(asset, "material", {
        "name": "terrain_solid", "rgba": ".32 .34 .36 1", "specular": ".2",
    })


def _generate_mix(cfg: TerrainConfig) -> np.ndarray:
    if cfg.size <= 0 or cfg.resolution <= 0:
        raise ValueError("Terrain size and resolution must be positive")
    count = max(9, round(cfg.size / cfg.resolution) + 1)
    rng = np.random.default_rng(cfg.seed)
    height = rng.normal(0.0, cfg.roughness, (count, count))
    for _ in range(4):
        height = (
            height + np.roll(height, 1, 0) + np.roll(height, -1, 0)
            + np.roll(height, 1, 1) + np.roll(height, -1, 1)
        ) / 5.0

    # Add sparse low platforms, matching the useful parts of holosoma's mixed terrain.
    for _ in range(max(4, count // 12)):
        width = int(rng.integers(2, max(3, count // 10)))
        row = int(rng.integers(1, max(2, count - width - 1)))
        col = int(rng.integers(1, max(2, count - width - 1)))
        height[row : row + width, col : col + width] += rng.uniform(0.03, 0.12)

    # Keep a stable spawn patch at the center.
    center = count // 2
    radius = max(2, round(0.65 / cfg.resolution))
    height[center - radius : center + radius + 1, center - radius : center + radius + 1] = 0.0
    return height.astype(np.float64)


def build_world(robot_xml: Path, mesh_dir: Path, cfg: TerrainConfig, timestep: float) -> BuiltWorld:
    """Inject the selected terrain into the bundled G1 MJCF."""
    root = ET.parse(robot_xml).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(mesh_dir.resolve()))

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{timestep:.12g}")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Invalid robot MJCF: {robot_xml}")
    _add_appearance(asset)

    friction = f"{cfg.friction:g} 0.005 0.001"
    common = {
        "contype": "2", "conaffinity": "1", "friction": friction,
        "solimp": ".99 .99 .01 .5 2", "solref": ".001 1",
    }
    heightfield = None

    if cfg.kind == "plane":
        ET.SubElement(worldbody, "geom", {
            **common, "name": "floor", "type": "plane", "size": "0 0 .05",
            "material": "terrain_grid", "pos": " ".join(map(str, cfg.position)),
        })
    elif cfg.kind == "mix":
        heightfield = _generate_mix(cfg)
        z_min = float(heightfield.min())
        z_range = max(float(np.ptp(heightfield)), 1e-6)
        normalized = (heightfield - z_min) / z_range
        rows, cols = normalized.shape
        ET.SubElement(asset, "hfield", {
            "name": "terrain_hfield", "nrow": str(rows), "ncol": str(cols),
            "size": f"{cfg.size / 2:g} {cfg.size / 2:g} {z_range:g} 0.01",
        })
        px, py, pz = cfg.position
        ET.SubElement(worldbody, "geom", {
            **common, "name": "floor", "type": "hfield", "hfield": "terrain_hfield",
            "material": "terrain_solid", "pos": f"{px:g} {py:g} {pz + z_min:g}",
        })
        heightfield = normalized
    elif cfg.kind == "mesh":
        if cfg.file is None:
            raise ValueError("Mesh terrain requires --terrain-file")
        terrain_file = cfg.file.expanduser().resolve()
        if not terrain_file.is_file():
            raise FileNotFoundError(f"Terrain file not found: {terrain_file}")
        if terrain_file.suffix.lower() not in {".obj", ".stl"}:
            raise ValueError("Terrain mesh must be an OBJ or STL file")
        ET.SubElement(asset, "mesh", {
            "name": "terrain_mesh", "file": str(terrain_file),
            "scale": f"{cfg.scale:g} {cfg.scale:g} {cfg.scale:g}",
        })
        ET.SubElement(worldbody, "geom", {
            **common, "name": "floor", "type": "mesh", "mesh": "terrain_mesh",
            "material": "terrain_solid", "pos": " ".join(map(str, cfg.position)),
        })
    else:
        raise ValueError(f"Unsupported terrain kind: {cfg.kind!r}")

    return BuiltWorld(ET.tostring(root, encoding="unicode"), heightfield)
