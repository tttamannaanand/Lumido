"""
Interactive 3D viewer for the live demo.
Mouse: left-drag = rotate, scroll = zoom, shift+drag = pan.  Keys: +/- point size, Q = quit.

Run:  .venv/bin/python pipeline/view.py                        (dense cloud)
      .venv/bin/python pipeline/view.py outputs/sparse_cloud.ply
"""

import sys
from pathlib import Path

import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs" / "dense_cloud.ply"
pcd = o3d.io.read_point_cloud(str(path))
print(f"{path}: {len(pcd.points):,} points")
o3d.visualization.draw_geometries(
    [pcd], window_name="Lumido - Lighthouse reconstruction", width=1600, height=1000
)
