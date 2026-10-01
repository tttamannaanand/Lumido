"""
STEP 6 - Render views of the final point cloud (the "screenshot" deliverable).

A tiny point-cloud renderer written with NumPy (no GUI needed):
  1. Place a virtual camera: position + a point to look at ("look-at" camera).
  2. Project every 3D point into that camera with the pinhole model.
  3. Z-buffer: when several points land on the same pixel, the nearest wins.
     We draw far points first and near points last, so near ones overwrite.
  4. Each point is drawn as a small square so the surface looks solid.

Output:
  outputs/render_*.png        individual views
  outputs/render_overview.png all views in one image

Run:  .venv/bin/python pipeline/step6_render.py
For an interactive 3D viewer instead:  .venv/bin/python pipeline/view.py
"""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
CLOUD = ROOT / "outputs" / "dense_cloud.ply"
OUT_DIR = ROOT / "outputs"
W, H = 1920, 1080
BACKGROUND = (28, 28, 32)


def look_at(eye, target, up=(0, 0, 1)):
    """Rotation R and translation t so that X_cam = R X_world + t looks from eye to target."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye
    z = z / np.linalg.norm(z)  # camera looks along +z
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)  # +x = right
    y = np.cross(z, x)  # +y = down (image convention)
    R = np.stack([x, y, z])
    return R, -R @ eye


def render(pts, cols, eye, target, fov_deg=50, splat=2, up=(0, 0, 1)):
    R, t = look_at(eye, target, up)
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    Xc = pts @ R.T + t
    front = Xc[:, 2] > 0.5
    Xc, c = Xc[front], cols[front]
    u = (f * Xc[:, 0] / Xc[:, 2] + W / 2).astype(int)
    v = (f * Xc[:, 1] / Xc[:, 2] + H / 2).astype(int)
    on = (u >= 0) & (u < W - splat) & (v >= 0) & (v < H - splat)
    u, v, z, c = u[on], v[on], Xc[on, 2], c[on]

    order = np.argsort(-z)  # far first, near last
    u, v, c = u[order], v[order], (c[order] * 255).astype(np.uint8)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = BACKGROUND
    for dy in range(splat):
        for dx in range(splat):
            img[v + dy, u + dx] = c  # later (nearer) writes win
    return np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR for OpenCV


def main():
    pcd = o3d.io.read_point_cloud(str(CLOUD))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    print(f"Loaded {len(pts):,} points")

    # Centre of the scene = the densest area (robust to stray points)
    centre = np.median(pts, axis=0)
    ground = np.percentile(pts[:, 2], 5)
    target = np.array([centre[0], centre[1], ground + 5])

    # Lighthouse = the red roof: points that are clearly more red than green/blue.
    # Take the highest of them (the lantern cap) as the close-up target.
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    red = (r > 0.35) & (r > 1.6 * g) & (r > 1.6 * b)
    red_pts = pts[red]
    top_red = red_pts[red_pts[:, 2] > np.percentile(red_pts[:, 2], 99)]
    top = np.median(top_red, axis=0)
    print(
        f"   lighthouse top found at E {top[0]:.0f} m, N {top[1]:.0f} m, "
        f"height {top[2] - ground:.0f} m above ground"
    )

    views = {
        "oblique_south": dict(eye=target + [0, -150, 90], target=target),
        "oblique_east": dict(eye=target + [150, 20, 80], target=target),
        "oblique_north": dict(eye=target + [-30, 150, 85], target=target),
        "top_down": dict(eye=target + [0, 0, 300], target=target, fov_deg=55, up=(0, 1, 0)),
        "lighthouse_closeup": dict(
            eye=top + [-28, -38, 6], target=top - [0, 0, 10], fov_deg=50, splat=3
        ),
    }
    tiles = []
    for name, v in views.items():
        img = render(pts, cols, **v)
        cv2.putText(
            img,
            name.replace("_", " "),
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(OUT_DIR / f"render_{name}.png"), img)
        tiles.append(cv2.resize(img, (W // 2, H // 2)))
        print(f"   saved outputs/render_{name}.png")

    # 2 x 3 overview grid (last tile = info panel)
    info = np.full_like(tiles[0], BACKGROUND)
    lines = [
        "Lumido - Lighthouse reconstruction",
        f"{len(pts):,} points",
        "200 images, own SIFT matching,",
        "own plane-sweep depth maps + fusion",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            info,
            text,
            (30, 90 + 60 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    tiles.append(info)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(str(OUT_DIR / "render_overview.png"), grid)
    print("   saved outputs/render_overview.png")


if __name__ == "__main__":
    main()
