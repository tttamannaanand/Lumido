"""
STEP 5 - Fuse all depth maps into one dense, coloured point cloud.

A single depth map is noisy. A 3D point we can trust should be seen at the
SAME place by several cameras. So for every reference image:
  1. Back-project each pixel with a depth into a 3D world point:
         X_cam   = depth * K^-1 [u, v, 1]
         X_world = R^T (X_cam - t)
  2. Project that world point into each neighbour camera and read the
     neighbour's own depth map at that pixel.
  3. If the neighbour's depth agrees within REL_TOL (e.g. 1%), that view
     "confirms" the point. We also back-project the neighbour's depth there.
  4. Keep the point only if at least MIN_CONFIRM neighbours confirm it, and
     place it at the AVERAGE of the confirming 3D positions (reduces noise).
Then all images' points are merged and cleaned:
  - crop to the survey area (removes far horizon / sky leftovers)
  - voxel merging: one averaged point per VOXEL-sized cube (removes duplicates,
    since the same surface is seen from many images); done in batches of 25
    images so it fits in 8 GB of RAM
  - statistical outlier removal: drop points unusually far from their neighbours

Output:
  outputs/dense_cloud.ply        the final point cloud (metres, Z up, 6 cm spacing)
  outputs/dense_cloud_lite.ply   15 cm version for quick viewing

Run:  .venv/bin/python pipeline/step5_fuse.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
MODEL_DIR = ROOT / "data" / "sfm" / "model"
DEPTH_DIR = ROOT / "data" / "depth"
OUT_FILE = ROOT / "outputs" / "dense_cloud.ply"
LITE_FILE = ROOT / "outputs" / "dense_cloud_lite.ply"   # smaller copy for quick viewing
LITE_VOXEL = 0.15

REL_TOL = 0.01        # neighbour depth must agree within 1 %
MIN_CONFIRM = 2       # number of neighbours that must confirm a point
VOXEL = 0.06          # metres: final point spacing
CROP_MARGIN = 25      # metres around the camera footprint to keep
OUTLIER_NB, OUTLIER_STD = 20, 2.0


def voxel_reduce(pts, cols):
    """Merge all points falling in the same VOXEL-sized cube into their average.
    Done in batches during fusion so memory stays small on an 8 GB laptop."""
    keys = np.floor(pts / VOXEL).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    out_p = np.stack([np.bincount(inverse, pts[:, i]) for i in range(3)], 1) / counts[:, None]
    out_c = np.stack([np.bincount(inverse, cols[:, i]) for i in range(3)], 1) / counts[:, None]
    return out_p.astype(np.float32), out_c.astype(np.float32)


def write_ply(path, pts, cols):
    """Binary PLY: x, y, z as 32-bit floats + red, green, blue as bytes (15 bytes per point).
    Readable by Open3D, CloudCompare, MeshLab, Blender."""
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(len(pts), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"] = pts
    rec["rgb"] = np.clip(np.asarray(cols) * 255, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def main():
    t0 = time.time()
    scene = np.load(DEPTH_DIR / "scene.npz")
    names, K, R, t = list(scene["names"]), scene["K"], scene["R"], scene["t"]
    neighbours = [[n for n in row if n >= 0] for row in scene["neighbours"]]
    Kinv = np.linalg.inv(K)

    # Lens parameters for undistorting the colour images (same as step 4)
    rec = pycolmap.Reconstruction(str(MODEL_DIR))
    cam = rec.cameras[next(iter(rec.cameras))]
    f, cx, cy, k1 = cam.params
    K_full = np.array([[f, 0, cx - 0.5], [0, f, cy - 0.5], [0, 0, 1]])

    print("1) Loading depth maps")
    depths = {}
    for k, name in enumerate(names):
        p = DEPTH_DIR / (Path(name).stem + ".npz")
        if p.exists():
            depths[k] = np.load(p)["depth"]
    h, w = next(iter(depths.values())).shape
    scale = w / cam.width
    print(f"   {len(depths)} depth maps of {w}x{h}")

    ys, xs = np.mgrid[0:h, 0:w]
    rays = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)]) .astype(np.float64)
    rays = (Kinv @ rays).T                                    # (h*w, 3) pixel -> camera ray

    def backproject(k, idx):
        """World points for pixels idx (flat indices) of image k."""
        X_cam = rays[idx] * depths[k].ravel()[idx, None]
        return (X_cam - t[k]) @ R[k]                         # = R^T (X_cam - t)

    print("2) Multi-view consistency check")
    all_pts, all_cols = [], []
    for done, (r, depth_r) in enumerate(depths.items(), 1):
        idx = np.flatnonzero(depth_r.ravel() > 0)
        if len(idx) == 0:
            continue
        X = backproject(r, idx)
        acc = X.copy()                     # running sum of confirming 3D positions
        count = np.zeros(len(idx), int)    # how many neighbours confirmed

        for n in neighbours[r]:
            if n not in depths:
                continue
            Xn = X @ R[n].T + t[n]         # point in neighbour camera coordinates
            z = Xn[:, 2]
            uvw = Xn @ K.T
            u = np.round(uvw[:, 0] / np.maximum(uvw[:, 2], 1e-9)).astype(int)
            v = np.round(uvw[:, 1] / np.maximum(uvw[:, 2], 1e-9)).astype(int)
            ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d_n = np.zeros(len(idx))
            d_n[ok] = depths[n][v[ok], u[ok]]
            ok &= d_n > 0
            ok &= np.abs(z - d_n) < REL_TOL * d_n
            if ok.any():
                acc[ok] += backproject(n, v[ok] * w + u[ok])
                count[ok] += 1

        keep = count >= MIN_CONFIRM
        pts = acc[keep] / (count[keep, None] + 1)

        colour = cv2.imread(str(IMG_DIR / names[r]))
        colour = cv2.undistort(colour, K_full, np.array([k1, 0, 0, 0]))
        colour = cv2.resize(colour, (w, h), interpolation=cv2.INTER_AREA)
        cols = colour.reshape(-1, 3)[idx[keep], ::-1] / 255.0   # BGR -> RGB

        all_pts.append(pts.astype(np.float32))
        all_cols.append(cols.astype(np.float32))
        if done % 25 == 0:  # squash duplicates every 25 images to save memory
            merged = voxel_reduce(np.concatenate(all_pts), np.concatenate(all_cols))
            all_pts, all_cols = [merged[0]], [merged[1]]
        if done % 20 == 0 or done == len(depths):
            total = sum(len(p) for p in all_pts)
            print(f"   {done}/{len(depths)} images, {total / 1e6:.1f} M points after merging duplicates "
                  f"({time.time() - t0:.0f}s)")

    pts, cols = voxel_reduce(np.concatenate(all_pts), np.concatenate(all_cols))
    del all_pts, all_cols
    print(f"   voxel merge ({VOXEL * 100:.0f} cm): {len(pts) / 1e6:.2f} M points")

    print("3) Cleaning")
    centres = np.array([-R[k].T @ t[k] for k in range(len(names))])
    lo = centres.min(axis=0) - CROP_MARGIN
    hi = centres.max(axis=0) + CROP_MARGIN
    inside = np.all((pts[:, :2] >= lo[:2]) & (pts[:, :2] <= hi[:2]), axis=1) & (pts[:, 2] < centres[:, 2].min())
    pts, cols = pts[inside], cols[inside]
    print(f"   crop to survey area: {inside.sum() / 1e6:.1f} M points kept")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    del pts, cols
    pcd, _ = pcd.remove_statistical_outlier(OUTLIER_NB, OUTLIER_STD)
    print(f"   outlier removal: {len(pcd.points) / 1e6:.2f} M points")
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    write_ply(OUT_FILE, pts, cols)
    lite = pcd.voxel_down_sample(LITE_VOXEL)
    write_ply(LITE_FILE, np.asarray(lite.points, np.float32), np.asarray(lite.colors, np.float32))

    ext = pts.max(axis=0) - pts.min(axis=0)
    print(f"\nSaved {OUT_FILE.relative_to(ROOT)}: {len(pts):,} points "
          f"({OUT_FILE.stat().st_size / 1e6:.0f} MB), extent {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} m")
    print(f"Saved {LITE_FILE.relative_to(ROOT)}: {len(lite.points):,} points "
          f"({LITE_FILE.stat().st_size / 1e6:.0f} MB)")
    print(f"Total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
