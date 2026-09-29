"""
STEP 3 - Structure-from-Motion: recover where every camera was.

Input: our own keypoints + verified matches from step 2.
We use pycolmap ONLY as a backend library for the optimisation part
(incremental registration + bundle adjustment). No COLMAP GUI, no COLMAP
feature extraction or matching - those are ours from step 2.

What it does:
  1. Creates a COLMAP database and fills it with:
       - one camera model shared by all images (focal length from EXIF, step 1)
       - our SIFT keypoints
       - our verified matches
  2. Re-checks each pair's geometry (verify_matches) so the mapper knows what
     kind of motion links the two images.
  3. Incremental SfM:
       - picks a strong starting pair, triangulates the first 3D points
       - adds images one at a time by solving where each camera must be to see
         the known 3D points (PnP)
       - triangulates new points, and regularly runs bundle adjustment: a big
         least-squares fit that nudges all cameras + points to minimise
         reprojection error (how far each 3D point lands from where it was seen)
  4. SfM only knows shape up to an unknown scale. We fix that with GPS: find
     the rotation + translation + scale that best maps our camera centres
     onto the GPS positions. Result: coordinates in METRES, Z pointing up.
  5. Saves the model, a sparse point cloud and a camera-position plot.

Output:
  data/sfm/model/            cameras, images, points3D (COLMAP binary format)
  outputs/sparse_cloud.ply   sparse point cloud
  outputs/step3_cameras.png  top-down view of cameras + points

Run:  .venv/bin/python pipeline/step3_sfm.py
"""
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
META_CSV = ROOT / "data" / "metadata.csv"
FEAT_FILE = ROOT / "data" / "features.npz"
MATCH_FILE = ROOT / "data" / "matches.npz"
SFM_DIR = ROOT / "data" / "sfm"
DB_PATH = SFM_DIR / "database.db"
MODEL_DIR = SFM_DIR / "model"
OUT_DIR = ROOT / "outputs"


def gps_to_local_metres(rows):
    """GPS (lat, lon, alt) -> local East, North, Up metres around the mean position."""
    lat = np.array([float(r["lat"]) for r in rows])
    lon = np.array([float(r["lon"]) for r in rows])
    alt = np.array([float(r["alt"]) for r in rows])
    R_EARTH = 6378137.0
    east = np.radians(lon - lon.mean()) * R_EARTH * np.cos(np.radians(lat.mean()))
    north = np.radians(lat - lat.mean()) * R_EARTH
    return np.stack([east, north, alt], axis=1)


def build_database(rows, feats, matches):
    if SFM_DIR.exists():
        shutil.rmtree(SFM_DIR)
    SFM_DIR.mkdir(parents=True)
    pycolmap.Database.open(DB_PATH).close()  # creates the empty database file

    # 1) Images + one shared camera. Model SIMPLE_RADIAL = focal, cx, cy, k1 (lens distortion).
    f = float(rows[0]["focal_px"])
    w, h = int(rows[0]["width"]), int(rows[0]["height"])
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "SIMPLE_RADIAL"
    opts.camera_params = f"{f},{w / 2},{h / 2},0"
    pycolmap.import_images(DB_PATH, IMG_DIR, pycolmap.CameraMode.SINGLE,
                           image_names=[r["name"] for r in rows], options=opts)

    db = pycolmap.Database.open(DB_PATH)
    name_to_id = {img.name: img.image_id for img in db.read_all_images()}
    names = list(feats["names"])

    # 2) Our keypoints: [x, y, scale, angle]. COLMAP wants the angle in radians.
    for k, name in enumerate(names):
        kp = feats[f"kp_{k}"].copy()
        kp[:, 3] = np.radians(kp[:, 3])
        db.write_keypoints(name_to_id[name], np.ascontiguousarray(kp, dtype=np.float32))

    # 3) Our verified matches
    pair_lines = []
    for i, j in matches["pairs"]:
        m = np.ascontiguousarray(matches[f"m_{i}_{j}"], dtype=np.uint32)
        db.write_matches(name_to_id[names[i]], name_to_id[names[j]], m)
        pair_lines.append(f"{names[i]} {names[j]}")
    db.close()

    pairs_txt = SFM_DIR / "pairs.txt"
    pairs_txt.write_text("\n".join(pair_lines))
    return pairs_txt


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(META_CSV)))
    feats = np.load(FEAT_FILE)
    matches = np.load(MATCH_FILE)
    print(f"Loaded {len(feats['names'])} images, {len(matches['pairs'])} verified pairs")

    print("1) Building database from our features + matches")
    pairs_txt = build_database(rows, feats, matches)

    print("2) Geometric verification of pairs")
    pycolmap.verify_matches(DB_PATH, pairs_txt)

    print("3) Incremental Structure-from-Motion (this is the slow part)")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True     # let bundle adjustment fine-tune our EXIF focal
    opts.ba_refine_extra_params = True     # ...and the lens distortion
    opts.min_model_size = 10
    models = pycolmap.incremental_mapping(DB_PATH, IMG_DIR, SFM_DIR / "raw_models", opts)
    if not models:
        raise SystemExit("SfM failed: no model was reconstructed")
    # If the images split into several disconnected groups, keep the biggest
    rec = max(models.values(), key=lambda r: r.num_reg_images())
    print(f"   {len(models)} model(s); largest has {rec.num_reg_images()} images")

    print("4) Aligning to GPS (gives real-world scale in metres)")
    enu = gps_to_local_metres(rows)
    name_to_enu = {r["name"]: enu[k] for k, r in enumerate(rows)}
    reg_names = [img.name for img in rec.images.values()]
    ransac = pycolmap.RANSACOptions()
    ransac.max_error = 5.0  # metres; consumer drone GPS is only accurate to a few metres
    sim3 = pycolmap.align_reconstruction_to_locations(
        rec, reg_names, np.array([name_to_enu[n] for n in reg_names]),
        min_common_images=3, ransac_options=ransac)
    if sim3 is None:
        print("   WARNING: GPS alignment failed, keeping arbitrary scale")
    else:
        rec.transform(sim3)
        centres = {img.name: img.projection_center() for img in rec.images.values()}
        err = np.array([np.linalg.norm(centres[n] - name_to_enu[n]) for n in reg_names])
        print(f"   scale factor {sim3.scale:.3f}; camera-to-GPS distance "
              f"median {np.median(err):.2f} m, max {err.max():.2f} m")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rec.write(MODEL_DIR)
    rec.export_PLY(OUT_DIR / "sparse_cloud.ply")

    # ---- report ----
    cam = rec.cameras[next(iter(rec.cameras))]
    print("\n5) Result")
    print(f"   registered images : {rec.num_reg_images()} / {len(rows)}")
    for flight in ["0283", "0285", "0287"]:
        total = sum(r["flight"] == flight for r in rows)
        got = sum(n.split("_")[1] == flight for n in reg_names)
        print(f"      flight {flight}: {got}/{total}")
    print(f"   sparse 3D points  : {rec.num_points3D()}")
    print(f"   mean reprojection error: {rec.compute_mean_reprojection_error():.3f} px")
    print(f"   refined camera: focal {cam.params[0]:.1f} px (EXIF said {float(rows[0]['focal_px']):.1f}), "
          f"k1 {cam.params[3]:.4f}")
    missing = sorted(set(r["name"] for r in rows) - set(reg_names))
    if missing:
        print(f"   not registered ({len(missing)}): " + ", ".join(n.split('_', 1)[1] for n in missing))

    # ---- top-down plot ----
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cols = np.array([p.color for p in rec.points3D.values()]) / 255.0
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=0.5)
    colours = {"0283": "tab:blue", "0285": "tab:orange", "0287": "tab:red"}
    for flight, c in colours.items():
        cs = np.array([img.projection_center() for img in rec.images.values()
                       if img.name.split("_")[1] == flight])
        if len(cs):
            ax.scatter(cs[:, 0], cs[:, 1], c=c, s=25, marker="^", label=f"cameras {flight}")
    # Crop the view: angled photos also see far-off rocks on the horizon
    cam_xy = np.array([img.projection_center()[:2] for img in rec.images.values()])
    lo, hi = cam_xy.min(axis=0) - 15, cam_xy.max(axis=0) + 15
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Structure-from-Motion: {rec.num_reg_images()} cameras, {rec.num_points3D()} points")
    ax.legend()
    fig.savefig(OUT_DIR / "step3_cameras.png", dpi=130, bbox_inches="tight")
    print("   saved data/sfm/model, outputs/sparse_cloud.ply, outputs/step3_cameras.png")


if __name__ == "__main__":
    main()
