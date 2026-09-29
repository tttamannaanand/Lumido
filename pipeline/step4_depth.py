"""
STEP 4 - Depth maps with our own multi-view plane-sweep stereo.

Idea (plane sweep):
  For a reference image, imagine a stack of flat planes parallel to the image,
  at many depths between "near" and "far". For each candidate depth d:
    - warp every neighbour image onto the reference image AS IF the whole scene
      sat on the plane at depth d (a plane-to-plane mapping is a homography).
    - where the guess d is correct for a pixel, the warped neighbour lines up
      with the reference there, so the local image patches look the same.
  We measure "looks the same" with ZNCC (zero-mean normalised cross-correlation)
  over a small window: it's +1 for identical patches and ignores brightness
  differences between photos. Each pixel's depth is the plane with the best score.

Details:
  - Images are first undistorted (remove lens distortion k1 found in step 3),
    because the homography maths assumes a perfect pinhole camera.
  - Neighbours: for each image we pick N_NEIGHBOURS images that share many
    sparse 3D points with it AND view them from a useful angle (too small an
    angle = depth is poorly defined; too large = patches look too different).
  - Depth range comes from the sparse SfM points seen in that image.
  - Planes are spaced evenly in 1/depth, which gives finer steps nearby.
  - Occlusions: a point may be hidden in some neighbours, so per plane we
    average only the BEST_K best neighbour scores, not all of them.
  - Sub-plane refinement: fit a parabola through the cost at the best plane and
    its two neighbours to get a depth between planes.
  - Pixels are rejected if they have no texture (sky, flat water) or a weak best
    score. Step 5 then keeps only depths that several views agree on.

Output:
  data/depth/<image>.npz     depth (metres, 0 = invalid) + matching cost
  data/depth/scene.npz       cameras, poses, neighbours (used by step 5)
  outputs/depth_maps/*.jpg   photo + coloured depth map side by side

Run:  .venv/bin/python pipeline/step4_depth.py            (all images)
      .venv/bin/python pipeline/step4_depth.py 0287_0020  (just one, for testing)
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
MODEL_DIR = ROOT / "data" / "sfm" / "model"
DEPTH_DIR = ROOT / "data" / "depth"
VIS_DIR = ROOT / "outputs" / "depth_maps"

DEPTH_SCALE = 0.75       # compute depth at 75% of the 1600 px working size (1200 x 900)
N_PLANES = 160           # number of depth hypotheses
N_NEIGHBOURS = 5         # neighbour images used per reference image
BEST_K = 2               # average the best K neighbour scores per plane (occlusion handling)
WINDOW = 7               # ZNCC window size in pixels
MIN_ANGLE, MAX_ANGLE = 3.0, 45.0   # accepted viewing-angle range for neighbours (degrees)
MAX_COST = 0.5           # reject pixels whose best cost (1 - ZNCC) is above this
MIN_TEXTURE = 4.0        # reject pixels whose local intensity std-dev is below this (grey levels)
SPECKLE_AREA = 600       # connected depth regions smaller than this (pixels) are removed
SPECKLE_PLANES = 2       # neighbouring pixels belong to one region if within this many planes
N_WORKERS = 4            # images processed in parallel (limited by 8 GB RAM)


# ------------------------------------------------------------ scene set-up ---
def load_scene():
    rec = pycolmap.Reconstruction(str(MODEL_DIR))
    images = sorted(rec.images.values(), key=lambda im: im.name)
    cam = rec.cameras[images[0].camera_id]
    f, cx, cy, k1 = cam.params
    # COLMAP puts pixel centres at +0.5; OpenCV at integer coordinates -> shift by 0.5
    K_full = np.array([[f, 0, cx - 0.5], [0, f, cy - 0.5], [0, 0, 1]])
    s = DEPTH_SCALE
    K = np.array([[f * s, 0, cx * s - 0.5], [0, f * s, cy * s - 0.5], [0, 0, 1]])

    names = [im.name for im in images]
    R = np.array([im.cam_from_world().rotation.matrix() for im in images])
    t = np.array([im.cam_from_world().translation for im in images])
    centres = np.array([im.projection_center() for im in images])
    id_to_idx = {im.image_id: k for k, im in enumerate(images)}

    # Which sparse 3D points does each image see?
    seen = [set() for _ in images]
    for pid, p in rec.points3D.items():
        for el in p.track.elements:
            seen[id_to_idx[el.image_id]].add(pid)
    xyz = {pid: p.xyz for pid, p in rec.points3D.items()}

    neighbours, depth_range = [], []
    for r in range(len(images)):
        pts = np.array([xyz[p] for p in seen[r]])
        # depth range from sparse points in front of this camera
        z = (pts @ R[r].T + t[r])[:, 2]
        z = z[z > 0]
        depth_range.append((0.8 * np.percentile(z, 2), 1.25 * np.percentile(z, 98)))

        # score every other image: shared points, if the viewing angle is useful
        scores = []
        for n in range(len(images)):
            if n == r:
                continue
            shared = seen[r] & seen[n]
            if len(shared) < 50:
                continue
            centre_pt = np.median(np.array([xyz[p] for p in shared]), axis=0)
            a, b = centres[r] - centre_pt, centres[n] - centre_pt
            angle = np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
            if MIN_ANGLE <= angle <= MAX_ANGLE:
                scores.append((len(shared), n))
        neighbours.append([n for _, n in sorted(scores, reverse=True)[:N_NEIGHBOURS]])

    return dict(names=names, K=K, K_full=K_full, k1=k1, R=R, t=t,
                neighbours=neighbours, depth_range=np.array(depth_range))


def load_undistorted(name, scene, colour=False):
    img = cv2.imread(str(IMG_DIR / name), cv2.IMREAD_COLOR if colour else cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(img, scene["K_full"], np.array([scene["k1"], 0, 0, 0]))
    img = cv2.resize(img, None, fx=DEPTH_SCALE, fy=DEPTH_SCALE, interpolation=cv2.INTER_AREA)
    return img if colour else img.astype(np.float32)


# ------------------------------------------------------------ plane sweep ---
def plane_sweep(r, scene):
    K, Kinv = scene["K"], np.linalg.inv(scene["K"])
    ref = load_undistorted(scene["names"][r], scene)
    h, w = ref.shape
    box = lambda x: cv2.boxFilter(x, -1, (WINDOW, WINDOW))

    mean_r = box(ref)
    var_r = box(ref * ref) - mean_r ** 2
    textured = var_r > MIN_TEXTURE ** 2

    nbrs = scene["neighbours"][r]
    srcs = [load_undistorted(scene["names"][n], scene) for n in nbrs]
    # relative pose: X_neighbour = R_rel @ X_ref + t_rel
    rel = []
    for n in nbrs:
        R_rel = scene["R"][n] @ scene["R"][r].T
        t_rel = scene["t"][n] - R_rel @ scene["t"][r]
        rel.append((R_rel, t_rel))

    d_near, d_far = scene["depth_range"][r]
    inv_depths = np.linspace(1 / d_near, 1 / d_far, N_PLANES)  # even steps in 1/depth
    normal = np.array([0, 0, 1.0])                              # planes face the camera

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pix = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w, np.float32)])   # 3 x (h*w)

    cost_volume = np.empty((N_PLANES, h, w), np.float16)
    for p, inv_d in enumerate(inv_depths):
        costs = []
        for src, (R_rel, t_rel) in zip(srcs, rel):
            # Homography induced by the plane z = d:  H = K (R + t n^T / d) K^-1
            H = K @ (R_rel + np.outer(t_rel, normal) * inv_d) @ Kinv
            warped = cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            # pixels that map outside the neighbour image can't be scored
            q = H @ pix
            qx, qy = q[0] / q[2], q[1] / q[2]
            inside = ((q[2] > 0) & (qx >= 0) & (qx < w - 1) & (qy >= 0) & (qy < h - 1)).reshape(h, w)

            mean_w = box(warped)
            var_w = box(warped * warped) - mean_w ** 2
            cov = box(ref * warped) - mean_r * mean_w
            zncc = cov / np.sqrt(np.maximum(var_r * var_w, 1e-6))
            c = 1.0 - zncc                                   # 0 = perfect match, 2 = opposite
            c[~inside | (var_w < MIN_TEXTURE ** 2)] = 2.0
            costs.append(c)
        costs = np.sort(np.stack(costs), axis=0)[:BEST_K].mean(axis=0)
        cost_volume[p] = costs

    # winner-takes-all + parabola fit for a sub-plane estimate
    best = np.argmin(cost_volume, axis=0)
    best_cost = np.take_along_axis(cost_volume, best[None], 0)[0].astype(np.float32)
    b = np.clip(best, 1, N_PLANES - 2)
    c0 = np.take_along_axis(cost_volume, (b - 1)[None], 0)[0].astype(np.float32)
    c1 = np.take_along_axis(cost_volume, b[None], 0)[0].astype(np.float32)
    c2 = np.take_along_axis(cost_volume, (b + 1)[None], 0)[0].astype(np.float32)
    denom = c0 - 2 * c1 + c2
    offset = np.where(denom > 1e-6, 0.5 * (c0 - c2) / np.maximum(denom, 1e-6), 0.0)
    offset = np.clip(offset, -0.5, 0.5)
    step = inv_depths[1] - inv_depths[0]
    inv_depth = inv_depths[0] + (b + offset) * step
    depth = (1.0 / inv_depth).astype(np.float32)

    valid = textured & (best_cost < MAX_COST) & (best > 0) & (best < N_PLANES - 1)
    # ignore a thin frame: undistortion leaves black borders there
    m = 12
    valid[:m], valid[-m:], valid[:, :m], valid[:, -m:] = False, False, False, False

    # Speckle filter: real surfaces form large smooth regions, while noise
    # (moving water, glare) forms small isolated blobs. OpenCV's filterSpeckles
    # groups neighbouring pixels whose values differ by <= SPECKLE_PLANES planes
    # and wipes out groups smaller than SPECKLE_AREA pixels.
    INVALID = -16
    plane_img = np.where(valid, np.round((b + offset) * 16), INVALID).astype(np.int16)
    cv2.filterSpeckles(plane_img, INVALID, SPECKLE_AREA, SPECKLE_PLANES * 16)
    valid &= plane_img != INVALID
    depth[~valid] = 0
    return depth, best_cost


# ---------------------------------------------------------- visualisation ---
def save_visual(r, scene, depth):
    colour = load_undistorted(scene["names"][r], scene, colour=True)
    valid = depth > 0
    vis = np.zeros_like(colour)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)  # red = near
        vis[~valid] = 0
        cv2.putText(vis, f"{lo:.0f} m", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"red = near ... blue = far ({hi:.0f} m)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(VIS_DIR / scene["names"][r]), np.hstack([colour, vis]), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ------------------------------------------------------------------- main ---
SCENE = {}


def process(r):
    cv2.setNumThreads(2)
    t0 = time.time()
    depth, cost = plane_sweep(r, SCENE)
    name = SCENE["names"][r]
    np.savez_compressed(DEPTH_DIR / (Path(name).stem + ".npz"), depth=depth, cost=cost.astype(np.float16))
    save_visual(r, SCENE, depth)
    return name, (depth > 0).mean(), time.time() - t0


def main():
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("1) Loading SfM model, choosing neighbours and depth ranges")
    SCENE.update(load_scene())
    np.savez(DEPTH_DIR / "scene.npz", names=np.array(SCENE["names"]), K=SCENE["K"],
             R=SCENE["R"], t=SCENE["t"], depth_range=SCENE["depth_range"],
             neighbours=np.array([nb + [-1] * (N_NEIGHBOURS - len(nb)) for nb in SCENE["neighbours"]]))
    todo = list(range(len(SCENE["names"])))
    if len(sys.argv) > 1:  # test mode: only images whose name contains the argument
        todo = [k for k in todo if sys.argv[1] in SCENE["names"][k]]
    todo = [k for k in todo if len(SCENE["neighbours"][k]) >= 2]
    print(f"   {len(todo)} images to process ({time.time() - t0:.0f}s)")

    print(f"2) Plane sweep: {N_PLANES} planes x {N_NEIGHBOURS} neighbours per image")
    ctx = mp.get_context("fork")
    pool = ctx.Pool(min(N_WORKERS, len(todo)))
    for k, (name, frac, secs) in enumerate(pool.imap_unordered(process, todo), 1):
        print(f"   [{k}/{len(todo)}] {name}: {100 * frac:.0f}% pixels with depth ({secs:.0f}s)")
    pool.close()  # let workers exit normally (terminating them makes pycolmap print a crash trace)
    pool.join()
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
