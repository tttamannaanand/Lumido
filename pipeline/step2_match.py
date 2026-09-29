"""
STEP 2 - Detect features and match them between images.

What it does:
  1. SIFT: finds up to N_FEATURES distinctive points ("keypoints") in every image
     and describes each one with a 128-number fingerprint ("descriptor").
     We convert descriptors to RootSIFT (L1-normalise, then square-root),
     which is known to match more reliably than plain SIFT.
  2. Pair selection: comparing all 200 images with each other = 19,900 pairs.
     Instead we use the GPS positions from step 1:
       - orbit flights (0285, 0287) all look at the lighthouse -> compare all of them
       - any other pair -> compare only if the drones were within PAIR_RADIUS_M
  3. Matching each pair:
       a. nearest-neighbour search on descriptors (FLANN, a fast approximate search)
       b. Lowe's ratio test: keep a match only if the best candidate is clearly
          better than the second-best (rejects ambiguous matches, e.g. on water)
       c. mutual check: A's best match is B, AND B's best match is A
       d. geometric check: fit an Essential matrix with RANSAC using the known
          camera focal length. Matches that don't agree with a single rigid
          camera motion are thrown away as outliers.
  4. Saves features, verified matches, stats and example pictures.

Output:
  data/features.npz   keypoints + descriptors for every image
  data/matches.npz    verified matches for every good image pair
  outputs/step2_*.jpg pictures of matches (for the presentation)

Run:  .venv/bin/python pipeline/step2_match.py
"""
import csv
import itertools
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
META_CSV = ROOT / "data" / "metadata.csv"
FEAT_FILE = ROOT / "data" / "features.npz"
MATCH_FILE = ROOT / "data" / "matches.npz"
OUT_DIR = ROOT / "outputs"

N_FEATURES = 8000       # max SIFT keypoints per image
PAIR_RADIUS_M = 70      # compare two photos if taken within this horizontal distance
ORBIT_FLIGHTS = {"0285", "0287"}
RATIO = 0.8             # Lowe's ratio test threshold
RANSAC_PX = 1.0         # max reprojection error (pixels) for a match to count as an inlier
MIN_INLIERS = 30        # pairs with fewer verified matches are dropped


# ---------------------------------------------------------------- metadata ---
def load_metadata():
    rows = list(csv.DictReader(open(META_CSV)))
    lat = np.array([float(r["lat"]) for r in rows])
    lon = np.array([float(r["lon"]) for r in rows])
    # Convert GPS degrees to metres East / North of the average position.
    # (A small area, so treating the Earth as flat here is accurate enough.)
    R_EARTH = 6378137.0
    east = np.radians(lon - lon.mean()) * R_EARTH * np.cos(np.radians(lat.mean()))
    north = np.radians(lat - lat.mean()) * R_EARTH
    return rows, np.stack([east, north], axis=1)


def select_pairs(rows, xy):
    pairs = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        both_orbit = rows[i]["flight"] in ORBIT_FLIGHTS and rows[j]["flight"] in ORBIT_FLIGHTS
        close = np.linalg.norm(xy[i] - xy[j]) < PAIR_RADIUS_M
        if both_orbit or close:
            pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------- features ---
def extract_features(names):
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    keypoints, descriptors = [], []
    for k, name in enumerate(names, 1):
        gray = cv2.imread(str(IMG_DIR / name), cv2.IMREAD_GRAYSCALE)
        kps, desc = sift.detectAndCompute(gray, None)
        # RootSIFT: L1-normalise then square-root each value
        desc = np.sqrt(desc / (desc.sum(axis=1, keepdims=True) + 1e-7)).astype(np.float32)
        # Keep x, y, size, angle of every keypoint
        keypoints.append(np.array([[p.pt[0], p.pt[1], p.size, p.angle] for p in kps], np.float32))
        descriptors.append(desc)
        if k % 20 == 0 or k == len(names):
            print(f"  features {k}/{len(names)}  ({len(kps)} keypoints in {name})")
    return keypoints, descriptors


# ---------------------------------------------------------------- matching ---
# These globals are filled in by the main process before the worker pool is
# forked, so every worker can read them without copying.
G = {}


def match_pair(pair):
    i, j = pair
    kp1, kp2 = G["kps"][i], G["kps"][j]
    d1, d2 = G["desc"][i], G["desc"][j]
    if len(d1) < MIN_INLIERS or len(d2) < MIN_INLIERS:
        return i, j, None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))

    # (a) + (b): two nearest neighbours each way, keep those passing the ratio test
    def ratio_matches(da, db):
        out = {}
        for m in flann.knnMatch(da, db, k=2):
            if len(m) == 2 and m[0].distance < RATIO * m[1].distance:
                out[m[0].queryIdx] = m[0].trainIdx
        return out

    fwd = ratio_matches(d1, d2)
    bwd = ratio_matches(d2, d1)
    # (c) mutual check
    m = np.array([(a, b) for a, b in fwd.items() if bwd.get(b) == a], dtype=np.int32)
    if len(m) < MIN_INLIERS:
        return i, j, None

    # (d) geometric verification with the Essential matrix
    p1, p2 = kp1[m[:, 0], :2], kp2[m[:, 1], :2]
    K = G["K"]
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC,
                                   prob=0.9999, threshold=RANSAC_PX)
    if E is None or mask is None:
        return i, j, None
    inliers = m[mask.ravel().astype(bool)]
    if len(inliers) < MIN_INLIERS:
        return i, j, None
    return i, j, inliers


def init_worker():
    cv2.setNumThreads(1)  # we parallelise over pairs instead


# ----------------------------------------------------------- visualisation ---
def draw_matches(name1, name2, kp1, kp2, matches, out_path, max_lines=300):
    img1 = cv2.imread(str(IMG_DIR / name1))
    img2 = cv2.imread(str(IMG_DIR / name2))
    canvas = np.hstack([img1, img2])
    w = img1.shape[1]
    rng = np.random.default_rng(0)
    pick = matches if len(matches) <= max_lines else matches[rng.choice(len(matches), max_lines, replace=False)]
    for a, b in pick:
        x1, y1 = kp1[a, :2]
        x2, y2 = kp2[b, :2]
        color = tuple(int(c) for c in rng.integers(64, 255, 3))
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w, int(y2)), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(canvas, (int(x2) + w, int(y2)), 3, color, -1)
    cv2.putText(canvas, f"{name1}  <->  {name2}   {len(matches)} verified matches",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.resize(canvas, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 90])


# -------------------------------------------------------------------- main ---
def main():
    OUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    rows, xy = load_metadata()
    names = [r["name"] for r in rows]

    f = float(rows[0]["focal_px"])
    w, h = int(rows[0]["width"]), int(rows[0]["height"])
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    print(f"1) Extracting SIFT features from {len(names)} images")
    kps, desc = extract_features(names)
    np.savez_compressed(FEAT_FILE, names=np.array(names),
                        **{f"kp_{k}": kps[k] for k in range(len(names))},
                        **{f"desc_{k}": desc[k].astype(np.float16) for k in range(len(names))})
    print(f"   done in {time.time() - t0:.0f}s")

    pairs = select_pairs(rows, xy)
    print(f"2) Selected {len(pairs)} image pairs out of {len(names) * (len(names) - 1) // 2} possible")

    print("3) Matching pairs (ratio test + mutual check + Essential-matrix RANSAC)")
    G.update(kps=kps, desc=desc, K=K)
    t1 = time.time()
    results = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=mp.cpu_count(), initializer=init_worker) as pool:
        for n, (i, j, inl) in enumerate(pool.imap_unordered(match_pair, pairs, chunksize=8), 1):
            if inl is not None:
                results[(i, j)] = inl
            if n % 500 == 0 or n == len(pairs):
                print(f"   {n}/{len(pairs)} pairs  ->  {len(results)} verified  ({time.time() - t1:.0f}s)")

    np.savez_compressed(MATCH_FILE,
                        pairs=np.array(list(results.keys()), dtype=np.int32).reshape(-1, 2),
                        **{f"m_{i}_{j}": v for (i, j), v in results.items()})

    # ---- stats: how well connected is each image? ----
    degree = np.zeros(len(names), int)
    for i, j in results:
        degree[i] += 1
        degree[j] += 1
    isolated = [names[k] for k in range(len(names)) if degree[k] == 0]
    print(f"\n4) Verified pairs: {len(results)}   "
          f"median matches per pair: {int(np.median([len(v) for v in results.values()]))}")
    for flight in ["0283", "0285", "0287"]:
        ks = [k for k, r in enumerate(rows) if r["flight"] == flight]
        print(f"   flight {flight}: avg {degree[ks].mean():.1f} matched partners per image, "
              f"{sum(degree[k] == 0 for k in ks)} images with none")
    if isolated:
        print(f"   {len(isolated)} images matched nothing (probably open water): "
              + ", ".join(n.split('_', 1)[1] for n in isolated))

    # ---- pictures: best pair inside each orbit flight, and best orbit<->nadir pair ----
    def best(filter_fn):
        cand = [(len(v), p) for p, v in results.items() if filter_fn(*p)]
        return max(cand)[1] if cand else None

    fl = [r["flight"] for r in rows]
    examples = {
        "step2_matches_orbit.jpg": best(lambda i, j: fl[i] == fl[j] == "0287"),
        "step2_matches_nadir.jpg": best(lambda i, j: fl[i] == fl[j] == "0283"),
        "step2_matches_cross_flight.jpg": best(lambda i, j: fl[i] != fl[j]),
    }
    for fname, p in examples.items():
        if p:
            draw_matches(names[p[0]], names[p[1]], kps[p[0]], kps[p[1]], results[p], OUT_DIR / fname)
            print(f"   saved outputs/{fname}")
    print(f"\nTotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
