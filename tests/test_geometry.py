import math
from pathlib import Path

import numpy as np
import pytest
import step1_prepare
import step2_match
import step3_sfm
import step5_fuse
import step6_render

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


# --- step 1: EXIF ---------------------------------------------------------------


def test_dms_to_degrees_handles_hemispheres():
    assert step1_prepare.dms_to_degrees((51, 30, 36), "N") == pytest.approx(51.51)
    assert step1_prepare.dms_to_degrees((0, 7, 30), "W") == pytest.approx(-0.125)
    assert step1_prepare.dms_to_degrees((33, 52, 4.8), "S") == pytest.approx(-33.868)


@pytest.mark.parametrize("unit, res", [(2, 25.4 * 200), (3, 10 * 200), (4, 200)])
def test_focal_in_pixels_is_unit_independent(unit, res):
    # 200 px per mm on the sensor, 8.8 mm lens -> 1760 px, whatever unit EXIF uses
    cam = {"FocalPlaneResolutionUnit": unit, "FocalPlaneXResolution": res, "FocalLength": 8.8}
    assert step1_prepare.focal_in_pixels(cam, 5472) == pytest.approx(1760)


# --- steps 2-3: GPS to metres and pair selection ---------------------------------


def _gps_rows(points):
    return [{"lat": lat, "lon": lon, "alt": alt} for lat, lon, alt in points]


def test_gps_to_local_metres_matches_known_distances():
    # 0.001 deg of latitude ~ 111.3 m anywhere on Earth
    enu = step3_sfm.gps_to_local_metres(_gps_rows([(10.0, 76.0, 50), (10.001, 76.0, 60)]))
    assert enu[:, :2].mean(axis=0) == pytest.approx([0, 0], abs=1e-6)
    assert enu[1, 1] - enu[0, 1] == pytest.approx(111.32, rel=1e-3)
    assert enu[1, 2] - enu[0, 2] == pytest.approx(10)


def test_longitude_shrinks_with_latitude():
    at_equator = step3_sfm.gps_to_local_metres(_gps_rows([(0, 0, 0), (0, 0.001, 0)]))
    at_60 = step3_sfm.gps_to_local_metres(_gps_rows([(60, 0, 0), (60, 0.001, 0)]))
    span_eq = at_equator[1, 0] - at_equator[0, 0]
    span_60 = at_60[1, 0] - at_60[0, 0]
    assert span_60 == pytest.approx(span_eq * math.cos(math.radians(60)), rel=1e-3)


def test_select_pairs_uses_distance_and_orbit_flights():
    rows = [{"flight": "0100"}, {"flight": "0100"}, {"flight": "0285"}, {"flight": "0287"}]
    xy = np.array([[0, 0], [50, 0], [500, 0], [-500, 0]], float)
    pairs = step2_match.select_pairs(rows, xy)
    assert (0, 1) in pairs  # 50 m apart
    assert (2, 3) in pairs  # 1 km apart, but both orbit flights
    assert (0, 2) not in pairs


# --- step 5: fusion output --------------------------------------------------------


def test_voxel_reduce_merges_points_in_the_same_cube():
    v = step5_fuse.VOXEL
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [5.0, 5.0, 5.0]]) * v
    cols = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    out_p, out_c = step5_fuse.voxel_reduce(pts, cols)
    assert len(out_p) == 2
    merged = np.argmin(out_p[:, 0])
    assert out_p[merged] == pytest.approx([0.15 * v] * 3)
    assert out_c[merged] == pytest.approx([0.5, 0, 0.5])


def test_write_ply_round_trips(tmp_path):
    pts = np.array([[1.5, -2.0, 3.25], [0, 0, 0]], np.float32)
    cols = np.array([[1.0, 0.5, 0.0], [0, 0, 1.2]])  # 1.2 must clip to 255
    path = tmp_path / "c.ply"
    step5_fuse.write_ply(path, pts, cols)

    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n")
    assert b"element vertex 2" in header
    rec = np.frombuffer(body, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    assert rec["xyz"] == pytest.approx(pts)
    assert rec["rgb"].tolist() == [[255, 127, 0], [0, 0, 255]]


def test_committed_point_cloud_is_well_formed():
    path = OUTPUTS / "dense_cloud_lite.ply"
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
    n = int(header.split(b"element vertex ")[1].split(b"\n")[0])
    assert n > 3_000_000
    assert path.stat().st_size == len(header) + 15 * n


# --- step 6: renderer -------------------------------------------------------------


def test_look_at_points_the_camera_at_the_target():
    R, t = step6_render.look_at(eye=(0, -10, 5), target=(0, 0, 0))
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1)
    target_cam = R @ np.zeros(3) + t
    assert target_cam[:2] == pytest.approx([0, 0], abs=1e-9)
    assert target_cam[2] == pytest.approx(math.hypot(10, 5))


def test_render_z_buffer_keeps_the_nearest_point():
    W, H = step6_render.W, step6_render.H
    # Two points on the optical axis: red far, green near
    pts = np.array([[0, 20.0, 0], [0, 10.0, 0]])
    cols = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    img = step6_render.render(pts, cols, eye=(0, 0, 0), target=(0, 1, 0), splat=2)
    assert img.shape == (H, W, 3)
    assert img[H // 2, W // 2].tolist() == [0, 255, 0]  # BGR: green
    assert img[0, 0].tolist() == list(step6_render.BACKGROUND[::-1])


def test_render_drops_points_behind_the_camera():
    pts = np.array([[0, -10.0, 0]])
    img = step6_render.render(pts, np.ones((1, 3)), eye=(0, 0, 0), target=(0, 1, 0))
    assert (img == np.array(step6_render.BACKGROUND[::-1], np.uint8)).all()
