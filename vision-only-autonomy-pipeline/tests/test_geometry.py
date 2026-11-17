import cv2
import numpy as np

from vision_autonomy.geometry import BevSpec, GroundProjector, camera_rotation


def test_camera_rotation_is_orthonormal_and_pitches_down():
    R = camera_rotation(np.deg2rad(10.0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    forward = R @ np.array([0.0, 0.0, 1.0])
    assert forward[0] > 0.9 and forward[2] < 0.0


def test_pixel_to_ground_inverts_projection(projector):
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(2, 25, 100), rng.uniform(-4, 4, 100), np.zeros(100)])
    px = projector.vehicle_to_pixel(ground)
    back, valid = projector.pixel_to_ground(px)
    assert valid.all()
    assert np.allclose(back, ground[:, :2], atol=1e-6)


def test_range_of_known_ground_point(projector):
    px = projector.vehicle_to_pixel(np.array([[6.0, 8.0, 0.0]]))
    assert np.isclose(projector.pixel_range(px)[0], 10.0, atol=1e-6)


def test_pixels_above_horizon_do_not_hit_ground(projector):
    row = projector.horizon_row()
    pts, valid = projector.pixel_to_ground(np.array([[320.0, row - 5.0], [320.0, row + 5.0]]))
    assert not valid[0] and valid[1]
    assert np.isinf(projector.pixel_range(np.array([[320.0, row - 5.0]]))[0])


def test_one_pixel_error_grows_quadratically_with_range(projector):
    """Flat-ground ranging: the same pixel error costs much more range far away."""
    def sensitivity(x):
        px = projector.vehicle_to_pixel(np.array([[x, 0.0, 0.0]]))
        return abs(projector.pixel_range(px + [0.0, -1.0])[0] - x)
    assert sensitivity(20.0) > 3.5 * sensitivity(10.0)


def test_bev_homography_agrees_with_point_projection(projector):
    spec = BevSpec()
    H = projector.bev_homography(spec)
    cells = np.array([[10.0, 20.0], [120.0, 200.0], [230.0, 100.0]])
    hom = np.column_stack([cells, np.ones(3)]) @ H.T
    via_h = hom[:, :2] / hom[:, 2:]
    ground = spec.to_ground(cells)
    direct = projector.vehicle_to_pixel(np.column_stack([ground, np.zeros(3)]))
    assert np.allclose(via_h, direct, atol=1e-6)


def test_bev_cell_roundtrip_and_shape():
    spec = BevSpec(x_min=1.0, x_max=11.0, y_min=-5.0, y_max=5.0, resolution=0.1)
    assert spec.shape == (100, 100)
    pts = np.array([[3.0, -2.0], [10.5, 4.0]])
    assert np.allclose(spec.to_ground(spec.to_cell(pts)), pts)


def test_ipm_places_ground_marker_at_metric_position(projector):
    spec = BevSpec()
    cam = projector.camera
    image = np.zeros((cam.height, cam.width), dtype=np.uint8)
    marker = np.array([[6.0, 1.5, 0.0]])
    u, v = projector.vehicle_to_pixel(marker)[0]
    cv2.circle(image, (int(round(u)), int(round(v))), 4, 255, -1)
    bev = projector.warp_to_bev(image, spec)
    rows, cols = np.nonzero(bev > 128)
    est = spec.to_ground(np.array([[cols.mean(), rows.mean()]]))[0]
    assert np.linalg.norm(est - marker[0, :2]) < 0.25


def test_bev_valid_mask_excludes_cells_outside_field_of_view(projector):
    spec = BevSpec()
    mask = projector.bev_valid_mask(spec)
    near_corner = spec.to_cell(np.array([[1.6, 5.5]]))[0].astype(int)
    centre = spec.to_cell(np.array([[6.0, 0.0]]))[0].astype(int)
    assert not mask[near_corner[1], near_corner[0]]
    assert mask[centre[1], centre[0]]


def test_invalid_height_rejected(cameras):
    import pytest
    with pytest.raises(ValueError):
        GroundProjector(cameras[1], height=0.0, pitch=0.1)
