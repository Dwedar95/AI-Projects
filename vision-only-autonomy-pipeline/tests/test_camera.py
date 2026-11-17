import cv2
import numpy as np

from vision_autonomy.camera import (
    FisheyeCamera,
    PinholeCamera,
    Rectifier,
    build_remap,
    calibrate_pinhole,
    checkerboard_points,
    intrinsics_from_fov,
    look_at_target_pose,
    reprojection_rmse,
)


def _random_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(2, 8, n)])


def test_intrinsics_from_fov_matches_definition():
    K = intrinsics_from_fov(640, 480, 90.0)
    assert np.isclose(K[0, 0], 320.0)
    assert np.isclose(K[0, 2], 319.5) and np.isclose(K[1, 2], 239.5)


def test_pinhole_projection_matches_opencv():
    K = intrinsics_from_fov(640, 480, 80.0)
    dist = np.array([-0.21, 0.06, 0.001, -0.0015, 0.01])
    cam = PinholeCamera(K, 640, 480, dist)
    pts = _random_points()
    ref, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
    assert np.allclose(cam.project(pts), ref.reshape(-1, 2), atol=1e-6)


def test_pinhole_unproject_inverts_distortion():
    cam = PinholeCamera(intrinsics_from_fov(640, 480, 80.0), 640, 480,
                        np.array([-0.2, 0.05, 0.001, 0.001, 0.0]))
    pts = _random_points(seed=1)
    rays = cam.unproject(cam.project(pts))
    expected = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert np.allclose(rays, expected, atol=1e-6)


def test_fisheye_projection_matches_opencv():
    K = np.array([[380.0, 0, 319.5], [0, 380.0, 239.5], [0, 0, 1]])
    dist = np.array([-0.03, 0.004, -0.001, 0.0002])
    cam = FisheyeCamera(K, 640, 480, dist)
    pts = _random_points(seed=2)
    ref, _ = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    assert np.allclose(cam.project(pts), ref.reshape(-1, 2), atol=1e-6)


def test_fisheye_roundtrip_wide_angle():
    cam = FisheyeCamera(np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]]), 640, 480,
                        np.array([-0.02, 0.003, 0.0, 0.0]))
    angles = np.deg2rad(np.linspace(1, 80, 40))
    rays = np.column_stack([np.sin(angles), 0.3 * np.sin(angles), np.cos(angles)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    assert np.allclose(cam.unproject(cam.project(rays)), rays, atol=1e-7)


def test_points_behind_camera_are_nan():
    cam = PinholeCamera(intrinsics_from_fov(640, 480, 90.0), 640, 480)
    assert np.all(np.isnan(cam.project(np.array([[0.0, 0.0, -1.0]]))))


def test_rectifier_undoes_fisheye_distortion(cameras):
    raw, rect = cameras
    rng = np.random.default_rng(0)
    ideal = cv2.GaussianBlur(rng.integers(0, 255, (480, 640), dtype=np.uint8), (0, 0), 4)
    mx, my = build_remap(rect, raw)  # synthesise the fisheye view of the ideal image
    distorted = cv2.remap(ideal, mx, my, cv2.INTER_LINEAR)
    restored = Rectifier(raw, rect)(distorted)
    err = np.abs(restored.astype(float) - ideal.astype(float))[40:-40, 40:-40]
    assert err.mean() < 1.0
    # Without rectification the images disagree clearly.
    assert np.abs(distorted.astype(float) - ideal.astype(float))[40:-40, 40:-40].mean() > 3.0


def test_calibration_recovers_intrinsics_from_synthetic_views():
    true = PinholeCamera(intrinsics_from_fov(640, 480, 70.0), 640, 480,
                         np.array([-0.12, 0.03, 0.0, 0.0, 0.0]))
    target = checkerboard_points(9, 6, 0.05)
    rng = np.random.default_rng(3)
    obj, img, cam_pts = [], [], []
    for _ in range(12):
        rvec = rng.uniform(-0.45, 0.45, 3)
        tvec = np.array([rng.uniform(-0.3, 0.0), rng.uniform(-0.2, 0.0), rng.uniform(0.6, 1.1)])
        R, t = look_at_target_pose(rvec, tvec)
        pc = target @ R.T + t
        obj.append(target)
        img.append(true.project(pc) + rng.normal(0, 0.05, (len(target), 2)))
        cam_pts.append(pc)
    est, rms = calibrate_pinhole(obj, img, (640, 480))
    assert rms < 0.2
    assert abs(est.K[0, 0] - true.K[0, 0]) / true.K[0, 0] < 0.01
    assert abs(est.dist[0] - true.dist[0]) < 0.02
    assert reprojection_rmse(true, cam_pts[0], img[0]) < 0.15  # only the added noise
