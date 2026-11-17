import numpy as np

from vision_autonomy.benchmark import make_cameras, run_closed_loop, run_open_loop
from vision_autonomy.egomotion import fit_rigid_2d, robust_rigid_2d
from vision_autonomy.geometry import BevSpec
from vision_autonomy.pipeline import STAGES, VisionPipeline
from vision_autonomy.planner import ArcRolloutPlanner, Obstacle, PlannerConfig, arc_points
from vision_autonomy.synthetic import (
    EgoPose,
    SceneConfig,
    SceneObject,
    SyntheticScene,
    integrate_ego,
)


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_rigid_fit_recovers_transform_and_rejects_outliers():
    rng = np.random.default_rng(0)
    src = rng.uniform(-3, 3, (200, 2))
    R, t = _rot(0.05), np.array([0.2, -0.03])
    dst = src @ R.T + t
    R_est, t_est = fit_rigid_2d(src, dst)
    assert np.allclose(R_est, R, atol=1e-9) and np.allclose(t_est, t, atol=1e-9)
    dst[:50] += rng.uniform(0.5, 1.0, (50, 2))  # 25 percent independently moving points
    R_rob, t_rob, inliers = robust_rigid_2d(src, dst)
    assert np.allclose(t_rob, t, atol=1e-6) and np.allclose(R_rob, R, atol=1e-6)
    assert inliers[50:].all() and not inliers[:50].any()


def test_integrate_ego_closes_a_circle():
    pose = EgoPose(0.0, 0.0, 0.0)
    steps = 100
    for _ in range(steps):
        pose = integrate_ego(pose, speed=2.0, curvature=0.5, dt=2 * np.pi / (2.0 * 0.5) / steps)
    assert abs(pose.x) < 1e-9 and abs(pose.y) < 1e-9


def test_scene_ground_truth_matches_projection_geometry(cameras, projector):
    raw, rect = cameras
    obj = SceneObject(1, "red", 1.0, 1.5, (10.0, 1.0))
    scene = SyntheticScene(rect, SceneConfig(objects=[obj], noise_sigma=0.0))
    frame = scene.render(0.0, EgoPose(0.0, 0.0, 0.0))
    truth = frame.objects[0]
    assert np.isclose(truth.range_m, np.hypot(10.0, 1.0))
    foot = np.array([[0.5 * (truth.box[0] + truth.box[2]), truth.box[3]]])
    assert abs(projector.pixel_range(foot)[0] - truth.range_m) < 0.05
    assert truth.visible_fraction == 1.0


def test_flow_egomotion_recovers_speed_and_yaw_rate(cameras):
    raw, rect = cameras
    scene = SyntheticScene(rect, SceneConfig(objects=[]), raw_camera=raw)
    pipe = VisionPipeline(raw, rect)
    pose, dt = EgoPose(3.0, 0.4, 0.05), 0.1
    for _ in range(6):
        result = pipe.step(scene.render(0.0, pose).image, dt)
        pose = integrate_ego(pose, speed=1.5, curvature=0.1, dt=dt)
    assert result.ego is not None and result.ego.valid
    assert abs(result.ego.speed - 1.5) < 0.12
    assert abs(result.ego.yaw_rate - 0.15) < 0.04


def test_traversability_separates_road_from_grass(cameras):
    raw, rect = cameras
    scene = SyntheticScene(rect, SceneConfig(objects=[]), raw_camera=raw)
    pipe = VisionPipeline(raw, rect)
    pose = EgoPose(0.0, 0.0, 0.0)
    for _ in range(4):
        result = pipe.step(scene.render(0.0, pose).image, 0.1)
        pose = integrate_ego(pose, 1.0, 0.0, 0.1)
    spec = pipe.cfg.bev
    trav = result.traversability
    road = spec.to_cell(np.array([[6.0, 0.0], [8.0, 2.0], [4.0, -2.0]])).astype(int)
    grass = spec.to_cell(np.array([[6.0, 5.0], [8.0, -5.0], [10.0, 5.5]])).astype(int)
    assert all(trav[r, c] > 0.8 for c, r in road)
    assert all(trav[r, c] < 0.2 for c, r in grass)
    assert set(result.timings_ms) == set(STAGES)


def test_arc_points_follow_circle():
    pts = arc_points(0.5, np.array([2.0 * np.pi]))  # half circle of radius 2
    assert np.allclose(pts[0], [0.0, 4.0], atol=1e-9)
    assert np.allclose(arc_points(0.0, np.array([3.0]))[0], [3.0, 0.0])


def _planner(**kw):
    return ArcRolloutPlanner(BevSpec(), PlannerConfig(**kw))


def test_planner_drives_straight_on_free_ground():
    planner = _planner()
    plan = planner.plan(np.ones(BevSpec().shape, dtype=np.float32), dt=0.1)
    assert plan.target_curvature == 0.0 and plan.curvature == 0.0
    assert plan.speed == planner.config.max_speed


def test_planner_steers_around_obstacle_and_away_from_blocked_side():
    spec = BevSpec()
    free = np.ones(spec.shape, dtype=np.float32)
    plan = _planner().plan(free, 0.1, obstacles=(Obstacle(5.0, 0.3, 0.4),))
    assert plan.target_curvature < 0.0  # obstacle slightly left, so pass on the right
    grid = free.copy()
    grid[:, : spec.shape[1] // 2 - 10] = 0.0  # everything to the left is not drivable
    grid[: 120, :] = np.minimum(grid[:120, :], 0.0)  # and a wall 7.5 m ahead
    grid[:120, spec.shape[1] // 2 + 40:] = 1.0       # with an opening on the right
    plan = _planner().plan(grid, 0.1)
    assert plan.target_curvature < 0.0


def test_planner_stops_when_everything_is_blocked():
    planner = _planner()
    plan = planner.plan(np.zeros(BevSpec().shape, dtype=np.float32), 0.1,
                        obstacles=(Obstacle(1.0, 0.0, 3.0),))
    assert plan.speed == 0.0
    assert not np.isfinite(plan.costs).any()


def test_planner_slew_rate_limits_curvature_change():
    planner = _planner(max_curvature_rate=0.5)
    free = np.ones(BevSpec().shape, dtype=np.float32)
    commands = [planner.plan(free, 0.1, obstacles=(Obstacle(4.0, 0.2, 0.5),)).curvature
                for _ in range(10)]
    steps = np.abs(np.diff([0.0, *commands]))
    assert steps.max() <= 0.05 + 1e-12
    assert abs(commands[-1]) > abs(commands[0])


def test_planner_hysteresis_prevents_flip_flop():
    spec = BevSpec()
    free = np.ones(spec.shape, dtype=np.float32)
    selections = {True: [], False: []}
    for use_hysteresis in (True, False):
        planner = _planner(hysteresis=0.08 if use_hysteresis else 0.0)
        for k in range(20):
            # An obstacle dead ahead, and noise that makes the left half look slightly
            # worse on even frames and the right half on odd frames.
            grid = free.copy()
            half = spec.shape[1] // 2
            if k % 2 == 0:
                grid[:, :half] = 0.9
            else:
                grid[:, half:] = 0.9
            plan = planner.plan(grid, 0.1, obstacles=(Obstacle(5.0, 0.0, 0.4),))
            selections[use_hysteresis].append(plan.selected)
    assert len(set(selections[True])) == 1
    assert len(set(selections[False])) == 2


def test_open_loop_benchmark_quality():
    report = run_open_loop(frames=70)
    assert report.online.mota > 0.85
    assert report.online.idf1 > 0.7
    assert report.online.precision > 0.95
    assert np.mean(report.range_rel_err) < 0.05
    assert np.mean(report.speed_err) < 0.15
    assert report.timings["total"]["mean_ms"] < 100.0


def test_closed_loop_stays_on_road_and_clear_of_obstacles():
    report = run_closed_loop(frames=110)
    assert report.on_road_ratio == 1.0
    assert report.min_obstacle_clearance_m > 0.5
    assert report.distance_m > 15.0
    assert report.max_curvature_step <= 0.05 + 1e-9


def test_make_cameras_share_image_size():
    raw, rect = make_cameras()
    assert raw.size == rect.size == (640, 480)
