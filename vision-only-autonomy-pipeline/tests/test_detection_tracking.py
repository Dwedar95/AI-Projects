import numpy as np

from vision_autonomy.detection import ColorBlobDetector, Detection
from vision_autonomy.kalman import BoxKalmanFilter, xyah_to_xyxy, xyxy_to_xyah
from vision_autonomy.tracker import (
    ByteTracker,
    TrackerConfig,
    ema_smooth,
    interpolate_gaps,
    iou_matrix,
    linear_assignment,
)


def _det(x, y, w=40, h=80, score=0.9, label="red"):
    return Detection((x, y, x + w, y + h), score, label)


def test_blob_detector_finds_box_and_label():
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    img[100:160, 50:90] = (40, 40, 225)      # red, BGR
    img[30:50, 200:260] = (225, 90, 30)      # blue
    dets = sorted(ColorBlobDetector().detect(img), key=lambda d: d.label)
    assert [d.label for d in dets] == ["blue", "red"]
    assert np.allclose(dets[1].box, (49.5, 99.5, 89.5, 159.5))
    assert dets[1].bottom_center == (69.5, 159.5)
    assert dets[1].score > 0.9


def test_blob_detector_ignores_dull_colours_and_scores_small_blobs_lower():
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    img[100:160, 50:90] = (90, 110, 150)     # desaturated reddish
    img[20:28, 20:28] = (40, 40, 225)        # small vivid red, 64 px
    dets = ColorBlobDetector().detect(img)
    assert len(dets) == 1
    assert 0.3 < dets[0].score < 0.6


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(iou_matrix(a, b), [[1.0, 1.0 / 3.0, 0.0]])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_linear_assignment_respects_gate():
    cost = np.array([[0.1, 0.9], [0.9, 0.95]])
    matches, un_rows, un_cols = linear_assignment(cost, max_cost=0.8)
    assert matches == [(0, 0)]
    assert un_rows == [1] and un_cols == [1]


def test_kalman_learns_constant_velocity():
    kf = BoxKalmanFilter()
    box = np.array([100.0, 100.0, 140.0, 180.0])
    mean, cov = kf.initiate(xyxy_to_xyah(box))
    for k in range(1, 25):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xyxy_to_xyah(box + [5.0 * k, 0, 5.0 * k, 0]))
    assert abs(mean[4] - 5.0) < 0.3
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_tracker_keeps_identity_of_moving_object():
    tracker = ByteTracker()
    ids = set()
    for k in range(20):
        out = tracker.update([_det(50 + 6 * k, 60)])
        ids.update(t.track_id for t in out)
    assert ids == {1}
    assert len(out) == 1 and abs(out[0].box[0] - (50 + 6 * 19)) < 6


def test_new_track_needs_two_hits():
    tracker = ByteTracker()
    assert tracker.update([_det(50, 60)]) == []
    assert len(tracker.update([_det(52, 60)])) == 1
    # A single spurious detection never becomes a reported track.
    tracker.update([_det(52, 60), _det(250, 30, label="blue")])
    out = tracker.update([_det(52, 60)])
    assert [t.track_id for t in out] == [1]


def test_low_confidence_detection_keeps_track_alive():
    """ByteTrack second stage: a weak detection continues an existing track."""
    tracker = ByteTracker(TrackerConfig(coast_frames=0))
    for k in range(5):
        tracker.update([_det(50 + 4 * k, 60)])
    out = tracker.update([_det(50 + 4 * 5, 60, score=0.2)])
    assert [t.track_id for t in out] == [1]
    # The same weak detection alone must not create a track.
    fresh = ByteTracker()
    for k in range(5):
        assert fresh.update([_det(50 + 4 * k, 60, score=0.2)]) == []


def test_track_is_recovered_with_same_id_after_occlusion():
    tracker = ByteTracker()
    for k in range(10):
        tracker.update([_det(50 + 5 * k, 60)])
    for _ in range(6):
        tracker.update([])
    out = tracker.update([_det(50 + 5 * 16, 60)])
    assert [t.track_id for t in out] == [1]


def test_coasting_reports_prediction_for_limited_time():
    tracker = ByteTracker(TrackerConfig(coast_frames=2))
    for k in range(10):
        tracker.update([_det(50 + 5 * k, 60)])
    assert len(tracker.update([])) == 1
    assert len(tracker.update([])) == 1
    assert tracker.update([]) == []


def test_lost_track_is_removed_after_timeout():
    tracker = ByteTracker(TrackerConfig(max_time_lost=5))
    for k in range(5):
        tracker.update([_det(50 + 5 * k, 60)])
    for _ in range(7):
        tracker.update([])
    assert tracker.get_track(1) is None
    for k in range(3):
        out = tracker.update([_det(90 + 5 * k, 60)])
    assert [t.track_id for t in out] == [2]


def test_class_aware_association_prevents_identity_theft():
    """A blue object passing over a lost red track must not inherit its id."""
    tracker = ByteTracker()
    for _ in range(5):
        tracker.update([_det(100, 60, label="red")])
    for _ in range(3):
        out = tracker.update([_det(102, 60, label="blue")])
    assert [(t.track_id, t.label) for t in out] == [(2, "blue")]


def test_gap_interpolation_and_ema():
    history = {7: {0: np.array([0.0, 0, 10, 10]), 4: np.array([40.0, 0, 50, 10])},
               8: {0: np.zeros(4), 50: np.ones(4)}}
    filled = interpolate_gaps(history, max_gap=10)
    assert sorted(filled[7]) == [0, 1, 2, 3, 4]
    assert np.allclose(filled[7][2], [20.0, 0, 30, 10])
    assert sorted(filled[8]) == [0, 50]  # gap too long, left untouched
    smoothed = ema_smooth(np.array([[0.0] * 4, [10.0] * 4, [10.0] * 4]), alpha=0.5)
    assert np.allclose(smoothed[:, 0], [0.0, 5.0, 7.5])
