import numpy as np
import pytest

from vision_autonomy.metrics import evaluate_tracking

A = np.array([0.0, 0, 10, 10])
B = np.array([100.0, 100, 120, 120])


def test_perfect_tracking_scores_one():
    gt = [{1: A, 2: B}] * 10
    hyp = [{7: A, 9: B}] * 10
    m = evaluate_tracking(gt, hyp)
    assert m.mota == 1.0 and m.idf1 == 1.0 and m.id_switches == 0
    assert m.motp_iou == pytest.approx(1.0)


def test_identity_switch_is_counted_once_and_lowers_idf1():
    gt = [{1: A}] * 10
    hyp = [{5: A}] * 5 + [{6: A}] * 5
    m = evaluate_tracking(gt, hyp)
    assert m.id_switches == 1
    assert m.mota == pytest.approx(1.0 - 1 / 10)
    assert m.idf1 == pytest.approx(0.5)  # only half the frames carry the main identity


def test_false_positives_and_misses():
    gt = [{1: A}] * 4
    hyp = [{1: A, 2: B}, {1: A}, {}, {1: A + 100}]
    m = evaluate_tracking(gt, hyp)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 2, 2)
    assert m.mota == pytest.approx(1.0 - 4 / 4)
    assert m.precision == pytest.approx(0.5) and m.recall == pytest.approx(0.5)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        evaluate_tracking([{1: A}], [])
