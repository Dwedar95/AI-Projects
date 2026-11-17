import numpy as np
import pytest

from anomalybags.synthetic import ANOMALY_TYPES, make_benchmark


def test_same_seed_same_data():
    a, b = make_benchmark(seed=5), make_benchmark(seed=5)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.labels, b.labels)
    assert not np.array_equal(a.test, make_benchmark(seed=6).test)


def test_labels_match_events_and_are_flagged_synthetic():
    bench = make_benchmark(seed=1, n_events=8)
    assert bench.synthetic is True
    assert len(bench.events) == 8
    mask = np.zeros_like(bench.labels)
    for event in bench.events:
        assert event.kind in ANOMALY_TYPES
        mask[event.start : event.end] = 1
    assert np.array_equal(mask, bench.labels)
    assert 0.0 < bench.contamination < 0.3


def test_events_do_not_overlap():
    events = make_benchmark(seed=2, n_events=10).events
    for first, second in zip(events[:-1], events[1:], strict=True):
        assert first.end + 100 <= second.start


@pytest.mark.parametrize("kind", ANOMALY_TYPES)
def test_single_type_benchmarks(kind):
    bench = make_benchmark(seed=0, kinds=(kind,), n_events=4)
    assert {e.kind for e in bench.events} == {kind}


def test_variance_change_raises_local_variance():
    bench = make_benchmark(seed=0, kinds=("variance_change",), n_events=4)
    event = bench.events[0]
    inside = np.diff(bench.test[event.start : event.end]).std()
    assert inside > 1.5 * np.diff(bench.train).std()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_benchmark(kinds=("drift",))
    with pytest.raises(ValueError):
        make_benchmark(n_test=600, n_events=50)
