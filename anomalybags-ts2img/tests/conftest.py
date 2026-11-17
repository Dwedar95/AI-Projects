import numpy as np
import pytest

from anomalybags.synthetic import make_benchmark


@pytest.fixture(scope="session")
def small_bench():
    """Small synthetic benchmark shared by the slower tests."""
    return make_benchmark(seed=3, n_train=1500, n_test=2000, n_events=4)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
