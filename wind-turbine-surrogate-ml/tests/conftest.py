import numpy as np
import pytest

from wtsurrogate.dataset import BemOracle, generate_dataset
from wtsurrogate.rotor import reference_rotor


@pytest.fixture(scope="session")
def rotor():
    return reference_rotor("naca4415")


@pytest.fixture(scope="session")
def oracle():
    return BemOracle()


@pytest.fixture(scope="session")
def dataset(oracle):
    return generate_dataset(240, seed=1, oracle=oracle)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
