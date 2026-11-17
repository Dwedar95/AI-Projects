import numpy as np
import pytest

from vision_autonomy.benchmark import make_cameras
from vision_autonomy.geometry import GroundProjector


@pytest.fixture(scope="session")
def cameras():
    return make_cameras()


@pytest.fixture(scope="session")
def projector(cameras):
    return GroundProjector(cameras[1], height=1.2, pitch=np.deg2rad(8.0))
