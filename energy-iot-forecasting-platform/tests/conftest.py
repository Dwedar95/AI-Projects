import pytest

from energy_iot.platform import EnergyPlatform, SiteConfig


@pytest.fixture(scope="module")
def platform() -> EnergyPlatform:
    """Platform with four simulated days at 15 minute resolution."""
    p = EnergyPlatform(SiteConfig(seed=3))
    p.run(hours=96)
    return p
