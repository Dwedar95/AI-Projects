"""BEM data generator and machine-learning surrogate workflow for a wind turbine rotor.

Reference implementation by the author of Dwedar, Koese, Bayram, Jesser (2024),
IEEE SGRE. The rotor geometry is illustrative and the polars are approximations;
this is not the CFD data or the code of the paper.
"""

from wtsurrogate.airfoils import AIRFOIL_NAMES, get_airfoil
from wtsurrogate.bem import BemResult, solve_bem
from wtsurrogate.dataset import BemOracle, FeatureBuilder, generate_dataset
from wtsurrogate.doe import DesignSpace, latin_hypercube, sample_design
from wtsurrogate.pipeline import run_pipeline
from wtsurrogate.rotor import Rotor, reference_rotor

__all__ = [
    "AIRFOIL_NAMES",
    "BemOracle",
    "BemResult",
    "DesignSpace",
    "FeatureBuilder",
    "Rotor",
    "generate_dataset",
    "get_airfoil",
    "latin_hypercube",
    "reference_rotor",
    "run_pipeline",
    "sample_design",
    "solve_bem",
]

__version__ = "0.1.0"
