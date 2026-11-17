"""Reachability-based safety gate for aerial-ground multi-robot coordination."""

from .bus import MessageBus, topic_matches
from .gate import Box, SafetyConstraints, SafetyGate, Verdict
from .reachability import QuadrotorModel, ReachTube, UnicycleModel, build_tube
from .semantic import MockSemanticProvider, SemanticPlan, merge_constraints, parse_plan
from .simulation import RunSummary, ScenarioConfig, Simulation
from .zonotope import Zonotope, convex_hull_enclosure

__all__ = [
    "Box", "MessageBus", "MockSemanticProvider", "QuadrotorModel", "ReachTube", "RunSummary",
    "SafetyConstraints", "SafetyGate", "ScenarioConfig", "SemanticPlan", "Simulation",
    "UnicycleModel", "Verdict", "Zonotope", "build_tube", "convex_hull_enclosure",
    "merge_constraints", "parse_plan", "topic_matches",
]
__version__ = "0.1.0"
