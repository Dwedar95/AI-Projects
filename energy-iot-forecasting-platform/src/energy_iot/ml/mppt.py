"""Maximum power point tracking: classic baselines and ML voltage prediction.

Controllers
-----------
* :class:`PerturbObserve`: fixed-step hill climbing on measured power.
* :class:`IncrementalConductance`: compares dI/dV with -I/V.
* :class:`MLVoltagePredictor`: sets the operating voltage directly to the
  MPP voltage predicted by a regressor from an irradiance sensor and a cell
  temperature sensor.
* :class:`MLSeededPerturbObserve`: Perturb and Observe that jumps to the ML
  prediction when the two disagree by more than a threshold.

All controllers are evaluated in the same closed loop against the single-diode
module model. Tracking efficiency is harvested energy divided by the energy
available at the true MPP.

This is a reference implementation of the idea in Mahdy, Dwedar and Jesser
(IEEE ICMI 2026). It does not reproduce the experimental setup or the numbers
of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from ..simulators.pv import PVModule


@dataclass(frozen=True)
class Observation:
    """What a controller can measure in one control period."""

    voltage_v: float
    current_a: float
    irradiance_wm2: float  # irradiance sensor reading (noisy)
    cell_temp_c: float  # temperature sensor reading (noisy)


class MPPTController(Protocol):
    """A controller maps the latest observation to the next voltage setpoint."""

    name: str

    def reset(self, v_init: float) -> None: ...

    def step(self, obs: Observation) -> float: ...


class PerturbObserve:
    """Perturb and Observe with a fixed voltage step."""

    def __init__(self, step_v: float = 0.3) -> None:
        self.name = "perturb_and_observe"
        self.step_v = step_v
        self.reset(0.0)

    def reset(self, v_init: float) -> None:
        self._v = v_init
        self._prev_p: float | None = None
        self._direction = 1.0

    def step(self, obs: Observation) -> float:
        p = obs.voltage_v * obs.current_a
        if self._prev_p is not None and p < self._prev_p:
            self._direction = -self._direction
        self._prev_p = p
        self._v = obs.voltage_v + self._direction * self.step_v
        return self._v


class IncrementalConductance:
    """Incremental Conductance with a fixed voltage step."""

    def __init__(self, step_v: float = 0.3, tolerance: float = 2e-3) -> None:
        self.name = "incremental_conductance"
        self.step_v = step_v
        self.tolerance = tolerance
        self.reset(0.0)

    def reset(self, v_init: float) -> None:
        self._prev_v: float | None = None
        self._prev_i = 0.0

    def step(self, obs: Observation) -> float:
        v, i = obs.voltage_v, obs.current_a
        if self._prev_v is None:
            move = 1.0
        else:
            dv, di = v - self._prev_v, i - self._prev_i
            if abs(dv) < 1e-9:
                move = 0.0 if abs(di) < self.tolerance else float(np.sign(di))
            else:
                # dP/dV = I + V*dI/dV. Positive left of the MPP, negative right of it.
                slope = i + v * di / dv
                move = 0.0 if abs(slope) < self.tolerance * max(abs(i), 1e-6) else np.sign(slope)
        self._prev_v, self._prev_i = v, i
        return v + float(move) * self.step_v


def _mpp_features(x: np.ndarray) -> np.ndarray:
    """Feature map ``[G, log G, T]`` for the MPP voltage regressors.

    The MPP voltage is close to linear in log irradiance and in temperature,
    so the log term lets a linear model fit well.
    """
    g = np.clip(x[:, 0], 1.0, None)
    return np.column_stack([g, np.log(g), x[:, 1]])


def candidate_models(seed: int = 0) -> dict[str, RegressorMixin]:
    """The scikit-learn regressors compared for MPP voltage prediction."""
    return {
        "linear_log_features": make_pipeline(
            FunctionTransformer(_mpp_features), LinearRegression()
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=60, min_samples_leaf=3, random_state=seed
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=seed
        ),
    }


@dataclass(frozen=True)
class SensorNoise:
    """Standard deviations of the simulated sensor errors."""

    irradiance_rel: float = 0.02
    temperature_c: float = 0.5
    current_rel: float = 0.003


def make_mpp_dataset(
    module: PVModule, n_samples: int, seed: int = 0, noise: SensorNoise | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sample operating conditions and label them with the true MPP voltage.

    Features are noisy sensor readings ``[irradiance, cell temperature]``; the
    label is the numerically computed MPP voltage at the true conditions.
    """
    noise = noise or SensorNoise()
    rng = np.random.default_rng(seed)
    g = rng.uniform(50.0, 1100.0, size=n_samples)
    t = rng.uniform(0.0, 70.0, size=n_samples)
    v_mpp, _ = module.mpp_batch(g, t)
    g_meas = g * (1.0 + noise.irradiance_rel * rng.normal(size=n_samples))
    t_meas = t + noise.temperature_c * rng.normal(size=n_samples)
    return np.column_stack([g_meas, t_meas]), v_mpp


@dataclass(frozen=True)
class ModelScore:
    """Hold-out accuracy of one MPP voltage regressor."""

    name: str
    mae_v: float
    r2: float


def train_mpp_models(
    module: PVModule,
    n_train: int = 3000,
    n_test: int = 1000,
    seed: int = 0,
    noise: SensorNoise | None = None,
) -> tuple[dict[str, RegressorMixin], list[ModelScore]]:
    """Fit all candidate regressors and score them on an independent test set."""
    x_train, y_train = make_mpp_dataset(module, n_train, seed, noise)
    x_test, y_test = make_mpp_dataset(module, n_test, seed + 10_000, noise)
    models = candidate_models(seed)
    scores = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        scores.append(
            ModelScore(
                name, float(mean_absolute_error(y_test, pred)), float(r2_score(y_test, pred))
            )
        )
    return models, scores


class MLVoltagePredictor:
    """Sets the voltage to the regressor's MPP prediction every control period."""

    def __init__(self, model: RegressorMixin, name: str = "ml_voltage_prediction") -> None:
        self.name = name
        self.model = model

    def reset(self, v_init: float) -> None:
        self._v = v_init

    def predict(self, obs: Observation) -> float:
        """Predicted MPP voltage for the sensor readings in ``obs``."""
        x = np.array([[obs.irradiance_wm2, obs.cell_temp_c]])
        return float(self.model.predict(x)[0])

    def step(self, obs: Observation) -> float:
        self._v = self.predict(obs)
        return self._v


class MLSeededPerturbObserve:
    """Perturb and Observe that is re-seeded by the ML prediction after large changes."""

    def __init__(self, model: RegressorMixin, step_v: float = 0.1, jump_v: float = 1.0) -> None:
        self.name = "ml_seeded_perturb_and_observe"
        self.jump_v = jump_v
        self._ml = MLVoltagePredictor(model)
        self._po = PerturbObserve(step_v)

    def reset(self, v_init: float) -> None:
        self._po.reset(v_init)

    def step(self, obs: Observation) -> float:
        v_pred = self._ml.predict(obs)
        if abs(v_pred - obs.voltage_v) > self.jump_v:
            self._po.reset(v_pred)
            return v_pred
        return self._po.step(obs)


def irradiance_scenario(n_steps: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Test profile with steps, linear ramps and fast cloud-like fluctuation.

    Returns ``(irradiance_wm2, cell_temp_c)`` per control period. The first
    third holds random levels for 100 periods each, the second third ramps
    between 200 and 1000 W/m2, the last third is a smooth random walk.
    """
    rng = np.random.default_rng(seed)
    third = n_steps // 3
    levels = rng.uniform(150.0, 1000.0, size=third // 100 + 1)
    steps = np.repeat(levels, 100)[:third]
    ramp_period = 300
    phase = (np.arange(third) % ramp_period) / ramp_period
    ramps = 200.0 + 800.0 * (1.0 - np.abs(2.0 * phase - 1.0))
    n_walk = n_steps - 2 * third
    walk = np.cumsum(rng.normal(0.0, 12.0, size=n_walk))
    walk = 600.0 + walk - np.linspace(0.0, walk[-1], n_walk)
    g = np.clip(np.concatenate([steps, ramps, walk]), 60.0, 1100.0)
    # Cell temperature follows irradiance with a first-order lag.
    target = 20.0 + 0.03 * g
    t = np.empty(n_steps)
    t[0] = target[0]
    for k in range(1, n_steps):
        t[k] = t[k - 1] + 0.01 * (target[k] - t[k - 1])
    return g, t


@dataclass(frozen=True)
class TrackingResult:
    """Outcome of one closed-loop tracking run."""

    name: str
    efficiency: float
    voltage_v: np.ndarray
    power_w: np.ndarray
    mpp_power_w: np.ndarray


def simulate_tracking(
    module: PVModule,
    controller: MPPTController,
    irradiance_wm2: np.ndarray,
    cell_temp_c: np.ndarray,
    seed: int = 0,
    noise: SensorNoise | None = None,
    v_init: float | None = None,
) -> TrackingResult:
    """Run a controller in closed loop and compute its tracking efficiency.

    In each control period the commanded voltage is applied to the module, the
    resulting power is harvested, and the controller sees noisy measurements
    to decide the next setpoint.
    """
    noise = noise or SensorNoise()
    rng = np.random.default_rng(seed)
    n = len(irradiance_wm2)
    _, p_mpp = module.mpp_batch(irradiance_wm2, cell_temp_c)
    v_max = 1.4 * module.voc_ref_v
    v = 0.75 * module.voc_ref_v if v_init is None else v_init
    controller.reset(v)
    v_trace = np.empty(n)
    p_trace = np.empty(n)
    for k in range(n):
        g, t = float(irradiance_wm2[k]), float(cell_temp_c[k])
        i = float(module.current(v, g, t))
        v_trace[k] = v
        p_trace[k] = max(v * i, 0.0)
        obs = Observation(
            voltage_v=v,
            current_a=i * (1.0 + noise.current_rel * rng.normal()),
            irradiance_wm2=g * (1.0 + noise.irradiance_rel * rng.normal()),
            cell_temp_c=t + noise.temperature_c * rng.normal(),
        )
        v = float(np.clip(controller.step(obs), 0.0, v_max))
    efficiency = float(p_trace.sum() / p_mpp.sum())
    return TrackingResult(controller.name, efficiency, v_trace, p_trace, p_mpp)


def compare_trackers(
    module: PVModule,
    controllers: list[MPPTController],
    irradiance_wm2: np.ndarray,
    cell_temp_c: np.ndarray,
    seed: int = 0,
) -> list[TrackingResult]:
    """Evaluate several controllers on the same profile and noise seed."""
    return [
        simulate_tracking(module, c, irradiance_wm2, cell_temp_c, seed=seed) for c in controllers
    ]
