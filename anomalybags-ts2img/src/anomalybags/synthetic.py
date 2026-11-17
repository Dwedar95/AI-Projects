"""Synthetic benchmark generator. All data produced here is SYNTHETIC.

The normal signal is a sum of two sinusoids plus AR(1) noise. Four anomaly types
can be injected into the test split:

* ``spike``: one to three consecutive points pushed away from the signal
* ``level_shift``: a constant offset over a segment
* ``frequency_change``: the oscillation speeds up or slows down over a segment
  (phase-continuous, so there is no jump at the segment borders)
* ``variance_change``: the noise standard deviation is multiplied over a segment

The generator is intended for controlled ablations, not as a substitute for real
benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANOMALY_TYPES: tuple[str, ...] = ("spike", "level_shift", "frequency_change", "variance_change")

_SEGMENT_LENGTH: dict[str, tuple[int, int]] = {
    "spike": (1, 3),
    "level_shift": (40, 120),
    "frequency_change": (60, 150),
    "variance_change": (60, 150),
}


@dataclass(frozen=True)
class AnomalyEvent:
    """One injected anomaly: type, half-open index range and its magnitude parameter."""

    kind: str
    start: int
    end: int
    magnitude: float


@dataclass(frozen=True)
class Benchmark:
    """A synthetic train/test pair. ``train`` is anomaly-free."""

    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    events: tuple[AnomalyEvent, ...]
    seed: int
    synthetic: bool = True

    @property
    def contamination(self) -> float:
        return float(self.labels.mean())


def _ar1_noise(rng: np.random.Generator, n: int, sigma: float, rho: float) -> np.ndarray:
    """Stationary AR(1) noise with marginal standard deviation ``sigma``."""
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - rho**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + innov[t]
    return out


def _signal(
    rng: np.random.Generator,
    n: int,
    periods: tuple[float, float],
    phases: tuple[float, float],
    noise_sigma: float,
    freq_factor: np.ndarray | None = None,
    noise_factor: np.ndarray | None = None,
) -> np.ndarray:
    factor = np.ones(n) if freq_factor is None else freq_factor
    phase1 = phases[0] + 2.0 * np.pi * np.cumsum(factor / periods[0])
    phase2 = phases[1] + 2.0 * np.pi * np.cumsum(factor / periods[1])
    noise = _ar1_noise(rng, n, noise_sigma, rho=0.5)
    if noise_factor is not None:
        noise = noise * noise_factor
    return np.sin(phase1) + 0.5 * np.sin(phase2) + noise


def _place_events(
    rng: np.random.Generator,
    n: int,
    kinds: tuple[str, ...],
    n_events: int,
    margin: int,
    gap: int,
) -> list[tuple[str, int, int]]:
    """Draw non-overlapping segments, cycling through ``kinds``."""
    placed: list[tuple[str, int, int]] = []
    attempts = 0
    while len(placed) < n_events and attempts < 200 * n_events:
        attempts += 1
        kind = kinds[len(placed) % len(kinds)]
        lo, hi = _SEGMENT_LENGTH[kind]
        length = int(rng.integers(lo, hi + 1))
        start = int(rng.integers(margin, n - margin - length))
        end = start + length
        if all(end + gap <= s or start >= e + gap for _, s, e in placed):
            placed.append((kind, start, end))
    if len(placed) < n_events:
        raise ValueError("could not place all anomalies; use a longer series or fewer events")
    return sorted(placed, key=lambda item: item[1])


def make_benchmark(
    seed: int = 0,
    n_train: int = 4000,
    n_test: int = 6000,
    kinds: tuple[str, ...] = ANOMALY_TYPES,
    n_events: int = 8,
    noise_sigma: float = 0.15,
    difficulty: float = 1.0,
) -> Benchmark:
    """Generate a synthetic benchmark with ``n_events`` anomalies in the test split.

    ``difficulty`` scales anomaly magnitudes: values below 1 make anomalies weaker
    and harder to detect. The same ``seed`` always gives the same data.
    """
    unknown = set(kinds) - set(ANOMALY_TYPES)
    if unknown or not kinds:
        raise ValueError(f"kinds must be a non-empty subset of {ANOMALY_TYPES}")
    rng = np.random.default_rng(seed)
    periods = (float(rng.uniform(40.0, 60.0)), float(rng.uniform(13.0, 21.0)))
    phases = (float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi)))
    train = _signal(rng, n_train, periods, phases, noise_sigma)

    placed = _place_events(rng, n_test, kinds, n_events, margin=150, gap=100)
    freq_factor = np.ones(n_test)
    noise_factor = np.ones(n_test)
    additive = np.zeros(n_test)
    labels = np.zeros(n_test, dtype=int)
    events: list[AnomalyEvent] = []
    for kind, start, end in placed:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if kind == "spike":
            magnitude = sign * difficulty * float(rng.uniform(1.5, 2.5))
            additive[start:end] += magnitude
        elif kind == "level_shift":
            magnitude = sign * difficulty * float(rng.uniform(0.8, 1.4))
            additive[start:end] += magnitude
        elif kind == "frequency_change":
            ratio = 1.0 + difficulty * float(rng.uniform(0.6, 1.0))
            magnitude = ratio if sign > 0 else 1.0 / ratio
            freq_factor[start:end] = magnitude
        else:
            magnitude = 1.0 + difficulty * float(rng.uniform(2.0, 3.5))
            noise_factor[start:end] = magnitude
        labels[start:end] = 1
        events.append(AnomalyEvent(kind, start, end, float(magnitude)))

    test = _signal(rng, n_test, periods, phases, noise_sigma, freq_factor, noise_factor) + additive
    return Benchmark(train, test, labels, tuple(events), seed)
