"""Seeded synthetic mission log generator.

The generator produces plausible-looking logs for exercising the toolkit. It
is NOT a robot simulator and its output is not field data. Every mission
carries ``"synthetic": true`` in its ``mission_start`` event.

A :class:`BuildProfile` describes how a software build behaves on average
(intervention rate, gate conservativeness, localisation quality, fault rate).
Two profiles with different parameters give a baseline and a candidate log
for the regression gates.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .events import Event

ENVIRONMENTS = ("indoor_warehouse", "outdoor_campus", "construction_site")
TERRAINS = ("flat", "gravel", "grass", "stairs")
PLATFORMS = ("legged", "ground", "aerial")
_LUX_RANGE = {
    "indoor_warehouse": (80.0, 600.0),
    "outdoor_campus": (2000.0, 60000.0),
    "construction_site": (20.0, 40000.0),
}
STATE_PERIOD_S = 5
COMMAND_PERIOD_S = 2
GATE_MARGIN_M = 0.8  # the simulated gate withholds motion below this separation


@dataclass(frozen=True)
class BuildProfile:
    """Average behaviour of a software build in the synthetic generator."""

    name: str
    interventions_per_hour: float = 1.5
    withhold_base_prob: float = 0.01
    localisation_error_m: float = 0.08
    faults_per_hour: float = 1.2
    critical_fault_prob: float = 0.2
    close_encounters_per_hour: float = 6.0
    speed_mps: float = 1.0
    out_of_odd_prob: float = 0.08


BASELINE = BuildProfile(name="v1.4.0")
CANDIDATE_BETTER = BuildProfile(
    name="v1.5.0-rc1",
    interventions_per_hour=1.0,
    localisation_error_m=0.07,
    faults_per_hour=1.0,
)
CANDIDATE_REGRESSED = BuildProfile(
    name="v1.5.0-rc2",
    interventions_per_hour=3.2,
    withhold_base_prob=0.05,
    localisation_error_m=0.11,
    critical_fault_prob=0.5,
)


def generate_mission(
    profile: BuildProfile, mission_index: int, seed: int = 0, platform: str | None = None
) -> list[Event]:
    """Generate the events of one synthetic mission (1 second internal tick)."""
    # The build name is part of the seed so that two builds do not share random streams.
    rng = np.random.default_rng([seed, zlib.crc32(profile.name.encode()), mission_index])
    mission_id = f"{profile.name}-m{mission_index:04d}"
    platform = platform or PLATFORMS[int(rng.integers(len(PLATFORMS)))]
    environment = ENVIRONMENTS[int(rng.integers(len(ENVIRONMENTS)))]
    terrain = "flat" if environment == "indoor_warehouse" else TERRAINS[int(rng.integers(4))]
    lux_lo, lux_hi = _LUX_RANGE[environment]
    lux = float(np.exp(rng.uniform(np.log(lux_lo), np.log(lux_hi))))
    out_of_odd = rng.random() < profile.out_of_odd_prob
    if out_of_odd:
        lux = float(rng.uniform(2.0, 15.0))  # night operation without lights
    duration = int(rng.integers(8 * 60, 20 * 60))
    slope_base = {"flat": 1.0, "gravel": 4.0, "grass": 6.0, "stairs": 28.0}[terrain]

    def ev(t: float, etype: str, **fields: object) -> Event:
        return Event(mission_id=mission_id, build=profile.name, t=float(t), type=etype, **fields)

    events = [
        ev(
            0,
            "mission_start",
            platform=platform,
            environment=environment,
            terrain=terrain,
            planned_distance_m=round(duration * profile.speed_mps * 0.9, 1),
            synthetic=True,
        )
    ]
    p_tick = 1.0 / 3600.0
    distance = 0.0
    soc = float(rng.uniform(0.75, 1.0))
    manual_until = -1
    encounter_until = -1
    encounter_depth = 0.0
    interventions = 0
    command_id = 0
    speed = 0.0
    separation = 4.0
    outcome = "success"
    end_t = duration
    # Out-of-ODD conditions make the build struggle: more interventions, worse localisation.
    stress = 3.0 if out_of_odd else 1.0
    # Mission-to-mission variation (site difficulty), so missions are not identically distributed.
    stress *= float(rng.lognormal(0.0, 0.3))
    for t in range(1, duration):
        autonomous = t >= manual_until
        if t >= encounter_until and rng.random() < profile.close_encounters_per_hour * p_tick:
            encounter_until = t + int(rng.integers(6, 20))
            encounter_depth = float(rng.uniform(0.35, 1.1))
        base_sep = 3.5 + 1.0 * np.sin(t / 37.0) + 0.3 * rng.normal()
        separation = float(max(encounter_depth if t < encounter_until else base_sep, 0.15))

        if t % COMMAND_PERIOD_S == 0:
            command_id += 1
            cid = f"c{command_id}"
            proposed_speed = float(max(profile.speed_mps * (1.0 + 0.1 * rng.normal()), 0.0))
            events.append(
                ev(t, "command_proposed", command_id=cid, speed_mps=round(proposed_speed, 3))
            )
            if not autonomous:
                events.append(ev(t, "command_withheld", command_id=cid, reason="manual_control"))
                speed = 0.4 * profile.speed_mps  # teleoperated crawl
            elif separation < GATE_MARGIN_M:
                events.append(ev(t, "command_withheld", command_id=cid, reason="unsafe_separation"))
                speed = 0.0
            elif rng.random() < profile.withhold_base_prob:
                events.append(
                    ev(t, "command_withheld", command_id=cid, reason="reachability_margin")
                )
                speed = 0.0
            else:
                events.append(ev(t, "command_published", command_id=cid))
                speed = proposed_speed
        distance += speed
        soc -= (0.00025 + 0.0002 * speed) * (1.0 + 0.1 * rng.random())

        if autonomous and rng.random() < profile.interventions_per_hour * stress * p_tick:
            interventions += 1
            length = int(np.clip(rng.exponential(35.0), 5, 180))
            manual_until = t + length
            kind = (
                "e_stop"
                if rng.random() < 0.15
                else ("teleop_takeover" if rng.random() < 0.6 else "remote_assist")
            )
            cause = ("localisation", "obstacle", "planner_stuck")[int(rng.integers(3))]
            events.append(ev(t, "intervention", kind=kind, duration_s=length, cause=cause))
        if rng.random() < profile.faults_per_hour * p_tick:
            critical = rng.random() < profile.critical_fault_prob
            severity = "critical" if critical else ("major" if rng.random() < 0.3 else "minor")
            code = ("SENSOR_TIMEOUT", "MOTOR_OVERTEMP", "COMMS_DROP")[int(rng.integers(3))]
            events.append(ev(t, "fault", code=code, severity=severity))
            if critical:
                outcome, end_t = "failed", t + 1
        if t % STATE_PERIOD_S == 0:
            mode = "manual" if t < manual_until else ("stopped" if speed == 0.0 else "autonomous")
            loc = abs(profile.localisation_error_m * stress * (1.0 + 0.5 * rng.normal()))
            events.append(
                ev(
                    t,
                    "state",
                    mode=mode,
                    distance_m=round(distance, 2),
                    speed_mps=round(speed, 3),
                    localisation_error_m=round(float(loc), 4),
                    separation_m=round(separation, 3),
                    battery_soc=round(max(soc, 0.0), 4),
                    lighting_lux=round(lux * (1.0 + 0.05 * rng.normal()), 1),
                    link_latency_ms=round(float(40.0 + rng.exponential(25.0)), 1),
                    slope_deg=round(abs(slope_base * (1.0 + 0.2 * rng.normal())), 2),
                )
            )
        if outcome == "failed":
            break
        if interventions > 3 or soc < 0.1:
            outcome, end_t = "aborted", t + 1
            break
    events.append(
        ev(end_t, "mission_end", outcome=outcome, distance_m=round(distance, 2), duration_s=end_t)
    )
    return events


def generate_log(profile: BuildProfile, n_missions: int, seed: int = 0) -> list[Event]:
    """Generate ``n_missions`` synthetic missions for one build."""
    events: list[Event] = []
    for k in range(n_missions):
        events.extend(generate_mission(profile, k, seed))
    return events
