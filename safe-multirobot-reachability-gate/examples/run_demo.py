"""Run the reachability gate on a simulated aerial-ground team and print measured results.

Everything printed is computed in this run. The world, the robots and the semantic
provider are simulated; the "VLM/LLM" is the deterministic offline mock.

Usage: python examples/run_demo.py [--cycles 120] [--seed 0]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from reach_gate.gate import SafetyGate  # noqa: E402
from reach_gate.reachability import (  # noqa: E402
    QuadrotorModel,
    UnicycleModel,
    contingency_inputs,
)
from reach_gate.semantic import parse_plan  # noqa: E402
from reach_gate.simulation import ScenarioConfig, Simulation, default_world  # noqa: E402


def containment_study(model, name: str, trials: int, rng: np.random.Generator) -> None:
    """Sample disturbed trajectories and count how many leave the reachable sets."""
    escaped = checks = 0
    widths = []
    for _ in range(trials):
        if isinstance(model, UnicycleModel):
            measured = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)])
            command = np.array([rng.uniform(-1, 1), rng.uniform(-1.2, 1.2)])
        else:
            measured = np.concatenate([rng.uniform(-2, 2, 3), rng.uniform(-1, 1, 3)])
            command = model.clip(rng.uniform(-1.5, 1.5, 3))
        inputs = contingency_inputs(model, command, horizon=15, commit_steps=5)
        x0 = model.measurement_set(measured)
        sets = model.reach(x0, inputs)
        widths.append(2.0 * sets[-1].radius()[:2].max())
        for start in x0.sample(rng, 4, boundary=True):
            x = start
            for k, u in enumerate(inputs):
                x = model.simulate_step(x, u, rng)
                checks += 1
                escaped += int(not sets[k + 1].contains_point(x))
    print(f"  {name:<10} {trials * 4} trajectories, {checks} state checks, "
          f"escaped: {escaped}, mean final xy width of the set: {np.mean(widths):.2f} m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", type=int, default=6, help="number of seeds in the sweep")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Safe multi-robot reachability gate: simulated demo")
    print("horizon 15 steps at dt = 0.1 s, command committed for 5 steps, then verified stop\n")

    print("1) Reachable sets against sampled plant trajectories (bounded random disturbances)")
    containment_study(UnicycleModel(), "unicycle", 150, rng)
    containment_study(QuadrotorModel(), "quadrotor", 150, rng)

    print("\n2) Single gate decisions for the ground vehicle next to the rack")
    base, robots, _, _ = default_world()
    ugv = robots[1].model
    gate = SafetyGate()
    cases = [("drive past the rack", np.array([5.5, 3.2, 0.0]), np.array([1.0, 0.0])),
             ("drive into the rack", np.array([5.5, 5.5, 0.0]), np.array([1.0, 0.0])),
             ("leave the workspace", np.array([19.2, 3.0, 0.0]), np.array([1.0, 0.0]))]
    for label, state, command in cases:
        verdict = gate.verify("ugv", ugv, ugv.measurement_set(state), command, base)
        reason = "published" if verdict.safe else (
            f"withheld ({verdict.violations[0].kind} at step {verdict.violations[0].step}: "
            f"{verdict.violations[0].detail})")
        print(f"  {label:<22} command {command.tolist()} -> {reason}")

    sim = Simulation(ScenarioConfig(seed=args.seed, cycles=args.cycles))
    print("\n3) Semantic layer (deterministic mock provider), first answer")
    plan = parse_plan(sim.provider.propose(sim._scene()))
    for zone in plan.keep_out:
        print(f"  keep-out '{zone.label}': centre {zone.center}, half size {zone.half_size}, "
              f"height {zone.height} m")
    for d in plan.directives:
        print(f"  {d.robot_id}: task {d.task_id}, goal {d.goal}, max speed {d.max_speed}, "
              f"altitude band {d.altitude_band}")
    print(f"  min separation requested: {plan.min_separation} m")

    summary = sim.run()
    print(f"\n4) Simulation: quadrotor + quadruped-like robot + ground vehicle, seed {args.seed}")
    for line in summary.lines():
        print("  " + line)

    print("\n5) Per robot")
    for robot in sim.robots:
        rid = robot.robot_id
        print(f"  {rid}: published {sim.bus.count(f'fleet/{rid}/cmd/verified')}, "
              f"withheld {sim.bus.count(f'fleet/{rid}/cmd/withheld')}, "
              f"tasks done {sim.bus.count(f'fleet/{rid}/task/done')}")

    print("\n6) First withheld commands on the bus")
    shown = 0
    for msg in sim.bus.log:
        if msg.topic.endswith("/cmd/withheld") and shown < 5:
            p = msg.payload
            command = [round(v, 2) for v in p["command"]]
            print(f"  #{msg.seq} {msg.topic}: {command} -> {p['reason']} "
                  f"at step {p['step']} ({p['detail']})")
            shown += 1

    if args.sweep > 0:
        print(f"\n7) Seed sweep, {args.cycles} cycles each (safety must hold, progress may not)")
        print("  seed  verif.  published  withheld  injected/withheld  tasks  min gap  "
              "collisions  containment")
        for seed in range(args.sweep):
            r = Simulation(ScenarioConfig(seed=seed, cycles=args.cycles)).run()
            print(f"  {seed:4d}  {r.verifications:6d}  {r.published:9d}  {r.withheld:8d}  "
                  f"{r.injected_unsafe:8d}/{r.injected_withheld:<8d}  {r.tasks_done}/"
                  f"{r.tasks_total}    {r.min_body_gap_m:5.2f} m  {r.collisions:10d}  "
                  f"{r.containment_violations:11d}")


if __name__ == "__main__":
    main()
