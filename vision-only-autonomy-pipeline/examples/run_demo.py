"""Run the full pipeline on synthetic scenes and print measured results.

Everything printed here is computed in this run on the local CPU. The scenes are
synthetic (see vision_autonomy.synthetic); no real camera data is involved.

Usage: python examples/run_demo.py [--frames 150] [--save-dir out]
"""

from __future__ import annotations

import argparse
import pathlib
import platform
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vision_autonomy.benchmark import make_cameras, run_closed_loop, run_open_loop  # noqa: E402
from vision_autonomy.pipeline import STAGES, VisionPipeline  # noqa: E402
from vision_autonomy.synthetic import (  # noqa: E402
    EgoPose,
    SceneConfig,
    SyntheticScene,
    default_objects,
)
from vision_autonomy.tracker import TrackerConfig  # noqa: E402


def print_tracking(title: str, metrics) -> None:
    print(f"  {title:<34} MOTA {metrics.mota:6.3f}  IDF1 {metrics.idf1:6.3f}  "
          f"IDSW {metrics.id_switches:2d}  recall {metrics.recall:5.3f}  "
          f"precision {metrics.precision:5.3f}")


def save_visualisation(save_dir: pathlib.Path) -> None:
    """Write one annotated frame and its traversability grid for the docs."""
    raw, rect = make_cameras()
    scene = SyntheticScene(rect, SceneConfig(objects=default_objects()), raw_camera=raw)
    pipe = VisionPipeline(raw, rect)
    pose = EgoPose(0.0, 0.0, 0.0)
    for i in range(30):
        frame = scene.render(i * 0.1, pose)
        result = pipe.step(frame.image, 0.1)
        pose = EgoPose(pose.x + 0.2, 0.0, 0.0)
    image = pipe.rectifier(frame.image)
    for rt in result.tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in rt.track.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(image, f"#{rt.track.track_id} {rt.range_m:.1f}m", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / "raw_fisheye.png"), frame.image)
    cv2.imwrite(str(save_dir / "rectified_tracks.png"), image)
    cv2.imwrite(str(save_dir / "traversability_bev.png"),
                (result.traversability * 255).astype(np.uint8))
    print(f"\nSaved visualisations to {save_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--save-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    print("Vision-only autonomy pipeline: synthetic benchmark")
    print(f"python {platform.python_version()}, opencv {cv2.__version__}, numpy {np.__version__}, "
          f"{cv2.getNumberOfCPUs()} CPU threads, {platform.machine()}")
    print(f"stream: 640x480 synthetic fisheye, 10 Hz, {args.frames} frames, 8 moving objects\n")

    clean = run_open_loop(frames=args.frames)
    print("1) Tracking (IoU 0.5, objects over 150 px2 and at least half visible)")
    print_tracking("all detections, online", clean.online)
    no_coast = run_open_loop(frames=args.frames, drop_probability=0.2,
                             tracker=TrackerConfig(coast_frames=0))
    coast = run_open_loop(frames=args.frames, drop_probability=0.2)
    print_tracking("20% dropped, online, no coasting", no_coast.online)
    print_tracking("20% dropped, online, coast 2", coast.online)
    print_tracking("20% dropped, offline gap interp.", no_coast.interpolated)

    print("\n2) Flat-ground range from the box foot point (matched tracks)")
    edges = [0, 10, 20, 30]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (clean.range_true_m >= lo) & (clean.range_true_m < hi)
        if sel.any():
            print(f"  {lo:2d}-{hi:2d} m: n={int(sel.sum()):4d}  "
                  f"mean abs error {clean.range_abs_err_m[sel].mean():5.2f} m  "
                  f"mean rel error {100 * clean.range_rel_err[sel].mean():4.1f} %")
    print(f"  all     : n={len(clean.range_abs_err_m):4d}  "
          f"mean abs error {clean.range_abs_err_m.mean():5.2f} m  "
          f"mean rel error {100 * clean.range_rel_err.mean():4.1f} %")

    print("\n3) Ego-motion from Farneback flow on the near-field IPM (true speed 2.0 m/s)")
    print(f"  speed error    mean {clean.speed_err.mean():.3f} m/s, "
          f"p95 {np.percentile(clean.speed_err, 95):.3f} m/s")
    print(f"  yaw-rate error mean {clean.yaw_rate_err.mean():.4f} rad/s, "
          f"p95 {np.percentile(clean.yaw_rate_err, 95):.4f} rad/s")
    print(f"  valid estimates: {100 * clean.ego_valid_ratio:.0f} % of frames")

    print("\n4) Per-stage latency on this CPU (single process, first 5 frames skipped)")
    for stage in (*STAGES, "total"):
        t = clean.timings[stage]
        print(f"  {stage:<15} mean {t['mean_ms']:6.2f} ms   p95 {t['p95_ms']:6.2f} ms")
    print(f"  throughput: {clean.timings['total']['fps']:.1f} frames per second")

    loop = run_closed_loop(frames=200)
    print("\n5) Closed loop: planner drives a sinusoidal road with 3 parked objects")
    print(f"  distance travelled {loop.distance_m:.1f} m in {loop.frames * 0.1:.0f} s")
    print(f"  on-road ratio {100 * loop.on_road_ratio:.0f} %, "
          f"max offset from centre line {loop.max_lateral_offset_m:.2f} m "
          f"(road half width 3.50 m)")
    print(f"  minimum clearance to an object {loop.min_obstacle_clearance_m:.2f} m, "
          f"stops {loop.stops}, candidate switches {loop.candidate_switches}")
    print(f"  largest curvature step {loop.max_curvature_step:.3f} 1/m per cycle (limit 0.050)")

    if args.save_dir is not None:
        save_visualisation(args.save_dir)


if __name__ == "__main__":
    main()
