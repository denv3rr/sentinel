from __future__ import annotations

import argparse
import json

from sentinel.ml.runners import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel detector micro-benchmarks")
    parser.add_argument("--model", default="yolov8n.pt", help="Model weights path")
    parser.add_argument("--profile", default="balanced", help="Detector profile")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--frames", type=int, default=120, help="Loop count for warm/motion loops")
    parser.add_argument("--width", type=int, default=640, help="Synthetic frame width")
    parser.add_argument("--height", type=int, default=360, help="Synthetic frame height")
    args = parser.parse_args()

    result = run_benchmark(
        model_name=args.model,
        profile=args.profile,
        confidence=args.confidence,
        frames=args.frames,
        frame_width=args.width,
        frame_height=args.height,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

