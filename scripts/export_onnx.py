#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO26-seg to a Jetson Nano-friendly ONNX model"
    )
    parser.add_argument("--weights", default="yolo26n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=args.opset,
        )
    )

    if exported.resolve() != args.output.resolve():
        shutil.move(str(exported), str(args.output))

    print("Exported ONNX model to {}".format(args.output))


if __name__ == "__main__":
    main()
