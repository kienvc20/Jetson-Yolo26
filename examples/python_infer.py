#!/usr/bin/env python3

import argparse

import cv2

from yolo26seg_runtime import Yolo26Seg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--input-name", default="images")
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError("Could not read image: {}".format(args.image))

    model = Yolo26Seg(
        args.engine,
        args.imgsz,
        args.imgsz,
        args.input_name,
    )

    print("Bindings:")
    for binding in model.bindings:
        print(
            "  {} {} {} {} {} bytes".format(
                binding.index,
                "input" if binding.is_input else "output",
                binding.name,
                binding.dtype,
                binding.bytes,
            )
        )

    for _ in range(5):
        model.infer(image)

    stats = model.infer(image)
    print("preprocess: {:.3f} ms".format(stats.preprocess_ms))
    print("TensorRT:  {:.3f} ms".format(stats.inference_ms))


if __name__ == "__main__":
    main()
