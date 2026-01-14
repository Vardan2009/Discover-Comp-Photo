#!/usr/bin/env python3
"""
This script:
- Loads the subject image from --image
- Lets you draw a polygon around the subject in an OpenCV window
- Saves a binary mask (0/1 float32) to --output-mask
- Saves a preview PNG of the mask to --output-preview
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive polygon masking tool (solution).")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the subject image (e.g., 04-Composites/images/subject_for_mask.png).",
    )
    parser.add_argument(
        "--output-mask",
        required=True,
        help="Path to save the polygon mask as a .npy file (float32, values 0 or 1).",
    )
    parser.add_argument(
        "--output-preview",
        default=None,
        help="Optional path to save a PNG preview of the mask.",
    )
    return parser.parse_args()


def draw_polygon_mask(image_path: str, mask_path: str, preview_path: str | None = None) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    points: List[Tuple[int, int]] = []

    window_name = (
        "Draw polygon around subject "
        "(left-click: add point, Enter: save, r: reset, q/Esc: cancel)"
    )

    display = img.copy()

    def reset_canvas() -> None:
        nonlocal display, points, mask
        display = img.copy()
        mask[:] = 0
        points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal display, points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], (0, 0, 255), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1000, w), min(800, h))
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Interactive masking instructions:")
    print("  - Left-click to add polygon points around the subject.")
    print("  - Press Enter to finish and save the mask.")
    print("  - Press 'r' to reset and start over.")
    print("  - Press 'q' or Esc to cancel without saving.")

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 13: 
            break
        elif key in (ord("q"), 27): 
            points = []
            break
        elif key == ord("r"):
            reset_canvas()

    cv2.destroyWindow(window_name)

    if len(points) < 3:
        print("[WARN] Not enough points to form a polygon; mask not saved.")
        return

    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    np.save(mask_path, mask.astype(np.float32))
    print(f"[OK] Saved polygon mask to: {mask_path}")

    if preview_path:
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        cv2.imwrite(preview_path, mask * 255)
        print(f"[OK] Saved mask preview PNG to: {preview_path}")


def main() -> None:
    args = parse_args()
    draw_polygon_mask(args.image, args.output_mask, args.output_preview)


if __name__ == "__main__":
    main()

