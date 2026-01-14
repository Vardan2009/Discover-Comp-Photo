#!/usr/bin/env python3
"""
This script:
- Loads the subject image from --subject
- Lets you draw a polygon around the subject in an OpenCV window
- Saves a binary mask (0/1 float32) to --output-mask
- Saves a preview PNG of the mask to --output-preview
"""

import argparse
import os
from typing import Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive subject placement tool (solution).")
    parser.add_argument(
        "--subject",
        required=True,
        help="Path to the refined subject image (e.g., PNG).",
    )
    parser.add_argument(
        "--background",
        required=True,
        help="Path to the background image.",
    )
    parser.add_argument(
        "--output-placement",
        required=True,
        help="Path to save placement info as a .npy file (dict with keys 'y' and 'x').",
    )
    return parser.parse_args()


def load_images(subject_path: str, background_path: str) -> Tuple[np.ndarray, np.ndarray]:
    subj = cv2.imread(subject_path, cv2.IMREAD_UNCHANGED)
    if subj is None:
        raise FileNotFoundError(f"Could not read subject image at: {subject_path}")

    bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if bg is None:
        raise FileNotFoundError(f"Could not read background image at: {background_path}")

    return subj, bg


def interactive_place(subject: np.ndarray, background: np.ndarray) -> Tuple[int, int, float]:
    """
    Let the user interactively place (and scale) the subject on the background.

    Returns:
        (top_y, left_x, scale) where:
        - top_y, left_x are the integer coordinates of the subject's top-left corner.
        - scale is the final scale factor applied to the refined subject image.
    """
    bg_h, bg_w = background.shape[:2]

    if subject.shape[2] == 4:
        subj_base_bgr = cv2.cvtColor(subject, cv2.COLOR_BGRA2BGR)
    else:
        subj_base_bgr = subject.copy()

    scale = 1.0
    subj_vis = subj_base_bgr.copy()
    subj_h, subj_w = subj_vis.shape[:2]

    window_name = (
        "Place subject on background "
        "(drag with mouse, Enter: save, r: reset, q/Esc: cancel)"
    )

    y = max(0, bg_h - subj_h - 50)
    x = max(0, (bg_w - subj_w) // 2)

    dragging = False

    def render_frame() -> np.ndarray:
        frame = background.copy()
        roi = frame[y : y + subj_h, x : x + subj_w]
        if roi.shape[0] <= 0 or roi.shape[1] <= 0:
            return frame

        blended = cv2.addWeighted(roi, 0.4, subj_vis, 0.6, 0)
        frame[y : y + subj_h, x : x + subj_w] = blended
        return frame

    def mouse_callback(event, mx, my, flags, param):
        nonlocal dragging, x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            if x <= mx <= x + subj_w and y <= my <= y + subj_h:
                dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            new_x = mx - subj_w // 2
            new_y = my - subj_h // 2
            y = max(0, min(new_y, bg_h - subj_h))
            x = max(0, min(new_x, bg_w - subj_w))
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, bg_w), min(800, bg_h))
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Interactive placement instructions:")
    print("  - Click and drag the subject to reposition it.")
    print("  - Press '+' or '=' to scale the subject up.")
    print("  - Press '-' to scale the subject down.")
    print("  - Press Enter to confirm placement.")
    print("  - Press 'r' to reset position and size.")
    print("  - Press 'q' or Esc to cancel (no placement will be saved).")

    while True:
        frame = render_frame()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:
            break
        elif key in (ord("q"), 27): 
            x = -1
            y = -1
            break
        elif key == ord("r"):
            scale = 1.0
            subj_vis = subj_base_bgr.copy()
            subj_h, subj_w = subj_vis.shape[:2]
            y = max(0, bg_h - subj_h - 50)
            x = max(0, (bg_w - subj_w) // 2)
        elif key in (ord("+"), ord("=")):
            scale = min(scale * 1.1, 3.0)
            new_w = max(10, int(subj_base_bgr.shape[1] * scale))
            new_h = max(10, int(subj_base_bgr.shape[0] * scale))
            subj_vis = cv2.resize(subj_base_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            subj_h, subj_w = subj_vis.shape[:2]
            # Keep subject within frame after scaling
            y = max(0, min(y, bg_h - subj_h))
            x = max(0, min(x, bg_w - subj_w))
        elif key == ord("-"):
            scale = max(scale / 1.1, 0.2)
            new_w = max(10, int(subj_base_bgr.shape[1] * scale))
            new_h = max(10, int(subj_base_bgr.shape[0] * scale))
            subj_vis = cv2.resize(subj_base_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            subj_h, subj_w = subj_vis.shape[:2]
            y = max(0, min(y, bg_h - subj_h))
            x = max(0, min(x, bg_w - subj_w))

    cv2.destroyWindow(window_name)
    return y, x, scale


def main() -> None:
    args = parse_args()
    subject, background = load_images(args.subject, args.background)

    y, x, scale = interactive_place(subject, background)
    if y < 0 or x < 0:
        print("[WARN] Placement cancelled; no file written.")
        return

    placement_info = {"y": int(y), "x": int(x), "scale": float(scale)}
    os.makedirs(os.path.dirname(args.output_placement), exist_ok=True)
    np.save(args.output_placement, placement_info)
    print(f"[OK] Saved placement info to: {args.output_placement}")


if __name__ == "__main__":
    main()

