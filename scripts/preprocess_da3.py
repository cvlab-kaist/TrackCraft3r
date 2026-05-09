"""Run Depth-Anything-V3 on a video / image directory.

Outputs three NPY files that plug into `build_user_npz.py`:

    depth.npy       : (T, H, W) float32 — z-depth (DA3 outputs z-depth directly,
                      no Euclidean undistortion needed). Pass
                      `--depth_convention z` to build_user_npz.py.

    extrinsics.npy  : (T, 4, 4) float32 — W2C (NOT C2W).
                      Verified against DA3 source:
                          da3.py:225      output.extrinsics = affine_inverse(c2w)
                          da3_streaming.py:57 docstring "extrinsics: (w2c)"
                          colmap.py:40    "prediction.extrinsics,  # w2c"
                      Pass `--extrinsics_convention w2c` to build_user_npz.py.

    intrinsics.npy  : (T, 3, 3) float32 — rescaled to the original image
                      resolution (DA3's raw intrinsics are at the model's
                      processing resolution; we rescale here).

Setup (one-time):
    git clone https://github.com/ByteDance-Seed/depth-anything-3
    cd Depth-Anything-V3 && pip install -e .

Usage:
    python scripts/preprocess_da3.py \\
        --frame_dir /path/to/frames/    # OR --video_path X.mp4
        --output_dir ./preproc/da3/
        --da3_root  /path/to/Depth-Anything-V3
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom


def _read_frames(frame_dir=None, video_path=None):
    if frame_dir:
        exts = (".png", ".jpg", ".jpeg")
        files = sorted(f for f in os.listdir(frame_dir)
                       if f.lower().endswith(exts))
        if not files:
            raise RuntimeError(f"no images in {frame_dir}")
        return [Image.open(os.path.join(frame_dir, f)).convert("RGB") for f in files]
    if video_path:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames
    raise ValueError("Pass --frame_dir or --video_path")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame_dir", type=str, default=None,
                   help="Directory of input frames (.png / .jpg).")
    p.add_argument("--video_path", type=str, default=None,
                   help="Input video file (alternative to --frame_dir).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--da3_root", type=str, default=None,
                   help="Path to Depth-Anything-V3 repo (added to sys.path). "
                        "Skip if depth_anything_3 is pip-installed.")
    p.add_argument("--model_name", type=str,
                   default="depth-anything/DA3NESTED-GIANT-LARGE",
                   help="DA3 hub model id. Smaller alternatives: da3-large / da3-base.")
    p.add_argument("--process_res", type=int, default=504)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.da3_root:
        sys.path.insert(0, os.path.join(args.da3_root, "src"))
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        sys.exit(
            "Failed to import depth_anything_3. Either pass --da3_root <path-to-repo>\n"
            "or `pip install -e .` from the Depth-Anything-V3 repo.")

    pil_images = _read_frames(args.frame_dir, args.video_path)
    T = len(pil_images)
    orig_W, orig_H = pil_images[0].size
    print(f"  {T} frames at {orig_W}x{orig_H}")

    print(f"  loading {args.model_name} on {args.device}...")
    model = DepthAnything3.from_pretrained(args.model_name).to(args.device).eval()

    print(f"  running inference (process_res={args.process_res})...")
    with torch.no_grad():
        pred = model.inference(
            image=pil_images,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )

    depth_proc = pred.depth                      # (T, H_proc, W_proc) z-depth
    extr_3x4   = pred.extrinsics                 # (T, 3, 4) W2C
    intr_proc  = pred.intrinsics                 # (T, 3, 3) at H_proc x W_proc
    H_proc, W_proc = depth_proc.shape[1], depth_proc.shape[2]

    # 1. Resize depth to original image resolution.
    if (H_proc, W_proc) != (orig_H, orig_W):
        sh, sw = orig_H / H_proc, orig_W / W_proc
        depth = np.stack([
            zoom(depth_proc[t], (sh, sw), order=1) for t in range(T)
        ]).astype(np.float32)
    else:
        depth = depth_proc.astype(np.float32)

    # 2. Pad extrinsics (T, 3, 4) -> (T, 4, 4).
    extr = np.zeros((T, 4, 4), dtype=np.float32)
    extr[:, :3, :] = extr_3x4
    extr[:, 3, 3] = 1.0

    # 3. Rescale intrinsics from processing res to original res.
    sx, sy = orig_W / W_proc, orig_H / H_proc
    intr = intr_proc.copy().astype(np.float32)
    intr[:, 0, 0] *= sx
    intr[:, 1, 1] *= sy
    intr[:, 0, 2] *= sx
    intr[:, 1, 2] *= sy

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "depth.npy"), depth)
    np.save(os.path.join(args.output_dir, "extrinsics.npy"), extr)
    np.save(os.path.join(args.output_dir, "intrinsics.npy"), intr)
    print(f"  wrote {args.output_dir}/{{depth,extrinsics,intrinsics}}.npy")
    print(f"    depth shape:  {depth.shape}  (z-depth)")
    print(f"    extrinsics:   {extr.shape}    (W2C, raw — frame-0 not normalized)")
    print(f"    intrinsics:   {intr.shape}    (rescaled to {orig_W}x{orig_H})")
    print(f"    fx_fy_cx_cy[0]: "
          f"[{intr[0, 0, 0]:.2f}, {intr[0, 1, 1]:.2f}, {intr[0, 0, 2]:.2f}, {intr[0, 1, 2]:.2f}]")


if __name__ == "__main__":
    main()
