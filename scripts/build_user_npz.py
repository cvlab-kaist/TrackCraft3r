"""Build a TrackCraft3R-format NPZ from a user video + pre-computed depth/camera.

Combines RGB frames from a video file (or a directory of images) with depth
maps and camera parameters obtained from an external estimator (DepthAnything-3
or ViPE) into a single NPZ that matches the release convention used by
`evaluation/`.

Output schema matches kubric_da3 / kubric_vipe (minus the dense GT tracks):
    images_jpeg_bytes : (T,)            JPEG-encoded RGB frames
    depth_map         : (T, H, W)       z-depth (Euclidean depth is
                                        auto-converted via undistort_depthmap)
    extrinsics_w2c    : (T, 4, 4)       OpenCV world-to-camera, frame-0 = I
    fx_fy_cx_cy       : (4,)            intrinsics

After running this, `inference_user_video.py` consumes the NPZ.

Usage:
    python scripts/build_user_npz.py \\
        --video_path my_video.mp4 \\
        --depth_npy my_video_depth.npy \\
        --extrinsics_npy my_video_extr.npy \\
        --intrinsics_npy my_video_intrinsics.npy \\
        --num_frames 12 --frame_stride 5 \\
        --output_npz my_video.npz \\
        --depth_convention euclidean   # or 'z' if already z-depth
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def _undistort_depthmap(depth_euclidean, fx, fy, cx, cy):
    """Convert Euclidean (ray) depth to z-depth using intrinsics. Matches the
    formula used in training / eval_dense_kubric."""
    H, W = depth_euclidean.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    return depth_euclidean / np.sqrt(1 + x ** 2 + y ** 2)


def _read_video_frames(video_path):
    """Read all frames from a video file into a list of HxWx3 uint8 arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {video_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _read_image_dir(image_dir):
    """Read PNG/JPG images from a directory, sorted by filename."""
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith(exts))
    if not files:
        raise RuntimeError(f"no images found in {image_dir}")
    frames = []
    for fn in files:
        img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
        frames.append(np.array(img))
    return frames


def _frame_zero_normalize_w2c(extrinsics_w2c):
    inv0 = np.linalg.inv(extrinsics_w2c[0])
    return np.stack([
        extrinsics_w2c[t] @ inv0 for t in range(extrinsics_w2c.shape[0])
    ]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str,
                   help="Input video file (.mp4 / .mov / etc.) OR image directory.")
    p.add_argument("--depth_npy", type=str, required=True,
                   help="(T, H, W) depth array. See --depth_convention.")
    p.add_argument("--extrinsics_npy", type=str, required=True,
                   help="Per-frame camera extrinsics. See --extrinsics_convention.")
    p.add_argument("--intrinsics_npy", type=str, required=True,
                   help="Either (4,) [fx, fy, cx, cy] or (3, 3) K matrix.")

    p.add_argument("--depth_convention", type=str, default="z",
                   choices=["z", "euclidean"],
                   help="'z'=already z-depth (DA3 + ViPE both output z-depth). "
                        "'euclidean'=ray distance (raw kubric TIFFs). "
                        "If euclidean, undistort_depthmap is applied per frame.")
    p.add_argument("--extrinsics_convention", type=str, default="w2c",
                   choices=["w2c", "c2w"],
                   help="Convention of the input array. Internally normalized to w2c.")
    p.add_argument("--intrinsics_resolution", type=int, nargs=2, default=None,
                   metavar=("H", "W"),
                   help="(H, W) the intrinsics correspond to. Defaults to the depth "
                        "map's HxW. If your intrinsics are at a different scale, set this.")

    p.add_argument("--output_npz", type=str, required=True)
    args = p.parse_args()

    # 1. Load arrays
    depth_full = np.load(args.depth_npy).astype(np.float32)         # (T_total, H, W)
    extr_full  = np.load(args.extrinsics_npy).astype(np.float64)    # (T_total, 4, 4)
    K_in       = np.load(args.intrinsics_npy).astype(np.float64)    # (4,) or (3, 3)

    # Normalize intrinsics to [fx, fy, cx, cy]
    if K_in.ndim == 3 and K_in.shape[1:] == (3, 3):
        # Per-frame (T, 3, 3) — DA3 raw output. Use the first frame
        # (per-frame variation is sub-pixel for DA3).
        K_in = K_in[0]
    if K_in.shape == (3, 3):
        fx, fy, cx, cy = K_in[0, 0], K_in[1, 1], K_in[0, 2], K_in[1, 2]
    elif K_in.shape == (4,):
        fx, fy, cx, cy = K_in
    else:
        raise ValueError(f"intrinsics shape {K_in.shape} not in {{(T,3,3), (3,3), (4,)}}")
    print(f"  intrinsics (input): fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # 2. Convert c2w -> w2c if needed, then frame-0 normalize.
    if args.extrinsics_convention == "c2w":
        extr_w2c = np.stack([np.linalg.inv(extr_full[t]) for t in range(extr_full.shape[0])])
    else:
        extr_w2c = extr_full
    extr_w2c = _frame_zero_normalize_w2c(extr_w2c)

    # 3. Read video frames
    if os.path.isfile(args.video_path):
        frames = _read_video_frames(args.video_path)
    else:
        frames = _read_image_dir(args.video_path)
    print(f"  loaded {len(frames)} frames from {args.video_path}")

    # Keep all frames (no subsampling). `inference_user_video.py` decides
    # num_frames / frame_stride at runtime.
    T = min(len(frames), depth_full.shape[0], extr_w2c.shape[0])
    frames_sel = frames[:T]
    depth_sel  = depth_full[:T].copy()
    extr_sel   = extr_w2c[:T].copy()
    print(f"  keeping {T} frames (no stride subsampling at build time)")

    # 5. Resolution check + intrinsic rescaling
    H_img, W_img = frames_sel[0].shape[:2]
    H_d, W_d = depth_sel.shape[1], depth_sel.shape[2]
    if (H_img, W_img) != (H_d, W_d):
        raise ValueError(
            f"image resolution {(H_img, W_img)} != depth resolution {(H_d, W_d)}. "
            f"Resize one of them before passing to this script.")
    if args.intrinsics_resolution is not None:
        H_K, W_K = args.intrinsics_resolution
        if (H_K, W_K) != (H_img, W_img):
            sx = W_img / W_K
            sy = H_img / H_K
            fx, fy = fx * sx, fy * sy
            cx, cy = cx * sx, cy * sy
            print(f"  intrinsics rescaled from {(H_K, W_K)} -> {(H_img, W_img)}: "
                  f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    fx_fy_cx_cy = np.array([fx, fy, cx, cy], dtype=np.float64)

    # 6. Convert Euclidean -> z-depth if needed
    if args.depth_convention == "euclidean":
        print("  converting Euclidean depth -> z-depth (undistort_depthmap)")
        depth_sel = np.stack([
            _undistort_depthmap(depth_sel[t], fx, fy, cx, cy) for t in range(depth_sel.shape[0])
        ]).astype(np.float32)

    # 7. Encode frames as JPEG bytes
    enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    images_jpeg_bytes = []
    for fr in frames_sel:
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, enc_params)
        assert ok, "JPEG encode failed"
        images_jpeg_bytes.append(buf.tobytes())
    images_jpeg_bytes = np.array(images_jpeg_bytes, dtype=object)

    # 8. Save
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        images_jpeg_bytes=images_jpeg_bytes,
        depth_map=depth_sel.astype(np.float32),
        extrinsics_w2c=extr_sel.astype(np.float32),
        fx_fy_cx_cy=fx_fy_cx_cy,
    )
    print(f"  saved {args.output_npz}")
    print(f"    T={T}, H={H_img}, W={W_img}")


if __name__ == "__main__":
    main()
