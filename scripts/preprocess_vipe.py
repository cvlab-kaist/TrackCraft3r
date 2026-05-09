"""Run ViPE on a video and save (depth, extrinsics, intrinsics) NPYs.

Outputs three NPY files that plug into `build_user_npz.py`:

    depth.npy       : (T, H, W) float32 — z-depth (ViPE outputs z-depth).
                      Pass `--depth_convention z` to build_user_npz.py.

    extrinsics.npy  : (T, 4, 4) float32 — C2W (OpenCV camera-to-world).
                      Verified at vipe/utils/io.py "cam2world matrices".
                      Pass `--extrinsics_convention c2w` to build_user_npz.py.

    intrinsics.npy  : (T, 3, 3) float32 — per-frame K. ViPE's intrinsics are
                      typically constant across frames in a single clip.

Setup (one-time):
    git clone https://github.com/nv-tlabs/vipe
    cd ViPE && pip install -e .   # follow ViPE's README for full deps

Usage:
    python scripts/preprocess_vipe.py \\
        --video_path my_video.mp4 \\
        --output_dir ./preproc/vipe/ \\
        --vipe_root  /path/to/ViPE
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str, required=True,
                   help="Input video (.mp4).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--vipe_root", type=str, required=True,
                   help="Path to ViPE repo (cloned + `pip install -e .`).")
    p.add_argument("--device", type=int, default=0,
                   help="CUDA device index (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--keep_intermediate", action="store_true",
                   help="Keep ViPE's per-modality folder layout after building NPYs.")
    args = p.parse_args()

    if not os.path.isdir(args.vipe_root):
        sys.exit(f"--vipe_root {args.vipe_root} does not exist.")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. ViPE expects a directory of .mp4 files (raw_mp4_stream). Stage our
    #    single input video in a tmp dir.
    work_in = tempfile.mkdtemp(prefix="vipe_in_")
    vid_name = os.path.splitext(os.path.basename(args.video_path))[0]
    staged_path = os.path.join(work_in, vid_name + ".mp4")
    shutil.copy(args.video_path, staged_path)

    # 2. Run ViPE.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    cmd = [
        sys.executable, "run.py",
        "pipeline=default", "streams=raw_mp4_stream",
        f"streams.base_path={work_in}",
        f"pipeline.output.path={out_dir}",
        "pipeline.post.depth_align_model=null",
    ]
    print(f"  running: {' '.join(cmd)}  (cwd={args.vipe_root})")
    subprocess.run(cmd, cwd=args.vipe_root, env=env, check=True)

    # 3. Read ViPE outputs from the per-modality layout.
    import torch
    depth_pt = torch.load(
        f"{out_dir}/depth/{vid_name}.pt",
        map_location="cpu", weights_only=False,
    )
    depth = depth_pt["depth"].cpu().numpy().astype(np.float32)   # (T, H, W) z-depth

    pose = np.load(f"{out_dir}/pose/{vid_name}.npz", allow_pickle=True)
    extr_c2w = pose["data"].astype(np.float32)                    # (T, 4, 4) c2w

    intr_npz = np.load(f"{out_dir}/intrinsics/{vid_name}.npz", allow_pickle=True)
    intr_4t = intr_npz["data"].astype(np.float64)                 # (T, 4) [fx, fy, cx, cy]

    T = intr_4t.shape[0]
    intr = np.zeros((T, 3, 3), dtype=np.float32)
    intr[:, 0, 0] = intr_4t[:, 0]
    intr[:, 1, 1] = intr_4t[:, 1]
    intr[:, 0, 2] = intr_4t[:, 2]
    intr[:, 1, 2] = intr_4t[:, 3]
    intr[:, 2, 2] = 1.0

    # 4. Save consolidated NPYs alongside the per-modality folders.
    np.save(os.path.join(out_dir, "depth.npy"), depth)
    np.save(os.path.join(out_dir, "extrinsics.npy"), extr_c2w)
    np.save(os.path.join(out_dir, "intrinsics.npy"), intr)
    print(f"  wrote {out_dir}/{{depth,extrinsics,intrinsics}}.npy")
    print(f"    depth shape:  {depth.shape}  (z-depth)")
    print(f"    extrinsics:   {extr_c2w.shape}  (C2W, raw — frame-0 not normalized)")
    print(f"    intrinsics:   {intr.shape}")

    if not args.keep_intermediate:
        for sub in ("rgb", "depth", "pose", "intrinsics", "mask"):
            shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
    shutil.rmtree(work_in, ignore_errors=True)


if __name__ == "__main__":
    main()
