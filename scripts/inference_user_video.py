"""Run TrackCraft3R inference on a user-built NPZ and save the dense prediction.

Reads a single NPZ produced by `build_user_npz.py` (no GT tracks; only
images + depth + camera + intrinsics), runs the model once, and writes
the dense per-pixel prediction to a `.npz` that `visualize_dense.py`
can load.

Output schema (only what visualize_dense.py needs):
    track_map : (T, H, W, 3)   predicted per-pixel 3D track in frame-0 cam space
                               (used for the overlaid track trails)
    recon_map : (T, H, W, 3)   per-frame depth back-projection in frame-0 cam space
                               (used for the per-frame RGB point cloud)
    rgb       : (T, H, W, 3)   model-resolution RGB frames (for point-cloud coloring)

Usage:
    python scripts/inference_user_video.py \\
        --checkpoint_path ./checkpoints/trackcraft3r/model.safetensors \\
        --input_npz my_video.npz \\
        --output_npz my_video_dense.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

# Make `evaluation` importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.dust3r_eval_utils import load_npz_data
from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor


def _frame0_query_grid(H, W, stride=8):
    vv, uu = np.meshgrid(np.arange(0, H, stride),
                         np.arange(0, W, stride), indexing="ij")
    return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--input_npz", type=str, required=True,
                   help="NPZ produced by build_user_npz.py.")
    p.add_argument("--output_npz", type=str, required=True)

    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--lora_rank", type=int, default=1024)
    p.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--num_frames", type=int, default=12,
                   help="Frames per model run (default 12 = model's training length).")
    p.add_argument("--frame_stride", type=int, default=5,
                   help="Sample every Nth frame (default 5).")

    p.add_argument("--regression_timestep", type=int, default=-1)
    p.add_argument("--track_latent_length", type=int, default=12)
    p.add_argument("--resize_mode", type=str, default="stretch", choices=["pad", "stretch"])
    p.add_argument("--diag_max_depth", type=float, default=80.0)
    p.add_argument("--pj_norm_percentile_lo", type=float, default=2.0)
    p.add_argument("--pj_norm_percentile_hi", type=float, default=98.0)
    args = p.parse_args()

    # Load video + depth + camera + intrinsics. `build_user_npz.py` keeps ALL
    # frames; we subsample here at (num_frames × frame_stride).
    in_npz = np.load(args.input_npz, allow_pickle=True)
    T_total = in_npz["images_jpeg_bytes"].shape[0]
    span = args.num_frames * args.frame_stride
    if span > T_total:
        sys.exit(
            f"need {span} frames (= num_frames {args.num_frames} × frame_stride "
            f"{args.frame_stride}) but the NPZ only has {T_total}.")
    num_frames = args.num_frames
    frame_stride = args.frame_stride
    print(f"  using {num_frames} frames at stride={frame_stride} "
          f"(indices [0, {frame_stride}, ..., {(num_frames - 1) * frame_stride}])")

    # load_npz_data needs `tracks_XYZ` + `visibility`. The user NPZ has neither.
    # Provide a 1-track dummy so load_npz_data succeeds; we won't use it.
    if "tracks_XYZ" not in in_npz.files:
        dummy_track = np.zeros((T_total, 1, 3), dtype=np.float32)
        dummy_vis = np.ones((T_total, 1), dtype=bool)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            data = {k: in_npz[k] for k in in_npz.files}
            data["tracks_XYZ"] = dummy_track
            data["visibility"] = dummy_vis
            np.savez_compressed(tmp.name, **data)
            tmp_path = tmp.name
    else:
        tmp_path = args.input_npz

    (video_list, _, _, intrinsics, _, _, _, extrinsics_w2c) = load_npz_data(
        tmp_path, num_frames=num_frames, frame_stride=frame_stride)
    if tmp_path != args.input_npz:
        os.unlink(tmp_path)

    # Same slicing pattern that load_npz_data uses for the other arrays.
    depth_map = in_npz["depth_map"][:span:frame_stride]

    # Build predictor.
    predictor = WanSceneFlowPredictor(
        checkpoint_path=args.checkpoint_path,
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        height=args.height, width=args.width, device=args.device,
        regression_timestep=args.regression_timestep,
        track_latent_length=args.track_latent_length,
        resize_mode=args.resize_mode,
        diag_max_depth=args.diag_max_depth,
        pj_norm_percentile_lo=args.pj_norm_percentile_lo,
        pj_norm_percentile_hi=args.pj_norm_percentile_hi,
    )

    # Run dense forward. Use a dense query grid so OOB doesn't drop anything;
    # the dense outputs we want come from the predictor's `_last_*` cache.
    H_img, W_img = video_list[0].height, video_list[0].width
    query_uv = _frame0_query_grid(H_img, W_img, stride=max(1, min(H_img, W_img) // 16))
    vis_dummy = np.ones((num_frames, query_uv.shape[0]), dtype=bool)

    with torch.no_grad():
        predictor.predict(
            video_list, query_uv, vis_dummy, intrinsics,
            depth_map=depth_map, extrinsics_w2c=extrinsics_w2c,
        )

    out = {
        "track_map": predictor._last_row_dense.astype(np.float32),
        "recon_map": predictor._last_pj_input.astype(np.float32),
        "rgb":       predictor._last_rgb_frames,
    }

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    print(f"saved {args.output_npz}")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")


if __name__ == "__main__":
    main()
