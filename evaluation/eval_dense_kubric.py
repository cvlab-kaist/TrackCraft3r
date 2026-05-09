"""Dense 3D tracking evaluation on Kubric (kubric_da3 / kubric_vipe).

Reads the unified-schema NPZ files produced by `build_clean_eval_dataset.py`:

    eval_dataset/kubric_<da3|vipe>/<seq>.npz containing
        images_jpeg_bytes : (T,)            JPEG bytes
        depth_map         : (T, H, W)       predicted depth (DA3 / ViPE)
        extrinsics_w2c    : (T, 4, 4)       predicted, frame-0 normalized
        extrinsics_w2c_gt : (T, 4, 4)       GT, frame-0 normalized
        fx_fy_cx_cy       : (4,)            GT intrinsics
        world_coords      : (P, T, 3)       dense GT 3D in world space
        occluded          : (P, T)
        is_bkgd           : (P, 1)
        H, W              : scalar          P = H * W

Output: per-vis-threshold tables of TAPVid3D Sim3-closed (st4rtrack threshold)
metric, both in predicted-camera and GT-camera world frames.
"""

import argparse
import glob
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .eval_worldtrack import compute_tapvid3d_st4rtrack
from .wan_scene_flow_predictor import WanSceneFlowPredictor


###############################################################################
# Dataset
###############################################################################

def _decode_jpeg_bytes(buf):
    arr = np.frombuffer(buf, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _normalize_w2c(w2c):
    inv0 = np.linalg.inv(w2c[0])
    return np.stack([w2c[t] @ inv0 for t in range(w2c.shape[0])]).astype(np.float32)


class KubricDenseDataset:
    """Iterate flat-NPZ kubric variants. Each NPZ already contains everything
    needed (predicted depth+camera, GT extrinsics, dense GT tracks)."""

    def __init__(self, data_root, num_frames=24, frame_stride=1, max_samples=50):
        self.samples = sorted(glob.glob(f"{data_root}/*.npz"))
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        print(f"KubricDenseDataset: {len(self.samples)} sequences from {data_root} "
              f"(num_frames={num_frames}, stride={frame_stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path = self.samples[idx]
        d = np.load(npz_path, allow_pickle=True)
        T_total = d["images_jpeg_bytes"].shape[0]
        frame_indices = list(range(0, T_total, self.frame_stride))[:self.num_frames]
        T = len(frame_indices)
        H, W = int(d["H"]), int(d["W"])

        images_pil = [_decode_jpeg_bytes(d["images_jpeg_bytes"][fi]) for fi in frame_indices]
        depth_map = d["depth_map"][frame_indices].astype(np.float32)

        # Frame-0 normalize w2c (in case source isn't already)
        extrinsics_w2c = _normalize_w2c(d["extrinsics_w2c"][frame_indices])
        gt_extr = _normalize_w2c(d["extrinsics_w2c_gt"][frame_indices])

        fx, fy, cx, cy = [float(v) for v in d["fx_fy_cx_cy"]]
        intr_4 = np.array([fx, fy, cx, cy], dtype=np.float32)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics_mat = np.tile(K[None], (T, 1, 1))

        # Dense GT
        world_coords = d["world_coords"][:, frame_indices, :].astype(np.float32)  # (P, T, 3)
        occluded     = d["occluded"][:, frame_indices]                             # (P, T) bool
        is_bkgd      = d["is_bkgd"]                                                 # (P, 1) bool

        # Transform world_coords -> GT cam-0 frame.
        gt_w2c0 = gt_extr[0]
        N = world_coords.shape[0]
        gt_cam0 = np.zeros((T, N, 3), dtype=np.float32)
        for t in range(T):
            gt_cam0[t] = (gt_w2c0[:3, :3] @ world_coords[:, t, :].T).T + gt_w2c0[:3, 3]
        gt_cam0 = gt_cam0.reshape(T, H, W, 3)

        # visibility shape (T, H, W). occluded is (P, T) row-major (v*W+u at frame 0).
        visibility = (~occluded).T.reshape(T, H, W)
        is_bkgd_map = is_bkgd.reshape(H, W)

        return {
            "seq_name":       os.path.splitext(os.path.basename(npz_path))[0],
            "images_pil":     images_pil,
            "gt_cam0":        gt_cam0,         # (T, H, W, 3) GT in GT-cam-0 frame
            "visibility":     visibility,      # (T, H, W) bool, True = visible
            "is_bkgd":        is_bkgd_map,     # (H, W) bool, True = background
            "intrinsics_4":   intr_4,
            "intrinsics_mat": intrinsics_mat,
            "extrinsics_w2c": extrinsics_w2c,  # predicted, frame-0 = I
            "depth_map":      depth_map,       # predicted z-depth
            "H": H, "W": W, "T": T,
        }


###############################################################################
# Wan dense prediction
###############################################################################

def predict_dense_wan(predictor, sample):
    """Trigger a dense forward pass with a dummy query and return the cached
    `_last_row_dense` (T, H_pred, W_pred, 3) in pred-cam-0 space."""
    images_pil = sample["images_pil"]
    cw, ch = images_pil[0].width // 2, images_pil[0].height // 2
    dummy_uv = np.array([[cw, ch]], dtype=np.float64)
    dummy_vis = np.ones((sample["T"], 1), dtype=bool)
    predictor.predict(
        images_pil, dummy_uv, dummy_vis, sample["intrinsics_4"],
        depth_map=sample["depth_map"],
        extrinsics_w2c=sample["extrinsics_w2c"],
    )
    return predictor._last_row_dense  # (T, H_pred, W_pred, 3)


def _bilinear_zoom(arr, zoom_h, zoom_w):
    """bilinear resize on the spatial dims of (T, H, W, ...) or (T, H, W)."""
    from scipy.ndimage import zoom
    if arr.ndim == 4:
        return zoom(arr, (1.0, zoom_h, zoom_w, 1.0), order=1)
    if arr.ndim == 3:
        return zoom(arr, (1.0, zoom_h, zoom_w), order=1)
    raise ValueError(f"Unsupported ndim: {arr.ndim}")


###############################################################################
# Eval loop
###############################################################################

def _resample_gt(gt_cam0, vis, H_gt, W_gt, H_pred, W_pred, eval_stride):
    h = np.arange(0, H_pred, eval_stride)
    w = np.arange(0, W_pred, eval_stride)
    h_gt = np.clip(np.round(h * (H_gt - 1) / max(H_pred - 1, 1)).astype(int), 0, H_gt - 1)
    w_gt = np.clip(np.round(w * (W_gt - 1) / max(W_pred - 1, 1)).astype(int), 0, W_gt - 1)
    return gt_cam0[:, h_gt][:, :, w_gt], vis[:, h_gt][:, :, w_gt]


def _metric_sweep(gt_flat, pred_flat, vis_flat, vis_pred_flat):
    """Compute (AJ, OC_ACC, pts) at vis thresholds 0.1..0.9 with st4rtrack metric."""
    gt_occ = ~vis_flat
    out = {}
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_occ = (vis_pred_flat < thr) if vis_pred_flat is not None else \
                   np.zeros_like(vis_flat)
        m = compute_tapvid3d_st4rtrack(gt_occ, gt_flat, pred_occ, pred_flat)
        tag = f"th{int(thr * 10):02d}"
        for k, v in m.items():
            out[f"{k}_{tag}"] = v
    return out


def eval_dataset(predictor, dataset, args):
    eval_stride = args.eval_stride
    all_results = []
    for idx in tqdm(range(len(dataset)), desc=f"Dense eval ({args.data_type})"):
        sample = dataset[idx]
        seq_name = sample["seq_name"]
        try:
            torch.cuda.empty_cache()
            pred_cam0 = predict_dense_wan(predictor, sample)[:sample["T"]]
            T = sample["T"]
            H_pred, W_pred = pred_cam0.shape[1], pred_cam0.shape[2]
            pred_vis_map = getattr(predictor, "_last_vis_dense", None)
            if pred_vis_map is not None:
                pred_vis_map = pred_vis_map[:T]

            # Optional common eval grid
            if args.eval_height > 0 and args.eval_width > 0 \
                    and (args.eval_height != H_pred or args.eval_width != W_pred):
                zh = args.eval_height / H_pred
                zw = args.eval_width / W_pred
                pred_cam0 = _bilinear_zoom(pred_cam0, zh, zw)
                if pred_vis_map is not None:
                    pred_vis_map = _bilinear_zoom(pred_vis_map, zh, zw)
                H_pred, W_pred = args.eval_height, args.eval_width

            # Resample GT to prediction grid
            gt_sampled, vis_sampled = _resample_gt(
                sample["gt_cam0"], sample["visibility"],
                sample["H"], sample["W"], H_pred, W_pred, eval_stride)
            if eval_stride > 1:
                pred_cam0 = pred_cam0[:, ::eval_stride, ::eval_stride]
                if pred_vis_map is not None:
                    pred_vis_map = pred_vis_map[:, ::eval_stride, ::eval_stride]

            H_eval, W_eval = gt_sampled.shape[1:3]
            gt_flat = gt_sampled.reshape(T, -1, 3)
            pred_flat = pred_cam0.reshape(T, -1, 3)
            vis_flat = vis_sampled.reshape(T, -1)
            pred_vis_flat = pred_vis_map.reshape(T, -1) if pred_vis_map is not None else None

            # Foreground filter + |xyz| < 50m (matches training supervision)
            valid = np.all(np.isfinite(gt_flat), axis=(0, 2))
            valid &= (np.max(np.abs(gt_flat), axis=(0, 2)) < args.max_abs_xyz)
            if not args.include_bg:
                fg = (~sample["is_bkgd"])
                if eval_stride > 1:
                    fg = fg[::eval_stride, ::eval_stride]
                if fg.shape != (H_eval, W_eval):
                    fg = np.array(Image.fromarray(fg.astype(np.uint8) * 255)
                                  .resize((W_eval, H_eval), Image.NEAREST)) > 0
                valid &= fg.reshape(-1)
            if valid.sum() == 0:
                print(f"  [{idx+1}/{len(dataset)}] {seq_name}: no valid points, skip")
                continue
            gt_flat = gt_flat[:, valid]
            pred_flat = pred_flat[:, valid]
            vis_flat = vis_flat[:, valid]
            if pred_vis_flat is not None:
                pred_vis_flat = pred_vis_flat[:, valid]

            metrics = _metric_sweep(gt_flat, pred_flat, vis_flat, pred_vis_flat)
            metrics["seq_name"] = seq_name
            all_results.append(metrics)

            print(f"  [{idx+1}/{len(dataset)}] {seq_name}: "
                  f"AJ@0.5={metrics.get('average_jaccard_th05', 0):.4f}  "
                  f"pts@0.5={metrics.get('average_pts_within_thresh_th05', 0):.4f}")

        except Exception as e:
            print(f"  [{idx+1}/{len(dataset)}] {seq_name}: ERROR — {e}")
            import traceback; traceback.print_exc()
            continue

    return all_results


def aggregate_and_save(all_results, output_dir, data_type):
    if not all_results:
        print(f"[WARN] No valid results for {data_type}")
        return
    keys = set()
    for r in all_results:
        for k in r.keys():
            if k != "seq_name":
                keys.add(k)
    finals = {}
    for k in sorted(keys):
        vals = [r[k] for r in all_results
                if k in r and r[k] is not None and not math.isnan(r[k])]
        finals[k] = float(np.mean(vals)) if vals else float("nan")

    log = "\n=== Final Results ===\n"
    log += "\nTAPVid3D Sim3 closed (st4rtrack threshold) across vis thresholds:\n"
    log += f"  {'thr':>5} | {'AJ':>7} | {'OC_ACC':>7} | {'pts':>7}\n"
    for i in range(1, 10):
        tag = f"th{i:02d}"
        thr = i / 10.0
        aj  = finals.get(f"average_jaccard_{tag}", float("nan"))
        oc  = finals.get(f"occlusion_accuracy_{tag}", float("nan"))
        pts = finals.get(f"average_pts_within_thresh_{tag}", float("nan"))
        log += f"  {thr:>5.1f} | {aj:>7.4f} | {oc:>7.4f} | {pts:>7.4f}\n"

    print(log)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"track_eval_{data_type}.txt")
    with open(out_path, "a") as f:
        f.write(f"\n=== Evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(log)
    print(f"Results saved to {out_path}")

    json_path = os.path.join(output_dir, f"per_video_{data_type}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Per-video results saved to {json_path}")


###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Dense 3D tracking eval on kubric")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--data_root", type=str, required=True,
                   help="Path to a kubric variant folder (e.g. ./eval_dataset/kubric_da3)")
    p.add_argument("--data_type", type=str, default=None,
                   help="Tag used in output filenames. Defaults to basename(data_root).")
    p.add_argument("--num_frames", type=int, default=24)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--eval_stride", type=int, default=1,
                   help="Spatial subsampling for the eval grid (1=full).")
    p.add_argument("--eval_height", type=int, default=480)
    p.add_argument("--eval_width", type=int, default=832)

    p.add_argument("--include_bg", action="store_true",
                   help="Also score background pixels (default: FG only).")
    p.add_argument("--max_abs_xyz", type=float, default=50.0,
                   help="Drop GT points whose |xyz| exceeds this (m).")

    p.add_argument("--output_dir", type=str, default="./eval_results/kubric")

    # Wan args (same as eval_worldtrack)
    p.add_argument("--lora_rank", type=int, default=1024)
    p.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--regression_timestep", type=int, default=-1)
    p.add_argument("--track_latent_length", type=int, default=12)
    p.add_argument("--resize_mode", type=str, default="stretch", choices=["pad", "stretch"])
    p.add_argument("--diag_max_depth", type=float, default=80.0)
    p.add_argument("--pj_norm_percentile_lo", type=float, default=2.0)
    p.add_argument("--pj_norm_percentile_hi", type=float, default=98.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.data_type is None:
        args.data_type = os.path.basename(args.data_root.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)

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

    dataset = KubricDenseDataset(
        data_root=args.data_root,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        max_samples=args.num_samples,
    )

    all_results = eval_dataset(predictor, dataset, args)
    aggregate_and_save(all_results, args.output_dir, args.data_type)


if __name__ == "__main__":
    main()
