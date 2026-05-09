"""WorldTrack evaluation entry-point.

Computes the TAPVid3D Sim3-closed metric with St4RTrack thresholds
({0.1, 0.3, 0.5, 1.0} m), swept over visibility thresholds (0.1..0.9).
For *_da3 / *_vipe datasets, the same metric is also computed against
the GT-camera world tracks loaded from the matching base dataset.
"""

import os, glob, argparse, json, time, math
import torch
import numpy as np
import einops
from tqdm import tqdm

from .dust3r_eval_utils import load_npz_data, estimate_sim3


def _sanitize_json(v):
    if isinstance(v, dict):
        return {k: _sanitize_json(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_json(vv) for vv in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.floating, np.integer)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(_sanitize_json(obj), f, indent=2)
    os.replace(tmp, path)


def _load_gt_world_tracks(npz_path, tracks_xyz_cam, num_frames, frame_stride):
    """Compute GT-world tracks from `extrinsics_w2c_gt` (embedded in *_da3 / *_vipe).

    Mirrors the world-coord computation that load_npz_data does for the file's
    canonical `extrinsics_w2c`, but uses the GT extrinsics co-stored as
    `extrinsics_w2c_gt`. Returns (T, N, 3) world tracks in GT camera-0 frame,
    or None if `extrinsics_w2c_gt` is not present.
    """
    in_npz = np.load(npz_path, allow_pickle=True)
    if 'extrinsics_w2c_gt' not in in_npz.files:
        return None
    total_needed = num_frames * frame_stride
    extr = in_npz['extrinsics_w2c_gt'][:total_needed:frame_stride].astype(np.float64)
    # Frame-0 normalize (matches load_npz_data convention)
    inv0 = np.linalg.inv(extr[0])
    extr = np.stack([extr[t] @ inv0 for t in range(extr.shape[0])])
    c2w = np.linalg.inv(extr)
    cam = tracks_xyz_cam[:num_frames].astype(np.float64)   # (T, N, 3)
    out = np.zeros_like(cam)
    for t in range(cam.shape[0]):
        out[t] = (c2w[t, :3, :3] @ cam[t].T).T + c2w[t, :3, 3]
    return out


###############################################################################
# TAPVid3D metric (Sim3-closed alignment, fixed St4RTrack thresholds)
###############################################################################

ST4RTRACK_THRESHOLDS_M = [0.1, 0.3, 0.5, 1.0]


def compute_tapvid3d_st4rtrack(gt_occluded, gt_tracks, pred_occluded, pred_tracks):
    """TAPVid3D metric in St4RTrack convention.

    Sim3-closed alignment on visible points; pts/jaccard at fixed thresholds
    {0.1, 0.3, 0.5, 1.0} m; OA from per-frame occlusion equality.

    All inputs in (T, N, ...) order. Returns dict with average_jaccard /
    average_pts_within_thresh / occlusion_accuracy / per-threshold values.
    """
    # Rearrange to (1, N, T, ...) batch format
    output_order = "() n t"
    order = "t n"
    gt_occluded = einops.rearrange(gt_occluded, f"{order} -> {output_order}")
    pred_occluded = einops.rearrange(pred_occluded, f"{order} -> {output_order}")
    gt_tracks = einops.rearrange(gt_tracks, f"{order} d -> {output_order} d")
    pred_tracks = einops.rearrange(pred_tracks, f"{order} d -> {output_order} d")

    # Sim3-closed alignment on visible points
    visible_flat = ~gt_occluded[0]
    gt_flat = gt_tracks[0]
    pred_flat = pred_tracks[0]
    gt_vis = gt_flat[visible_flat]
    pred_vis = pred_flat[visible_flat]
    if len(gt_vis) >= 3:
        s, R, t = estimate_sim3(pred_vis, gt_vis, ransac=False)
        orig_shape = pred_tracks.shape
        p = pred_tracks.reshape(-1, 3)
        p_aligned = s * (R @ p.T).T + t
        pred_tracks = p_aligned.reshape(orig_shape)

    eval_w = np.ones(gt_occluded.shape)
    sum_axes = (-2, -1)
    metrics = {}

    metrics["occlusion_accuracy"] = (np.sum(
        np.equal(pred_occluded, gt_occluded) * eval_w, axis=sum_axes
    ) / np.sum(eval_w, axis=sum_axes)).item()

    visible = ~gt_occluded
    pred_visible = ~pred_occluded

    all_frac, all_jac = [], []
    for thresh in ST4RTRACK_THRESHOLDS_M:
        within = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < thresh ** 2

        is_correct = np.logical_and(within, visible)
        count_correct = np.sum(is_correct * eval_w, axis=sum_axes)
        count_visible = np.sum(visible * eval_w, axis=sum_axes)
        frac_correct = count_correct / count_visible
        metrics[f"pts_within_{thresh}"] = frac_correct.item()
        all_frac.append(frac_correct)

        true_pos = np.sum((is_correct & pred_visible) * eval_w, axis=sum_axes)
        gt_pos = np.sum(visible * eval_w, axis=sum_axes)
        false_pos = (~visible) & pred_visible
        false_pos = false_pos | ((~within) & pred_visible)
        false_pos = np.sum(false_pos * eval_w, axis=sum_axes)
        jac = true_pos / (gt_pos + false_pos)
        metrics[f"jaccard_{thresh}"] = jac.item()
        all_jac.append(jac)

    metrics["average_pts_within_thresh"] = float(np.mean(np.stack(all_frac, axis=-1)))
    metrics["average_jaccard"] = float(np.mean(np.stack(all_jac, axis=-1)))
    return metrics


###############################################################################
# Per-vis-threshold metric sweep
###############################################################################

def compute_metric_sweep(gt_tracks, pred_tracks, vis_mask, vis_pred):
    """Compute (AJ, OC_ACC, pts) at each vis threshold ∈ {0.1..0.9}.

    gt_tracks/pred_tracks: (T, N, 3). vis_mask: (T, N) GT visibility.
    vis_pred: (T, N) predicted visibility prob in [0, 1].
    """
    gt_occ = ~vis_mask
    out = {}
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_occ = (vis_pred < thr)
        m = compute_tapvid3d_st4rtrack(gt_occ, gt_tracks, pred_occ, pred_tracks)
        tag = f'th{int(thr * 10):02d}'
        for k, v in m.items():
            out[f'{k}_{tag}'] = v
    return out


###############################################################################
# Prediction saving
###############################################################################

def save_prediction_as_npy(pred_tracks, save_dir, video_name):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(pred_tracks, torch.Tensor):
        pred_tracks = pred_tracks.detach().cpu().numpy()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{video_name}_track_{timestamp}.npy")
    np.save(save_path, pred_tracks)
    print(f"Saved prediction to {save_path}")
    return save_path


###############################################################################
# Single-NPZ eval (default mode + interleaved-stride mode)
###############################################################################

def _load_depth(npz_path, num_frames, frame_stride):
    """Load `depth_map` (DA3 / ViPE / GT, depending on the variant)."""
    in_npz = np.load(npz_path, allow_pickle=True)
    if 'depth_map' not in in_npz.files:
        return None
    total_needed = num_frames * frame_stride
    return in_npz['depth_map'][:total_needed:frame_stride]


def eval_single_npz(predictor, npz_path, num_frames, frame_stride=1,
                    save_predictions=False, save_dense=False, pred_dir=None):
    """Evaluate a single NPZ (frame-stride mode)."""
    (video_list, tracks_xyz_cam, tracks_uv, intrinsics, tracks_xyz_world, visibility,
     video_name, extrinsics_w2c) = load_npz_data(
        npz_path, num_frames=num_frames, frame_stride=frame_stride)

    # GT-camera world tracks: compute from the same NPZ's extrinsics_w2c_gt.
    gt_world_gtcam = _load_gt_world_tracks(
        npz_path, tracks_xyz_cam, num_frames, frame_stride)

    depth_map = _load_depth(npz_path, num_frames, frame_stride)

    visibility_mask = visibility[0]
    if visibility_mask.sum() == 0:
        print(f"Warning: No visible points in {video_name}")
        return None, None
    track_mask = visibility_mask.astype(bool)

    query_uv = np.array(tracks_uv)[0, track_mask]
    gt_tracks = tracks_xyz_world[:num_frames, track_mask]
    gt_tracks_gtcam = (gt_world_gtcam[:num_frames, track_mask]
                       if gt_world_gtcam is not None else None)
    vis_all = visibility[:num_frames, track_mask]

    pred_tracks = predictor.predict(
        video_list, query_uv, vis_all, intrinsics,
        depth_map=depth_map, extrinsics_w2c=extrinsics_w2c,
    )

    # Apply OOB mask (predictor filters query points outside model resolution)
    oob_mask = getattr(predictor, '_last_oob_mask', None)
    if oob_mask is not None and oob_mask.sum() < len(oob_mask):
        gt_tracks = gt_tracks[:, oob_mask]
        if gt_tracks_gtcam is not None:
            gt_tracks_gtcam = gt_tracks_gtcam[:, oob_mask]
        vis_all = vis_all[:, oob_mask]

    saved_path = None
    if save_predictions and pred_dir:
        saved_path = save_prediction_as_npy(pred_tracks, pred_dir, video_name)
        if save_dense and getattr(predictor, '_last_row_dense', None) is not None:
            dense_dir = os.path.join(pred_dir, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            dense_path = os.path.join(dense_dir, f"{video_name}.npz")
            data = {'row': predictor._last_row_dense,
                    'rgb': predictor._last_rgb_frames}
            if getattr(predictor, '_last_pj_input', None) is not None:
                data['pj_input'] = predictor._last_pj_input
            np.savez_compressed(dense_path, **data)
            print(f"  Saved dense: {dense_path}")

    # Per-query visibility predictions: sample _last_vis_dense at model-space query UVs
    vis_pred_per_q = None
    vis_dense = getattr(predictor, '_last_vis_dense', None)
    q_uv_model = getattr(predictor, '_last_query_uv_model', None)
    if vis_dense is not None and q_uv_model is not None:
        H_vd, W_vd = vis_dense.shape[-2], vis_dense.shape[-1]
        u_int = np.clip(q_uv_model[:, 0].astype(np.int64), 0, W_vd - 1)
        v_int = np.clip(q_uv_model[:, 1].astype(np.int64), 0, H_vd - 1)
        vis_pred_per_q = vis_dense[:, v_int, u_int]
        if oob_mask is not None and oob_mask.sum() < len(oob_mask):
            vis_pred_per_q = vis_pred_per_q[:, oob_mask]

    if gt_tracks.shape != pred_tracks.shape:
        print(f"Warning: shape mismatch gt={gt_tracks.shape}, pred={pred_tracks.shape}")
    if vis_pred_per_q is None:
        print("Warning: predictor produced no visibility — metric sweep needs --predict_vis")
        return None, saved_path
    if gt_tracks_gtcam is None:
        print(f"Warning: {npz_path} has no extrinsics_w2c_gt; cannot compute the metric")
        return None, saved_path

    # Sim3-align predictions with GT trajectories in the GT camera frame.
    result_dict = compute_metric_sweep(gt_tracks_gtcam, pred_tracks, vis_all, vis_pred_per_q)
    result_dict['video_name'] = video_name

    print(f"{video_name}:")
    for k, v in result_dict.items():
        if k != 'video_name':
            print(f"  {k}: {v}")
    print()
    return result_dict, saved_path


def eval_single_npz_interleaved_stride(predictor, npz_path, total_frames,
                                        interleave_stride=2, eval_frames=0,
                                        save_predictions=False, save_dense=False,
                                        pred_dir=None):
    """Interleaved-stride evaluation for long videos.

    Runs the model `interleave_stride` times with different temporal offsets.
    Each run shares the frame-0 query (no chaining, no drift). Predictions
    are merged by averaging overlapping frames.
    """
    (video_list_full, tracks_xyz_cam_full, tracks_uv_full, intrinsics,
     tracks_xyz_world_full, visibility_full, video_name,
     extrinsics_w2c_full) = load_npz_data(
        npz_path, num_frames=total_frames, frame_stride=1)

    T = len(video_list_full)
    if T < interleave_stride + 1:
        return eval_single_npz(predictor, npz_path, T,
                                save_predictions=save_predictions, save_dense=save_dense,
                                pred_dir=pred_dir)

    gt_world_gtcam = _load_gt_world_tracks(
        npz_path, tracks_xyz_cam_full, total_frames, frame_stride=1)

    in_npz = np.load(npz_path, allow_pickle=True)
    depth_map_full = in_npz['depth_map'][:T] if 'depth_map' in in_npz.files else None
    visibility_mask = visibility_full[0]
    if visibility_mask.sum() == 0:
        print(f"Warning: No visible points in {video_name}")
        return None, None
    track_mask = visibility_mask.astype(bool)

    query_uv = np.array(tracks_uv_full)[0, track_mask]
    gt_tracks = tracks_xyz_world_full[:T, track_mask]
    gt_tracks_gtcam = (gt_world_gtcam[:T, track_mask]
                       if gt_world_gtcam is not None else None)
    vis_all = visibility_full[:T, track_mask]
    M = query_uv.shape[0]

    run_indices = []
    for offset in range(interleave_stride):
        if offset == 0:
            run_indices.append(list(range(0, T, interleave_stride)))
        else:
            run_indices.append([0] + list(range(offset, T, interleave_stride)))

    print(f"  {video_name}: T={T}, interleaved stride={interleave_stride}, "
          f"{len(run_indices)} runs, frames per run: {[len(idx) for idx in run_indices]}")

    pred_sum = np.zeros((T, M, 3), dtype=np.float64)
    pred_count = np.zeros((T, M), dtype=np.float64)
    vis_sum = np.zeros((T, M), dtype=np.float64)
    vis_count = np.zeros((T, M), dtype=np.float64)
    global_oob_mask = None
    dense_row_sum = None; dense_row_cnt = None
    dense_pj_sum = None;  dense_pj_cnt = None
    dense_rgb_first = None; dense_rgb_cnt = None

    for ri, indices in enumerate(run_indices):
        print(f"  Run {ri}: {len(indices)} frames, indices=[{indices[0]}..{indices[-1]}]")
        video_list_run = [video_list_full[i] for i in indices]
        extrinsics_run = extrinsics_w2c_full[indices]
        depth_run = depth_map_full[indices] if depth_map_full is not None else None
        vis_run = vis_all[indices]

        pred_run = predictor.predict(
            video_list_run, query_uv, vis_run, intrinsics,
            depth_map=depth_run, extrinsics_w2c=extrinsics_run,
        )
        if pred_run is None:
            print(f"  Run {ri}: predictor returned None, skip")
            continue

        oob_mask = getattr(predictor, '_last_oob_mask', None)
        if oob_mask is not None and oob_mask.sum() < M:
            if ri == 0:
                global_oob_mask = oob_mask
            active_idx = np.where(oob_mask)[0]
        else:
            active_idx = np.arange(M)

        for fi, frame_idx in enumerate(indices):
            if fi < pred_run.shape[0]:
                pred_sum[frame_idx, active_idx] += pred_run[fi]
                pred_count[frame_idx, active_idx] += 1.0

        vis_dense_run = getattr(predictor, '_last_vis_dense', None)
        q_uv_model = getattr(predictor, '_last_query_uv_model', None)
        if vis_dense_run is not None and q_uv_model is not None:
            H_vd, W_vd = vis_dense_run.shape[-2], vis_dense_run.shape[-1]
            u_int = np.clip(q_uv_model[active_idx, 0].astype(np.int64), 0, W_vd - 1)
            v_int = np.clip(q_uv_model[active_idx, 1].astype(np.int64), 0, H_vd - 1)
            for fi, frame_idx in enumerate(indices):
                if fi < vis_dense_run.shape[0]:
                    vis_sum[frame_idx, active_idx] += vis_dense_run[fi, v_int, u_int]
                    vis_count[frame_idx, active_idx] += 1.0

        # Optional dense merge for --save_dense
        if save_dense:
            row_run = getattr(predictor, '_last_row_dense', None)
            if row_run is not None:
                if dense_row_sum is None:
                    H_r, W_r, C_r = row_run.shape[1], row_run.shape[2], row_run.shape[3]
                    dense_row_sum = np.zeros((T, H_r, W_r, C_r), dtype=np.float32)
                    dense_row_cnt = np.zeros((T,), dtype=np.float32)
                for fi, frame_idx in enumerate(indices):
                    if fi < row_run.shape[0]:
                        dense_row_sum[frame_idx] += row_run[fi].astype(np.float32)
                        dense_row_cnt[frame_idx] += 1.0
            pj_run = getattr(predictor, '_last_pj_input', None)
            if pj_run is not None:
                if dense_pj_sum is None:
                    H_p, W_p, C_p = pj_run.shape[1], pj_run.shape[2], pj_run.shape[3]
                    dense_pj_sum = np.zeros((T, H_p, W_p, C_p), dtype=np.float32)
                    dense_pj_cnt = np.zeros((T,), dtype=np.float32)
                for fi, frame_idx in enumerate(indices):
                    if fi < pj_run.shape[0]:
                        dense_pj_sum[frame_idx] += pj_run[fi].astype(np.float32)
                        dense_pj_cnt[frame_idx] += 1.0
            rgb_run = getattr(predictor, '_last_rgb_frames', None)
            if rgb_run is not None:
                if dense_rgb_first is None:
                    dense_rgb_first = np.zeros(
                        (T, rgb_run.shape[1], rgb_run.shape[2], 3), dtype=np.uint8)
                    dense_rgb_cnt = np.zeros((T,), dtype=np.bool_)
                for fi, frame_idx in enumerate(indices):
                    if fi < rgb_run.shape[0] and not dense_rgb_cnt[frame_idx]:
                        dense_rgb_first[frame_idx] = rgb_run[fi]
                        dense_rgb_cnt[frame_idx] = True

    pred_final = np.where(pred_count[..., None] > 0,
                          pred_sum / np.maximum(pred_count[..., None], 1e-10), 0.0)
    vis_final = np.where(vis_count > 0, vis_sum / np.maximum(vis_count, 1e-10), 0.5)
    has_vis_pred = vis_count.sum() > 0

    if global_oob_mask is not None and global_oob_mask.sum() < M:
        pred_final = pred_final[:, global_oob_mask]
        gt_tracks = gt_tracks[:, global_oob_mask]
        if gt_tracks_gtcam is not None:
            gt_tracks_gtcam = gt_tracks_gtcam[:, global_oob_mask]
        vis_all = vis_all[:, global_oob_mask]
        vis_final = vis_final[:, global_oob_mask]

    if eval_frames > 0 and eval_frames < pred_final.shape[0]:
        print(f"  Truncating for eval: {pred_final.shape[0]} -> {eval_frames} frames")
        pred_final = pred_final[:eval_frames]
        gt_tracks = gt_tracks[:eval_frames]
        if gt_tracks_gtcam is not None:
            gt_tracks_gtcam = gt_tracks_gtcam[:eval_frames]
        vis_all = vis_all[:eval_frames]
        vis_final = vis_final[:eval_frames]

    saved_path = None
    if save_predictions and pred_dir:
        saved_path = save_prediction_as_npy(pred_final, pred_dir, video_name)
        if save_dense and dense_row_sum is not None and dense_row_cnt.sum() > 0:
            dense_dir = os.path.join(pred_dir, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            dense_path = os.path.join(dense_dir, f"{video_name}.npz")
            cnt_row = np.maximum(dense_row_cnt, 1e-10)[:, None, None, None]
            data = {'row': (dense_row_sum / cnt_row).astype(np.float32)}
            if dense_rgb_first is not None:
                data['rgb'] = dense_rgb_first
            if dense_pj_sum is not None and dense_pj_cnt.sum() > 0:
                cnt_p = np.maximum(dense_pj_cnt, 1e-10)[:, None, None, None]
                data['pj_input'] = (dense_pj_sum / cnt_p).astype(np.float32)
            np.savez_compressed(dense_path, **data)
            print(f"  Saved dense: {dense_path}")

    if not has_vis_pred:
        print("Warning: predictor produced no visibility — metric sweep needs --predict_vis")
        return None, saved_path
    if gt_tracks_gtcam is None:
        print(f"Warning: {npz_path} has no extrinsics_w2c_gt; cannot compute the metric")
        return None, saved_path

    # Sim3-align predictions with GT trajectories in the GT camera frame.
    result_dict = compute_metric_sweep(gt_tracks_gtcam, pred_final, vis_all, vis_final)
    result_dict['video_name'] = video_name

    print(f"{video_name}:")
    for k, v in result_dict.items():
        if k != 'video_name':
            print(f"  {k}: {v}")
    print()
    return result_dict, saved_path


###############################################################################
# Aggregation + output
###############################################################################

def aggregate_results(all_results):
    """Average all numeric metric keys across sequences."""
    keys = set()
    for res in all_results:
        for k in res.keys():
            if k == 'video_name':
                continue
            keys.add(k)

    finals = {}
    for k in sorted(keys):
        vals = [res[k] for res in all_results
                if k in res and res[k] is not None and not math.isnan(res[k])]
        finals[k] = float(np.mean(vals)) if vals else float('nan')
    return finals


def _print_st4rtrack_table(log_str, finals, prefix, label):
    """Append one (AJ / OC_ACC / pts) table for thr ∈ {0.1..0.9}."""
    th_tags = sorted({
        k.split('_')[-1] for k in finals
        if k.startswith(prefix) and any(k.endswith(f'_th{i:02d}') for i in range(1, 10))
    })
    if not th_tags:
        return log_str
    log_str += f"\n{label} across vis thresholds:\n"
    log_str += f"  {'thr':>5} | {'AJ':>7} | {'OC_ACC':>7} | {'pts':>7}\n"
    for t in th_tags:
        thr_val = int(t[2:]) / 10.0
        aj  = finals.get(f'{prefix}average_jaccard_{t}',           float('nan'))
        oc  = finals.get(f'{prefix}occlusion_accuracy_{t}',        float('nan'))
        pts = finals.get(f'{prefix}average_pts_within_thresh_{t}', float('nan'))
        log_str += f"  {thr_val:>5.1f} | {aj:>7.4f} | {oc:>7.4f} | {pts:>7.4f}\n"
    return log_str


def print_and_save_results(output_dir, data_type, finals):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"track_eval_{data_type}.txt")

    log_str = "\n=== Final Results ===\n"
    log_str = _print_st4rtrack_table(
        log_str, finals, prefix='',
        label="TAPVid3D Sim3 closed (st4rtrack threshold)")

    print(log_str)
    with open(filepath, "a") as f:
        f.write(f"\n=== Evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(log_str)
    print(f"Results saved to {filepath}")


###############################################################################
# CLI
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="WorldTrack evaluation")
    parser.add_argument("--model_type", type=str, default="wan_scene_flow",
                        choices=["wan_scene_flow"])
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--lora_rank", type=int, default=1024)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)

    parser.add_argument("--regression_timestep", type=int, default=-1)
    parser.add_argument("--diagonal_condition_row", default=False, action="store_true")
    parser.add_argument("--pj_norm_inlier", default=False, action="store_true")
    parser.add_argument("--diag_max_depth", type=float, default=80.0)
    parser.add_argument("--pj_norm_percentile_lo", type=float, default=2.0)
    parser.add_argument("--pj_norm_percentile_hi", type=float, default=98.0)
    parser.add_argument("--pixel_delta", default=False, action="store_true")
    parser.add_argument("--track_latent_length", type=int, default=12)
    parser.add_argument("--resize_mode", type=str, default="stretch", choices=["pad", "stretch"])
    parser.add_argument("--predict_vis", default=False, action="store_true")
    parser.add_argument("--vis_separate_decoder", default=False, action="store_true")
    parser.add_argument("--pj_separate_encoder", default=False, action="store_true")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--data_types", nargs="+", required=True)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=50)

    parser.add_argument("--interleaved_stride", action="store_true", default=False)
    parser.add_argument("--interleave_stride", type=int, default=2)
    parser.add_argument("--eval_frames", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_dense", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def build_predictor(args):
    from .wan_scene_flow_predictor import WanSceneFlowPredictor
    return WanSceneFlowPredictor(
        checkpoint_path=args.checkpoint_path,
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        height=args.height,
        width=args.width,
        device=args.device,
        regression_timestep=args.regression_timestep,
        diagonal_condition_row=args.diagonal_condition_row,
        pj_norm_inlier=args.pj_norm_inlier,
        diag_max_depth=args.diag_max_depth,
        pj_norm_percentile_lo=args.pj_norm_percentile_lo,
        pj_norm_percentile_hi=args.pj_norm_percentile_hi,
        pixel_delta=args.pixel_delta,
        track_latent_length=args.track_latent_length,
        resize_mode=args.resize_mode,
        predict_vis=args.predict_vis,
        vis_separate_decoder=args.vis_separate_decoder,
        pj_separate_encoder=args.pj_separate_encoder,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    predictor = build_predictor(args)

    for data_type in args.data_types:
        npz_dir = os.path.join(args.data_root, data_type)
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not npz_files:
            print(f"No .npz files found in {npz_dir}")
            continue
        if args.num_samples > 0:
            npz_files = npz_files[:args.num_samples]

        print(f"\n{'='*60}\nEvaluating {data_type}: {len(npz_files)} sequences\n{'='*60}")
        pred_dir = os.path.join(args.output_dir, f"saved_predictions_{data_type}") \
            if args.save_predictions else None

        all_results = []
        saved_prediction_paths = []
        for npz_path in tqdm(npz_files, desc=data_type):
            try:
                if args.interleaved_stride:
                    result, saved_path = eval_single_npz_interleaved_stride(
                        predictor, npz_path, args.num_frames,
                        interleave_stride=args.interleave_stride,
                        eval_frames=args.eval_frames,
                        save_predictions=args.save_predictions,
                        save_dense=args.save_dense,
                        pred_dir=pred_dir,
                    )
                else:
                    result, saved_path = eval_single_npz(
                        predictor, npz_path, args.num_frames,
                        frame_stride=args.frame_stride,
                        save_predictions=args.save_predictions,
                        save_dense=args.save_dense,
                        pred_dir=pred_dir,
                    )
            finally:
                torch.cuda.empty_cache()

            if result is not None:
                all_results.append(result)
                try:
                    _atomic_write_json(
                        os.path.join(args.output_dir, f"per_video_{data_type}.json"),
                        all_results,
                    )
                except Exception as _e:
                    print(f"  [warn] live per_video dump failed: {_e}")
            if saved_path:
                saved_prediction_paths.append((npz_path, saved_path))

        if all_results:
            finals = aggregate_results(all_results)
            print_and_save_results(args.output_dir, data_type, finals)
        else:
            print(f"[WARN] No valid results for {data_type}")

        per_video_path = os.path.join(args.output_dir, f"per_video_{data_type}.json")
        _atomic_write_json(per_video_path, all_results)
        print(f"Per-video results saved to {per_video_path}")

        if args.save_predictions and saved_prediction_paths:
            mapping_file = os.path.join(args.output_dir, f"prediction_mapping_{data_type}.json")
            mapping = {
                npz_path: {
                    "prediction_path": pred_path,
                    "video_name": os.path.splitext(os.path.basename(npz_path))[0],
                    "eval_recon": False,
                }
                for npz_path, pred_path in saved_prediction_paths
            }
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            print(f"Saved prediction mapping to {mapping_file}")


if __name__ == "__main__":
    main()
