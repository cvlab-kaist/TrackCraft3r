"""
Visualization utilities for synthetic training data.
Used by both visualize_synthetic_data.py and train.py debug section.
"""

import os
import numpy as np
import imageio


def save_video(frames, path, fps=8):
    """Save list of numpy arrays (H, W, 3) uint8 as MP4."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


def render_rgb_video(sample):
    """Extract RGB frames from sample dict. video: list of PIL images."""
    return [np.array(img) for img in sample['video']]


def render_depth_video(sample):
    """Render depth maps as grayscale video (log-scale for large dynamic range)."""
    depth = sample['depth'].numpy()[:, 0]  # (T, H, W)
    frames = []
    for t in range(depth.shape[0]):
        d = depth[t]
        valid = (d > 1e-6) & np.isfinite(d)
        if valid.sum() > 0:
            log_d = np.zeros_like(d)
            log_d[valid] = np.log(d[valid])
            d_min, d_max = log_d[valid].min(), log_d[valid].max()
            d_norm = np.clip(1.0 - (log_d - d_min) / (d_max - d_min + 1e-8), 0, 1)
            d_norm[~valid] = 0
        else:
            d_norm = np.zeros_like(d)
        d_vis = (d_norm * 255).astype(np.uint8)
        frames.append(np.stack([d_vis, d_vis, d_vis], axis=-1))
    return frames


def render_point_map_birdseye(sample, elevation=45, azimuth=-20, canvas_size=512):
    """Bird's eye view of 3D point cloud from depth + camera."""
    depth = sample['depth'].numpy()       # (T, 1, H, W)
    intrinsic = sample['intrinsic'].numpy()
    extrinsic = sample['extrinsic'].numpy()

    T, _, H, W = depth.shape
    depth = depth[:, 0]  # (T, H, W)

    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    w2c_0 = np.linalg.inv(extrinsic[0])
    video_frames = render_rgb_video(sample)

    all_points, all_valid, all_colors = [], [], []
    for t in range(T):
        d = depth[t]
        K = intrinsic[t]
        c2w_t = extrinsic[t]
        valid = (d > 1e-6) & np.isfinite(d)
        d_clean = np.where(valid, d, 1.0)
        uvd = np.stack([u_coords * d_clean, v_coords * d_clean, d_clean], axis=-1)
        P_cam = uvd.reshape(-1, 3) @ np.linalg.inv(K).T
        T_rel = w2c_0 @ c2w_t
        P_homo = np.concatenate([P_cam, np.ones((H * W, 1), dtype=np.float32)], axis=-1)
        P_cam0 = (P_homo @ T_rel.T)[:, :3]
        all_points.append(P_cam0)
        all_valid.append(valid.reshape(-1))
        all_colors.append(video_frames[t].reshape(-1, 3))

    points = np.stack(all_points)    # (T, H*W, 3)
    valid_mask = np.stack(all_valid) # (T, H*W)
    colors = np.stack(all_colors)    # (T, H*W, 3)

    el, az = np.radians(elevation), np.radians(azimuth)
    Ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]], dtype=np.float32)
    R_view = Rx @ Ry

    all_rot = points[valid_mask].reshape(-1, 3) @ R_view.T
    if len(all_rot) == 0:
        return [np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)] * T
    gx_min, gx_max = all_rot[:, 0].min(), all_rot[:, 0].max()
    gy_min, gy_max = all_rot[:, 1].min(), all_rot[:, 1].max()
    max_range = max(gx_max - gx_min, gy_max - gy_min) * 1.1 or 1.0
    x_center = (gx_min + gx_max) / 2
    y_center = (gy_min + gy_max) / 2

    frames_out = []
    for t in range(T):
        valid_t = valid_mask[t]
        pts_rot = points[t][valid_t] @ R_view.T
        rgb = colors[t][valid_t]
        px = np.round(((pts_rot[:, 0] - x_center) / max_range + 0.5) * canvas_size).astype(np.int64)
        py = np.round(((pts_rot[:, 1] - y_center) / max_range + 0.5) * canvas_size).astype(np.int64)
        order = np.argsort(-pts_rot[:, 2])
        px, py, rgb = px[order], py[order], rgb[order]
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        ib = (px >= 0) & (px < canvas_size) & (py >= 0) & (py < canvas_size)
        canvas[py[ib], px[ib]] = rgb[ib]
        frames_out.append(canvas)
    return frames_out


def _normalize_c2w_np(extrinsic, camera_stats=None):
    """Normalize c2w matrices (numpy version of normalize_c2w_torch in train.py)."""
    c2w = extrinsic.astype(np.float32).copy()
    c2w_0_inv = np.linalg.inv(c2w[0])
    c2w_norm = c2w_0_inv[None] @ c2w  # (T, 4, 4), frame 0 → identity rotation

    if camera_stats is not None:
        mean_center = np.asarray(camera_stats['mean']).reshape(1, 3).astype(np.float32)
        max_dist = float(camera_stats['max'])
    else:
        centers = c2w_norm[:, :3, 3]
        mean_center = centers.mean(axis=0, keepdims=True)
        max_dist = float(np.max(np.linalg.norm(centers - mean_center, axis=-1)))

    c2w_norm[:, :3, 3] -= mean_center
    c2w_norm[:, :3, 3] *= 1.0 / (max_dist + 1e-6)
    return c2w_norm


def render_colored_trajectory(sample):
    """Colored trajectories projected to 2D. Colors: R=u/W, G=v/H, B=1/depth@frame0."""
    traj3d = sample['traj3d'].numpy()        # (T, N, 3)
    vis = sample['vis'].numpy()              # (T, N)
    intrinsic = sample['intrinsic'].numpy()
    extrinsic = sample['extrinsic'].numpy()

    T, N, _ = traj3d.shape
    H = sample['depth'].shape[2]
    W = sample['depth'].shape[3]
    is_dynpose = sample.get('dataset_type') == 'dynpose'

    # DynPose traj3d is stored in normalized camera coordinates,
    # so we must normalize extrinsics to match. Synthetic data (Kubric, PO, DR)
    # stores traj3d in raw world coordinates — use raw extrinsics.
    if is_dynpose:
        camera_stats = sample.get('camera_stats', None)
        extrinsic_norm = _normalize_c2w_np(extrinsic, camera_stats)
    else:
        extrinsic_norm = extrinsic

    trajs_uv = np.zeros((T, N, 2), dtype=np.float32)
    trajs_depth = np.zeros((T, N), dtype=np.float32)

    # Project ALL frames (including frame 0) to get proper depth for 1/z color
    for t in range(T):
        w2c = np.linalg.inv(extrinsic_norm[t])
        K = intrinsic[t]
        pts_homo = np.concatenate([traj3d[t], np.ones((N, 1), dtype=np.float32)], axis=-1)
        pts_cam = (w2c @ pts_homo.T).T[:, :3]
        uv_homo = (K @ pts_cam.T).T
        z = uv_homo[:, 2]
        trajs_uv[t] = uv_homo[:, :2] / (z[:, None] + 1e-8)
        trajs_depth[t] = z

    # Point colors: R=x/W, G=y/H, B=1/z (same as training pipeline)
    indices = np.arange(N)
    r = (indices % W).astype(np.float32) / max(W - 1, 1)
    g = (indices // W).astype(np.float32) / max(H - 1, 1)
    z0 = np.clip(trajs_depth[0], 1e-6, 100.0)
    z_rec = 1.0 / z0
    z_rec = np.nan_to_num(z_rec, nan=1.0, posinf=1.0, neginf=1.0)
    b = (z_rec - z_rec.min()) / (z_rec.max() - z_rec.min() + 1e-8)
    point_colors = np.stack([r, g, b], axis=-1)

    frames_out = []
    for t in range(T):
        u = np.round(trajs_uv[t, :, 0]).astype(np.int64)
        v = np.round(trajs_uv[t, :, 1]).astype(np.int64)
        valid = (vis[t] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d_v = trajs_depth[t][valid]
        order = np.argsort(-d_v)
        u_s = u[valid][order]
        v_s = v[valid][order]
        c_s = point_colors[valid][order]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[v_s, u_s] = (c_s * 255).astype(np.uint8)
        frames_out.append(canvas)
    return frames_out


def render_trajectory_overlay(sample):
    """Colored trajectories overlaid on original RGB frames."""
    traj_frames = render_colored_trajectory(sample)
    rgb_frames = render_rgb_video(sample)
    overlay_frames = []
    for rgb, traj in zip(rgb_frames, traj_frames):
        mask = (traj.sum(axis=-1) > 0)
        blended = rgb.copy()
        blended[mask] = (0.4 * rgb[mask].astype(np.float32) +
                         0.6 * traj[mask].astype(np.float32)).astype(np.uint8)
        overlay_frames.append(blended)
    return overlay_frames


def save_sample_visualizations(sample, save_dir, fps=8):
    """Save 5 visualization videos for a single sample dict.

    Args:
        sample: dict with keys video (list PIL), depth (T,1,H,W), intrinsic (T,3,3),
                extrinsic (T,4,4), traj3d (T,N,3), vis (T,N)
        save_dir: directory to save 01_rgb.mp4 ... 05_traj_overlay.mp4
    """
    os.makedirs(save_dir, exist_ok=True)
    save_video(render_rgb_video(sample),              os.path.join(save_dir, "01_rgb.mp4"),          fps=fps)
    save_video(render_depth_video(sample),            os.path.join(save_dir, "02_depth.mp4"),        fps=fps)
    save_video(render_point_map_birdseye(sample),     os.path.join(save_dir, "03_pointmap_bev.mp4"), fps=fps)
    save_video(render_colored_trajectory(sample),     os.path.join(save_dir, "04_traj_color.mp4"),   fps=fps)
    save_video(render_trajectory_overlay(sample),     os.path.join(save_dir, "05_traj_overlay.mp4"), fps=fps)
    print(f"  [viz] Saved to {save_dir}/")
