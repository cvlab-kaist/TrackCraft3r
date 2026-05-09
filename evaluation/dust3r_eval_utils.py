"""Self-contained NPZ loader + Sim(3) alignment helpers used by eval_worldtrack.

Extracted from St4RTrack (https://github.com/HavenFeng/St4RTrack):
    dust3r/datasets/tapvid3d.py  -> load_npz_data, project_points_to_video_frame
    dust3r/track_eval_util.py    -> estimate_sim3
"""

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm



def project_points_to_video_frame(camera_pov_points3d, camera_intrinsics, height, width):
    u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
    v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)
    f_u, f_v, c_u, c_v = camera_intrinsics
    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v
    masks = camera_pov_points3d[..., 2] >= 1
    masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
    return np.stack([u_d, v_d], axis=-1), masks


def load_npz_data(npz_path, num_frames=None, frame_stride=1, normalize_cam=True):
    """Load a WorldTrack NPZ.

    Required keys: images_jpeg_bytes, tracks_XYZ, fx_fy_cx_cy, visibility.
    Optional: extrinsics_w2c, depth_map, fx_fy_cx_cy_vipe, intrinsics_da3, ...
    """
    in_npz = np.load(npz_path, allow_pickle=True)

    images_jpeg_bytes = in_npz["images_jpeg_bytes"]
    tracks_xyz_cam = in_npz["tracks_XYZ"]
    intrinsics = in_npz["fx_fy_cx_cy"]
    visibility = in_npz["visibility"]

    extrinsics_w2c = in_npz['extrinsics_w2c'] if 'extrinsics_w2c' in in_npz.files else None
    print(f"Loaded {len(images_jpeg_bytes)} frames from {npz_path}, subsampled to {num_frames} (stride={frame_stride})")

    if num_frames is not None:
        total_needed = num_frames * frame_stride
        images_jpeg_bytes = images_jpeg_bytes[:total_needed:frame_stride]
        tracks_xyz_cam = tracks_xyz_cam[:total_needed:frame_stride]
        visibility = visibility[:total_needed:frame_stride]
        if extrinsics_w2c is not None:
            extrinsics_w2c = extrinsics_w2c[:total_needed:frame_stride]

    video_list = []
    for frame_bytes in images_jpeg_bytes:
        arr = np.frombuffer(frame_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        video_list.append(Image.fromarray(image_rgb))

    h, w = video_list[0].height, video_list[0].width
    tracks_uv, _ = project_points_to_video_frame(tracks_xyz_cam, intrinsics, h, w)

    if normalize_cam and (extrinsics_w2c is not None):
        first_inv = np.linalg.inv(extrinsics_w2c[0])
        for i in range(extrinsics_w2c.shape[0]):
            extrinsics_w2c[i] = extrinsics_w2c[i] @ first_inv

    if extrinsics_w2c is not None:
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
        tracks_xyz_world = np.zeros_like(tracks_xyz_cam)
        for i in range(tracks_xyz_cam.shape[0]):
            R = extrinsics_c2w[i, :3, :3]
            t = extrinsics_c2w[i, :3, 3]
            tracks_xyz_world[i] = (R @ tracks_xyz_cam[i].T).T + t
    else:
        tracks_xyz_world = tracks_xyz_cam
        extrinsics_w2c = np.tile(np.eye(4), (num_frames, 1, 1))

    video_name = os.path.splitext(os.path.basename(npz_path))[0]
    return (video_list, tracks_xyz_cam, tracks_uv, intrinsics,
            tracks_xyz_world, visibility, video_name, extrinsics_w2c)


def _estimate_sim3_closed_form(A, B):
    centroidA = A.mean(axis=0, keepdims=True)
    centroidB = B.mean(axis=0, keepdims=True)
    A_ = A - centroidA
    B_ = B - centroidB
    H = A_.T @ B_
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    varA = (A_ ** 2).sum()
    s_ = np.sum(S) / (varA + 1e-12)
    t_ = centroidB[0] - s_ * R_ @ centroidA[0]
    return s_, R_, t_


def estimate_sim3(A, B, ransac=True, ransac_iterations=1000,
                  inlier_threshold=0.05, refine_with_inliers=True):
    """Sim(3) alignment of A to B (B ≈ s·R·A + t)."""
    A = np.asarray(A); B = np.asarray(B)
    assert A.shape == B.shape
    N = A.shape[0]
    if N < 3:
        raise ValueError("Need >=3 points to estimate Sim(3).")
    if not ransac:
        return _estimate_sim3_closed_form(A, B)

    best_inliers_count = -1
    best_model = None
    best_inliers_mask = None
    rng = np.random.default_rng()
    for _ in tqdm(range(ransac_iterations)):
        subset_idx = rng.choice(N, size=3, replace=False)
        try:
            s_m, R_m, t_m = _estimate_sim3_closed_form(A[subset_idx], B[subset_idx])
        except np.linalg.LinAlgError:
            continue
        A_transformed = s_m * (R_m @ A.T).T + t_m
        dists = np.linalg.norm(A_transformed - B, axis=1)
        inliers_mask = (dists < inlier_threshold)
        inliers_count = int(np.sum(inliers_mask))
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inliers_mask = inliers_mask
            best_model = (s_m, R_m, t_m)

    if best_model is None:
        print("[WARN] RANSAC failed; falling back to closed-form on all points.")
        return _estimate_sim3_closed_form(A, B)

    if refine_with_inliers and best_inliers_mask is not None:
        inlierA = A[best_inliers_mask]
        inlierB = B[best_inliers_mask]
        if len(inlierA) >= 3:
            return _estimate_sim3_closed_form(inlierA, inlierB)
    return best_model


