"""
Synthetic dataset for training tracking_wan on Kubric, Point Odyssey, Dynamic Replica.
Outputs the same dict format as UnifiedDataset raw mode so that
train.py::_preprocess_raw_data() works unchanged.
"""

import os
import csv
import glob
import json
import random
import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter, GaussianBlur
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Utility functions (ported from St4RTrack)
# ---------------------------------------------------------------------------

def undistort_depthmap(depthmap, K):
    """Convert Euclidean depth to z-depth using intrinsics."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depthmap.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    x_factor = (uu - cx) / fx
    y_factor = (vv - cy) / fy
    return depthmap / np.sqrt(1 + x_factor ** 2 + y_factor ** 2)


def unproject_pixels(u, v, z, K, c2w):
    """Unproject 2D pixel coords + z-depth to world coordinates.

    Args:
        u, v: (N,) pixel coords
        z: (N,) z-depth
        K: (3,3) intrinsic
        c2w: (4,4) camera-to-world
    Returns:
        P_world: (N, 3) world coords
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X_cam = (u - cx) / fx * z
    Y_cam = (v - cy) / fy * z
    P_cam = np.stack([X_cam, Y_cam, z, np.ones_like(z)], axis=-1)  # (N, 4)
    P_world = (c2w @ P_cam.T).T[:, :3]
    return P_world


def load_16big_png_depth(depth_png):
    """Load Dynamic Replica depth (uint16 PNG reinterpreted as float16)."""
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def convert_ndc_to_pixel_intrinsics(focal_length_ndc, principal_point_ndc,
                                     image_width, image_height,
                                     intrinsics_format='ndc_isotropic'):
    """Convert NDC intrinsics to pixel-space (ported from St4RTrack)."""
    half_wh = np.array([image_width, image_height]) / 2.0
    if intrinsics_format.lower() == "ndc_norm_image_bounds":
        rescale = half_wh
    elif intrinsics_format.lower() == "ndc_isotropic":
        rescale = np.min(half_wh)
    else:
        raise ValueError(f"Unknown intrinsics format: {intrinsics_format}")
    focal_px = np.array(focal_length_ndc) * rescale
    pp_px = half_wh - np.array(principal_point_ndc) * rescale
    K = np.array([
        [focal_px[0], 0, pp_px[0]],
        [0, focal_px[1], pp_px[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def w2c_to_c2w(R, t):
    """Convert world-to-camera (R, t) to camera-to-world 4x4 matrix."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def kubric_cam_to_c2w(cam_rot, cam_trans):
    """Convert Kubric camera (w2c rot + trans) to c2w with OpenGL→OpenCV flip."""
    c2w = w2c_to_c2w(cam_rot, cam_trans)
    # OpenGL (Y-up, Z-back) → OpenCV (Y-down, Z-forward)
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return c2w @ flip


# Permutation matrix: OpenCV camera frame → NED body frame
# NED: x-forward, y-right, z-down;  OpenCV: x-right, y-down, z-forward
# ned_x = cv_z,  ned_y = cv_x,  ned_z = cv_y
_CV_TO_NED = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)


def tartanair_pose_to_c2w(tx, ty, tz, qx, qy, qz, qw):
    """Convert TartanAir pose (NED body frame, c2w) to OpenCV-convention c2w 4x4.

    TartanAir pose: position (tx,ty,tz) in world frame + quaternion (qx,qy,qz,qw)
    giving camera orientation in NED body frame (x-forward, y-right, z-down).
    """
    from scipy.spatial.transform import Rotation
    R_ned = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    R_c2w_cv = R_ned @ _CV_TO_NED  # body-to-world ∘ OpenCV-to-NED = OpenCV c2w
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w_cv
    c2w[:3, 3] = [tx, ty, tz]
    return c2w.astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class SyntheticAugmentation:
    """Spatial + photometric augmentation.

    Spatial: random padding, random base scale [0.75, 1.25], H/V flip 50%.
             All frames use the same scale and crop position (no per-frame drift)
             to keep RGB and traj3d GT consistent.
    Photometric: ColorJitter, GaussianBlur, Grayscale, per-frame noise.
    """

    def __init__(self, target_h, target_w, random_crop=True,
                 photometric_aug=True, aug_crop_pixels=16,
                 depth_aug=False):
        self.target_h = target_h
        self.target_w = target_w
        self.random_crop = random_crop
        self.photometric_aug = photometric_aug
        self.depth_aug = depth_aug

        if photometric_aug:
            # Photometric
            self._jitter = ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14
            )
            self._blur = GaussianBlur(11, sigma=(0.1, 2.0))
            self.color_aug_prob = 0.25
            self.blur_aug_prob = 0.25
            self.grayscale_prob = 0.1
            self.noise_prob = 0.15           # applied per-frame independently
            self.noise_sigma_range = (5, 20)
            # Spatial
            self.pad_bounds = [0, 25]        # random padding per edge
            self.resize_lim = [0.75, 1.25]   # base scale range
            self.h_flip_prob = 0.5
            self.v_flip_prob = 0.5

    def __call__(self, images, depths, intrinsics, no_spatial=False, traj_2d_frame0=None):
        """Apply augmentation.

        Args:
            images:          list of T np.ndarray (H, W, 3) uint8
            depths:          np.ndarray (T, H, W) float32
            intrinsics:      np.ndarray (T, 3, 3) float32
            no_spatial:      if True, skip all spatial transforms (for DynPose at native res)
            traj_2d_frame0:  optional (M, 2) array of 2D track positions at frame 0
                             (pixel coords in original image space, before any augmentation).
                             Used to center the crop window on visible tracks (MV-TAP style).

        Returns:
            images_out, depths_out, intrinsics_out, crop_info

        crop_info keys:
            sx_list, sy_list: per-frame x/y scale (list of T floats, all identical)
            left_list, top_list: per-frame crop offset (list of T ints, all identical)
            pad_x0, pad_x1, pad_y0, pad_y1: padding applied to all frames
            scale, left, top: frame-0 values (backward compat)
            do_h_flip, do_v_flip: bool
        """
        H_in, W_in = images[0].shape[:2]
        T = len(images)
        H_out, W_out = self.target_h, self.target_w

        # ================================================================
        # 1. Spatial parameters
        # ================================================================
        if no_spatial:
            # DynPose: native resolution, no spatial transform
            pad_x0 = pad_x1 = pad_y0 = pad_y1 = 0
            sx_list = [1.0] * T
            sy_list = [1.0] * T
            left_list = [max(0, (W_in - W_out) // 2)] * T
            top_list  = [max(0, (H_in - H_out) // 2)] * T
        else:
            # --- Random padding (same for all frames, MV-TAP style) ---
            if self.random_crop:
                pad_x0, pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1], 2).tolist()
                pad_y0, pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1], 2).tolist()
            else:
                pad_x0 = pad_x1 = pad_y0 = pad_y1 = 0
            H_pad = H_in + pad_y0 + pad_y1
            W_pad = W_in + pad_x0 + pad_x1

            # --- Random base scale ---
            min_scale = max(W_out / W_pad, H_out / H_pad)
            if self.random_crop:
                base_scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
                base_scale = max(base_scale, min_scale)
            else:
                base_scale = min_scale

            # --- Uniform scale for all frames (no per-frame drift) ---
            W_int = int(max(W_pad * base_scale, W_out + 10))
            H_int = int(max(H_pad * base_scale, H_out + 10))
            sx_val = W_int / W_pad
            sy_val = H_int / H_pad
            sx_list = [sx_val] * T
            sy_list = [sy_val] * T
            W_scaled_list = [W_int] * T
            H_scaled_list = [H_int] * T

            # --- Base crop position (random, anchored to frame 0 scale) ---
            W_scaled_0 = W_scaled_list[0]
            H_scaled_0 = H_scaled_list[0]
            if self.random_crop:
                x0_base = random.randint(0, max(0, W_scaled_0 - W_out))
                y0_base = random.randint(0, max(0, H_scaled_0 - H_out))
            else:
                x0_base = (W_scaled_0 - W_out) // 2
                y0_base = (H_scaled_0 - H_out) // 2

            # --- Trajectory-centered crop (MV-TAP style): shift base crop toward
            #     mean of visible frame-0 tracks in scaled image space ---
            if (traj_2d_frame0 is not None and len(traj_2d_frame0) > 0
                    and self.random_crop):
                trajs_s = traj_2d_frame0.astype(np.float32).copy()
                # Apply padding offset then scale to get coords in scaled image
                trajs_s[:, 0] = (trajs_s[:, 0] + pad_x0) * sx_list[0]
                trajs_s[:, 1] = (trajs_s[:, 1] + pad_y0) * sy_list[0]
                in_bounds = (
                    (trajs_s[:, 0] >= 0) & (trajs_s[:, 0] < W_scaled_0) &
                    (trajs_s[:, 1] >= 0) & (trajs_s[:, 1] < H_scaled_0)
                )
                if in_bounds.sum() > 0:
                    mid_x = float(np.mean(trajs_s[in_bounds, 0]))
                    mid_y = float(np.mean(trajs_s[in_bounds, 1]))
                    x0_base = int(np.clip(
                        int(mid_x) - W_out // 2, 0, max(0, W_scaled_0 - W_out)
                    ))
                    y0_base = int(np.clip(
                        int(mid_y) - H_out // 2, 0, max(0, H_scaled_0 - H_out)
                    ))

            # --- Same crop for all frames (no per-frame drift) ---
            left_list = [x0_base] * T
            top_list  = [y0_base] * T

        # --- Flip decisions (same for all frames) ---
        do_h_flip = self.photometric_aug and random.random() < self.h_flip_prob
        do_v_flip = self.photometric_aug and random.random() < self.v_flip_prob

        # ================================================================
        # 2. Photometric parameters (per-sample decisions, decided once)
        # ================================================================
        do_color = do_blur = do_gray = False
        jitter_params = None
        if self.photometric_aug:
            do_gray  = random.random() < self.grayscale_prob
            do_color = random.random() < self.color_aug_prob
            do_blur  = random.random() < self.blur_aug_prob
            if do_color:
                jitter_params = self._jitter.get_params(
                    self._jitter.brightness, self._jitter.contrast,
                    self._jitter.saturation, self._jitter.hue
                )

        # ================================================================
        # 3. Photometric augmentation on FULL-RES images
        #    gray/color/blur/noise
        # ================================================================
        images_photo = [img.copy() for img in images] if self.photometric_aug else images

        if self.photometric_aug:
            # Gray/color/blur/noise — all frames (per-sample except noise)
            for t in range(T):
                img_pil = Image.fromarray(images_photo[t])
                if do_gray:
                    img_pil = img_pil.convert('L').convert('RGB')
                if do_color:
                    fn_idx, bf, cf, sf, hf = jitter_params
                    for fn_id in fn_idx:
                        if fn_id == 0 and bf is not None:
                            img_pil = TF.adjust_brightness(img_pil, bf)
                        elif fn_id == 1 and cf is not None:
                            img_pil = TF.adjust_contrast(img_pil, cf)
                        elif fn_id == 2 and sf is not None:
                            img_pil = TF.adjust_saturation(img_pil, sf)
                        elif fn_id == 3 and hf is not None:
                            img_pil = TF.adjust_hue(img_pil, hf)
                if do_blur:
                    img_pil = self._blur(img_pil)
                img_arr = np.array(img_pil)
                # Per-frame independent noise (MV-TAP: 15% per frame)
                if random.random() < self.noise_prob:
                    noise_sigma = random.uniform(*self.noise_sigma_range)
                    noise = np.random.normal(0, noise_sigma, img_arr.shape).astype(np.int16)
                    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                images_photo[t] = img_arr

        # ================================================================
        # 4. Per-frame spatial: pad → resize → crop → flip
        #    Input: photometric-augmented full-res images (images_photo)
        # ================================================================
        images_out = []
        depths_out    = np.zeros((T, H_out, W_out), dtype=np.float32)
        intrinsics_out = np.zeros((T, 3, 3), dtype=np.float32)

        for t in range(T):
            sx_t   = sx_list[t]
            sy_t   = sy_list[t]
            left_t = left_list[t]
            top_t  = top_list[t]

            # --- Image: pad → resize → crop ---
            img = images_photo[t]
            if not no_spatial and (pad_x0 or pad_x1 or pad_y0 or pad_y1):
                img = np.pad(img, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode='constant')
            if not no_spatial:
                W_scaled_t = W_scaled_list[t]
                H_scaled_t = H_scaled_list[t]
                interp = cv2.INTER_LINEAR if W_scaled_t > img.shape[1] else cv2.INTER_AREA
                img = cv2.resize(img, (W_scaled_t, H_scaled_t), interpolation=interp)
            img = img[top_t:top_t + H_out, left_t:left_t + W_out]
            if do_h_flip:
                img = img[:, ::-1, :].copy()
            if do_v_flip:
                img = img[::-1, :, :].copy()
            images_out.append(img)

            # --- Depth: pad → resize → crop → flip ---
            d = depths[t]
            if not no_spatial and (pad_x0 or pad_x1 or pad_y0 or pad_y1):
                d = np.pad(d, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode='constant')
            if not no_spatial:
                d = cv2.resize(d, (W_scaled_t, H_scaled_t), interpolation=cv2.INTER_NEAREST)
            d = d[top_t:top_t + H_out, left_t:left_t + W_out]
            if do_h_flip:
                d = d[:, ::-1].copy()
            if do_v_flip:
                d = d[::-1, :].copy()
            depths_out[t] = d

            # --- Intrinsics: padding shift → scale → crop shift → flip ---
            K = intrinsics[t].copy()
            if not no_spatial:
                K[0, 2] += pad_x0          # principal point shifts with padding
                K[1, 2] += pad_y0
                K[0, 0] *= sx_t
                K[1, 1] *= sy_t
                K[0, 2] = K[0, 2] * sx_t - left_t
                K[1, 2] = K[1, 2] * sy_t - top_t
            if do_h_flip:
                K[0, 2] = W_out - 1 - K[0, 2]
            if do_v_flip:
                K[1, 2] = H_out - 1 - K[1, 2]
            intrinsics_out[t] = K

        # ================================================================
        # 5. Depth augmentation (simulate estimated depth inaccuracy)
        #    Applied after spatial transforms, before return.
        #    Only modifies depths_out — Pj(tj) in train.py is recomputed
        #    from augmented depth on-the-fly, so no extra adjustment needed.
        # ================================================================
        if self.depth_aug:
            depths_out = self._augment_depth(depths_out)

        # ================================================================
        # 6. Return crop_info
        # ================================================================
        crop_info = {
            # Per-frame spatial (for fg_mask and _make_dense_grid_direct)
            'sx_list':   sx_list,
            'sy_list':   sy_list,
            'left_list': left_list,
            'top_list':  top_list,
            # Padding (same for all frames)
            'pad_x0': pad_x0, 'pad_x1': pad_x1,
            'pad_y0': pad_y0, 'pad_y1': pad_y1,
            # Frame-0 values for backward compat
            'scale': sx_list[0],
            'left':  left_list[0],
            'top':   top_list[0],
            # Flip
            'do_h_flip': do_h_flip,
            'do_v_flip': do_v_flip,
        }
        return images_out, depths_out, intrinsics_out, crop_info

    # ------------------------------------------------------------------
    # Depth augmentation methods
    # ------------------------------------------------------------------

    def _augment_depth(self, depths):
        """Apply depth augmentation pipeline: scale+shift → blur → noise.

        Args:
            depths: np.ndarray (T, H, W) float32, already spatially augmented.

        Returns:
            np.ndarray (T, H, W) float32, depth-augmented.
        """
        # 1) Grid-based local scale + shift + Gaussian blur (DeltaV2 aug_depth style)
        depths = self._depth_scale_shift(depths)
        # 2) Gaussian blur on depth (TAPiP3D blur_depth style)
        depths = self._depth_blur(depths)
        # 3) Multi-resolution noise (TAPiP3D depth_noise style)
        depths = self._depth_noise(depths)
        return depths

    def _depth_scale_shift(self, depths):
        """Grid-based local scale + shift + Gaussian blur.

        Reference: DeltaV2 aug_depth (utils.py:151-194),
                   TAPiP3D delta_depth_aug (data_ops.py:710-795).
        """
        if random.random() > 0.5:
            return depths

        T, H, W = depths.shape
        mask = depths > 0
        if not mask.any():
            return depths

        grid_h, grid_w = 8, 8
        scale_lo, scale_hi = 0.85, 1.15
        shift_lo, shift_hi = -0.05, 0.05
        gn_kernel, gn_sigma = (7, 7), (2.0, 2.0)

        # Per-frame random scale/shift at low resolution, bilinear upsample
        scale_maps = np.stack([
            cv2.resize(
                np.random.uniform(scale_lo, scale_hi, (grid_h, grid_w)).astype(np.float32),
                (W, H), interpolation=cv2.INTER_LINEAR)
            for _ in range(T)])
        shift_maps = np.stack([
            cv2.resize(
                np.random.uniform(shift_lo, shift_hi, (grid_h, grid_w)).astype(np.float32),
                (W, H), interpolation=cv2.INTER_LINEAR)
            for _ in range(T)])

        shift_scale = depths[mask].mean()
        depths = depths.copy()
        depths[mask] = depths[mask] * scale_maps[mask] + shift_maps[mask] * shift_scale

        # Gaussian blur to smooth boundaries
        depths_t = torch.from_numpy(depths)
        depths_t = TF.gaussian_blur(depths_t, kernel_size=list(gn_kernel), sigma=list(gn_sigma))
        depths = depths_t.numpy()

        depths[~mask] = 0.0
        depths = np.clip(depths, 0, None)
        return depths

    def _depth_blur(self, depths):
        """Gaussian blur on depth maps.

        Reference: TAPiP3D blur_depth (data_ops.py:241-278).
        """
        if random.random() > 0.5:
            return depths

        T, H, W = depths.shape
        mask = depths > 0
        if not mask.any():
            return depths

        # Random odd kernel size in [7, 15]
        k = random.randrange(7, 16, 2)  # 7, 9, 11, 13, 15
        kernel_size = (k, k)

        depths_t = torch.from_numpy(depths.copy())
        depths_t = TF.gaussian_blur(depths_t, kernel_size=list(kernel_size))
        depths = depths_t.numpy()

        depths[~mask] = 0.0
        return depths

    def _depth_noise(self, depths):
        """Multi-resolution noise added to depth.

        Reference: TAPiP3D depth_noise + multi_res_noise_like_np
                   (data_ops.py:607-708).
        """
        if random.random() > 0.5:
            return depths

        T, H, W = depths.shape
        mask = depths > 0
        if not mask.any():
            return depths

        # Compute per-frame std of valid depth
        _depths = depths.copy()
        _depths[~mask] = np.nan
        with np.errstate(invalid='ignore'):
            frame_std = np.nanstd(_depths.reshape(T, -1), axis=-1)  # (T,)

        strength = np.random.uniform(0.02, 0.1)
        std = strength * frame_std  # (T,)

        # Multi-resolution noise generation
        noise = np.random.standard_normal((H, W, T)).astype(np.float32)
        w_cur, h_cur = W, H
        power = np.random.uniform(0.5, 2.0)
        for i in range(10):
            r = np.random.random() * 2 + 2
            w_cur = max(1, int(w_cur / (r ** i)))
            h_cur = max(1, int(h_cur / (r ** i)))
            new_noise = np.random.standard_normal((h_cur, w_cur, T)).astype(np.float32)
            resized = cv2.resize(new_noise, (W, H), interpolation=cv2.INTER_LINEAR)
            noise += resized * (power ** i)
            if w_cur == 1 or h_cur == 1:
                break

        noise_std = noise.std()
        if noise_std > 0:
            noise /= noise_std
        noise = noise.transpose(2, 0, 1)  # (T, H, W)

        depths = depths.copy()
        scaled_noise = std[:, None, None] * noise  # (T, H, W)
        depths[mask] = depths[mask] + scaled_noise[mask]
        depths[~mask] = 0.0
        depths = np.clip(depths, 0, None)
        return depths

    # ------------------------------------------------------------------
    # Eraser augmentation (static helper for projection)
    # ------------------------------------------------------------------

    @staticmethod
    def _eraser_project_to_2d(traj3d_t, w2c_t, K_t):
        """Project dense traj3d at frame t to 2D pixel coords at frame t.

        Args:
            traj3d_t: (H, W, 3) world coords at frame t
            w2c_t: (4, 4) world-to-camera for frame t
            K_t: (3, 3) intrinsics at frame t (augmented)

        Returns:
            proj_x: (H, W) x pixel coords at frame t
            proj_y: (H, W) y pixel coords at frame t
        """
        H, W, _ = traj3d_t.shape
        pts = traj3d_t.reshape(-1, 3)  # (H*W, 3)
        # World → camera
        pts_cam = pts @ w2c_t[:3, :3].T + w2c_t[:3, 3]  # (H*W, 3)
        # Camera → pixel
        uv_h = (K_t @ pts_cam.T).T  # (H*W, 3)
        z = uv_h[:, 2]
        uv = uv_h[:, :2] / (z[:, None] + 1e-8)  # (H*W, 2)
        proj_x = uv[:, 0].reshape(H, W)
        proj_y = uv[:, 1].reshape(H, W)
        return proj_x, proj_y


# ---------------------------------------------------------------------------
# Main Dataset Class
# ---------------------------------------------------------------------------

class SyntheticDataset(torch.utils.data.Dataset):
    """Loads Kubric, PointOdyssey, DynamicReplica synthetic data.

    Returns the same dict format as UnifiedDataset raw mode:
        video, prompt, depth, intrinsic, extrinsic, traj3d, vis, conf

    traj3d is always (T, H*W, 3) — dense H×W grid format:
        - Kubric: dense grid from 512×512 track grid (nearest-neighbor mapped)
        - PO/DR: sparse tracks scattered onto H×W grid, vis=0 for empty pixels
    This ensures uniform N = H*W across all samples for easy batch collation.
    """

    def __init__(
        self,
        dataset_configs,
        height=480,
        width=832,
        num_frames=81,
        max_points=16384,  # kept for backward compat; not used (dense H×W grid)
        use_augmentation=True,
        depth_augmentation=False,
        eraser_augmentation=False,
        repeat=1,
        kubric_fix=False,
    ):
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.max_points = max_points
        self.repeat = repeat
        self.eraser_augmentation = eraser_augmentation
        self.kubric_fix = kubric_fix
        self.load_from_cache = False  # for collate_fn selection

        # Augmentation
        if use_augmentation:
            self.aug = SyntheticAugmentation(
                height, width, random_crop=True, photometric_aug=True,
                depth_aug=depth_augmentation,
            )
        else:
            self.aug = SyntheticAugmentation(
                height, width, random_crop=False, photometric_aug=False,
                aug_crop_pixels=0,
            )

        # Build clip index
        self.clips = []  # list of (type, scene_info, frame_indices)
        self._clips_by_type = {}  # per-dataset clip pools for balanced sampling
        for cfg in dataset_configs:
            dtype = cfg['type']
            start = len(self.clips)
            if dtype == 'kubric':
                self._index_kubric(cfg)
            elif dtype == 'pointodyssey':
                self._index_pointodyssey(cfg)
            elif dtype == 'dynamic_replica':
                self._index_dynamic_replica(cfg)
            elif dtype == 'dynpose':
                self._index_dynpose(cfg)
            elif dtype == 'tartanair':
                self._index_tartanair(cfg)
            else:
                raise ValueError(f"Unknown dataset type: {dtype}")

            added = self.clips[start:]
            if dtype not in self._clips_by_type:
                self._clips_by_type[dtype] = []
            self._clips_by_type[dtype].extend(added)
            print(f"[SyntheticDataset] {dtype}: {len(added)} clips")

        # Proportional sampling: each dataset is sampled proportionally to its clip count.
        self._dataset_types = list(self._clips_by_type.keys())
        total_clips = len(self.clips)

        print(f"[SyntheticDataset] Total clips: {total_clips}")
        print(f"[SyntheticDataset] Proportional sampling across {len(self._dataset_types)} types: "
              + ", ".join(f"{dt}({len(self._clips_by_type[dt])}, "
                          f"p={len(self._clips_by_type[dt])/total_clips:.2f})"
                          for dt in self._dataset_types))

    # -----------------------------------------------------------------------
    # Clip indexing
    # -----------------------------------------------------------------------

    def _index_kubric(self, cfg):
        S = cfg.get('S', 16)  # St4RTrack default: 16
        strides = cfg.get('strides', [1])
        clip_step = cfg.get('clip_step', 1)
        num_samples = cfg.get('num_samples', 0)  # 0 = use all
        # Kubric tracks are extracted from frame-0 pixel grid (kubric_point_extractor.py --grid_frames 0).
        # Starting from later frames causes sparse coverage since many frame-0 tracks are
        # occluded/out-of-view. Must keep start_from_zero=True for Kubric.
        start_from_zero = cfg.get('start_from_zero', True)

        # Support root_path: auto-discovers {root}/{batch}/frames + {batch}/tracks_dense pairs.
        # Useful when rendering is split into multiple batch subfolders.
        if 'root_path' in cfg:
            root = cfg['root_path']
            raw_track_pairs = []
            for sub in sorted(os.listdir(root)):
                sub_full = os.path.join(root, sub)
                if not os.path.isdir(sub_full):
                    continue
                frames_dir = os.path.join(sub_full, 'frames')
                tracks_dir = os.path.join(sub_full, 'tracks_dense')
                if os.path.isdir(frames_dir) and os.path.isdir(tracks_dir):
                    raw_track_pairs.append((frames_dir, tracks_dir))
        else:
            raw_track_pairs = [(cfg['raw_path'], cfg['track_path'])]

        count = 0
        total_scenes = 0
        for raw_path, track_path in raw_track_pairs:
            scenes = sorted([d for d in os.listdir(raw_path)
                             if os.path.isdir(os.path.join(raw_path, d))])
            total_scenes += len(scenes)
            for scene_id in scenes:
                raw_dir = os.path.join(raw_path, scene_id, 'view_0001')
                track_file = os.path.join(track_path, scene_id, 'view_0001', 'tracks.npz')
                if not os.path.isdir(raw_dir) or not os.path.isfile(track_file):
                    continue
                # Count frames
                n_frames = len(glob.glob(os.path.join(raw_dir, 'rgba_*.png')))
                if n_frames < 2:
                    continue
                for stride in strides:
                    max_start = n_frames - S * stride
                    if max_start < 0:
                        continue
                    if start_from_zero:
                        # Only one clip per scene per stride, always starting at frame 0
                        starts = [0]
                    else:
                        starts = range(0, max_start + 1, clip_step)
                    for start in starts:
                        frame_idx = start + np.arange(S) * stride
                        if frame_idx[-1] >= n_frames:
                            break
                        self.clips.append(('kubric', {
                            'raw_dir': raw_dir,
                            'track_file': track_file,
                        }, frame_idx))
                        count += 1

        # Subsample if requested
        if num_samples > 0 and count > num_samples:
            kubric_clips = [c for c in self.clips if c[0] == 'kubric']
            other_clips = [c for c in self.clips if c[0] != 'kubric']
            random.shuffle(kubric_clips)
            self.clips = other_clips + kubric_clips[:num_samples]
            count = num_samples

        print(f"  [Kubric] {count} clips from {total_scenes} scenes (start_from_zero={start_from_zero})")

    def _index_pointodyssey(self, cfg):
        path = cfg['path']  # .../point_odyssey/train
        S = cfg.get('S', 16)  # St4RTrack default: 16
        strides = cfg.get('strides', [2, 3, 4])
        clip_step = cfg.get('clip_step', 32)
        N_min = cfg.get('N_min', 16)
        num_samples = cfg.get('num_samples', 0)
        skip_prefixes = cfg.get('skip_prefixes', ['ani', 'char', 'r'])
        start_from_zero = cfg.get('start_from_zero', False)

        fog_list = set()
        fog_list_path = cfg.get('fog_list_path',
                                os.path.join(os.path.dirname(path), 'po_fog_list.txt'))
        if os.path.isfile(fog_list_path):
            with open(fog_list_path) as f:
                fog_list = {line.strip() for line in f}

        sequences = sorted(glob.glob(os.path.join(path, '*/')))
        count = 0
        for seq_dir in sequences:
            seq_name = os.path.basename(seq_dir.rstrip('/'))
            # Skip certain prefixes (following St4RTrack)
            if any(seq_name.startswith(p) for p in skip_prefixes):
                continue
            if seq_name in fog_list:
                continue

            anno_path = os.path.join(seq_dir, 'anno.npz')
            if not os.path.isfile(anno_path):
                continue

            # Check frame count from rgb directory
            rgb_dir = os.path.join(seq_dir, 'rgbs')
            n_frames = len(glob.glob(os.path.join(rgb_dir, 'rgb_*.jpg')))

            # Quick check track count without loading full array
            try:
                with np.load(anno_path, allow_pickle=True) as a:
                    n_points = a['trajs_3d'].shape[1]
                    if n_points < N_min:
                        continue
            except Exception:
                continue

            for stride in strides:
                max_start = n_frames - S * stride
                if max_start < 0:
                    continue
                if start_from_zero:
                    starts = [0]
                else:
                    starts = range(0, max_start + 1, clip_step)
                for start in starts:
                    frame_idx = start + np.arange(S) * stride
                    if frame_idx[-1] >= n_frames:
                        break
                    self.clips.append(('pointodyssey', {
                        'seq_dir': seq_dir,
                        'anno_path': anno_path,
                    }, frame_idx))
                    count += 1

        if num_samples > 0 and count > num_samples:
            po_clips = [c for c in self.clips if c[0] == 'pointodyssey']
            other_clips = [c for c in self.clips if c[0] != 'pointodyssey']
            random.shuffle(po_clips)
            self.clips = other_clips + po_clips[:num_samples]
            count = num_samples

        print(f"  [PointOdyssey] {count} clips from {len(sequences)} sequences (start_from_zero={start_from_zero})")

    def _index_dynamic_replica(self, cfg):
        path = cfg['path']  # .../dynamic_replica/
        S = cfg.get('S', 6)   # St4RTrack default: 6
        strides = cfg.get('strides', [4, 5, 6])
        clip_step = cfg.get('clip_step', 32)  # St4RTrack train script: 32
        num_samples = cfg.get('num_samples', 0)
        start_from_zero = cfg.get('start_from_zero', False)

        # Find annotation file
        anno_file = None
        for candidate in ['frame_annotations_train_full.json',
                          'frame_annotations_train.json']:
            p = os.path.join(path, candidate)
            if os.path.isfile(p):
                anno_file = p
                break
        if anno_file is None:
            print(f"  [DynamicReplica] WARNING: No annotation file found in {path}")
            return

        with open(anno_file) as f:
            all_anno = json.load(f)

        # Group by sequence
        anno_by_seq = {}
        for a in all_anno:
            seq = a['sequence_name']
            if seq not in anno_by_seq:
                anno_by_seq[seq] = []
            anno_by_seq[seq].append(a)

        count = 0
        for seq_name, seq_anno in anno_by_seq.items():
            n_frames = len(seq_anno)
            for stride in strides:
                max_start = n_frames - S * stride
                if max_start < 0:
                    continue
                if start_from_zero:
                    starts = [0]
                else:
                    starts = range(0, max_start + 1, clip_step)
                for start in starts:
                    frame_idx = start + np.arange(S) * stride
                    if frame_idx[-1] >= n_frames:
                        break
                    # Verify all paths exist
                    try:
                        annos = [seq_anno[i] for i in frame_idx]
                        all_exist = all(
                            os.path.isfile(os.path.join(path, a['image']['path']))
                            and os.path.isfile(os.path.join(path, a['depth']['path']))
                            for a in annos
                        )
                        if not all_exist:
                            continue
                    except (KeyError, IndexError):
                        continue

                    self.clips.append(('dynamic_replica', {
                        'dataset_root': path,
                        'annotations': annos,
                    }, frame_idx))
                    count += 1

        if num_samples > 0 and count > num_samples:
            dr_clips = [c for c in self.clips if c[0] == 'dynamic_replica']
            other_clips = [c for c in self.clips if c[0] != 'dynamic_replica']
            random.shuffle(dr_clips)
            self.clips = other_clips + dr_clips[:num_samples]
            count = num_samples

        print(f"  [DynamicReplica] {count} clips from {len(anno_by_seq)} sequences (start_from_zero={start_from_zero})")

    def _index_dynpose(self, cfg):
        """Index real-world DynPose data from a CSV file.

        Config keys:
            csv_path: path to train.csv (columns: video,prompt,depth,intrinsic,extrinsic,track)
            video_base_path: directory containing .mp4 files
            S: number of frames per clip
            strides: list of temporal strides
            num_samples: 0 = use all (default)
        """
        csv_path = cfg['csv_path']
        video_base_path = cfg['video_base_path']
        S = cfg.get('S', 8)
        strides = cfg.get('strides', [1])
        num_samples = cfg.get('num_samples', 0)
        n_total_frames = cfg.get('n_total_frames', 81)

        # Read CSV
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        count = 0
        skipped = 0
        for row in rows:
            video_path = os.path.join(video_base_path, row['video'])
            depth_path = row['depth']
            intrinsic_path = row['intrinsic']
            extrinsic_path = row['extrinsic']
            track_path = row['track']

            # Quick existence check (video + track are the essentials)
            if not os.path.isfile(video_path) or not os.path.isfile(track_path):
                skipped += 1
                continue

            scene_info = {
                'video_path': video_path,
                'depth_path': depth_path,
                'intrinsic_path': intrinsic_path,
                'extrinsic_path': extrinsic_path,
                'track_path': track_path,
            }

            for stride in strides:
                if (S - 1) * stride >= n_total_frames:
                    continue
                frame_idx = np.arange(S) * stride
                self.clips.append(('dynpose', scene_info, frame_idx))
                count += 1

        if num_samples > 0 and count > num_samples:
            dp_clips = [c for c in self.clips if c[0] == 'dynpose']
            other_clips = [c for c in self.clips if c[0] != 'dynpose']
            random.shuffle(dp_clips)
            self.clips = other_clips + dp_clips[:num_samples]
            count = num_samples

        print(f"  [DynPose] {count} clips from {len(rows)} videos "
              f"(skipped {skipped}, strides={strides})")

    def _index_tartanair(self, cfg):
        """Index TartanAir v1 trajectories.

        Config keys:
            path: root directory containing env/Hard/env/Hard/P000/... structure
            S: number of frames per clip
            strides: list of temporal strides
            clip_step: step between clip start positions (default 64)
            num_samples: 0 = use all (default)
            start_from_zero: if True, only one clip per trajectory starting at 0
            max_depth: clamp sky depth (default 80.0)
        """
        path = cfg['path']
        S = cfg.get('S', 16)
        strides = cfg.get('strides', [1, 2, 4, 8])
        clip_step = cfg.get('clip_step', 64)
        num_samples = cfg.get('num_samples', 0)
        start_from_zero = cfg.get('start_from_zero', False)

        # Discover all trajectories: path/{env}/Hard/{env}/Hard/P{nnn}/
        # (HuggingFace download nests env name inside zip)
        traj_dirs = sorted(glob.glob(os.path.join(path, '*/Hard/*/Hard/P*')))
        if not traj_dirs:
            # Fallback: standard structure path/{env}/Hard/P{nnn}/
            traj_dirs = sorted(glob.glob(os.path.join(path, '*/Hard/P*')))

        count = 0
        for traj_dir in traj_dirs:
            pose_file = os.path.join(traj_dir, 'pose_left.txt')
            img_dir = os.path.join(traj_dir, 'image_left')
            depth_dir = os.path.join(traj_dir, 'depth_left')
            if not os.path.isfile(pose_file) or not os.path.isdir(img_dir):
                continue

            n_frames = len(glob.glob(os.path.join(img_dir, '*.png')))
            if n_frames < S:
                continue

            scene_info = {
                'traj_dir': traj_dir,
                'pose_file': pose_file,
                'img_dir': img_dir,
                'depth_dir': depth_dir,
                'max_depth': cfg.get('max_depth', 80.0),
            }

            for stride in strides:
                max_start = n_frames - S * stride
                if max_start < 0:
                    continue
                if start_from_zero:
                    starts = [0]
                else:
                    starts = range(0, max_start + 1, clip_step)
                for start in starts:
                    frame_idx = start + np.arange(S) * stride
                    if frame_idx[-1] >= n_frames:
                        break
                    self.clips.append(('tartanair', scene_info, frame_idx))
                    count += 1

        if num_samples > 0 and count > num_samples:
            ta_clips = [c for c in self.clips if c[0] == 'tartanair']
            other_clips = [c for c in self.clips if c[0] != 'tartanair']
            random.shuffle(ta_clips)
            self.clips = other_clips + ta_clips[:num_samples]
            count = num_samples

        print(f"  [TartanAir] {count} clips from {len(traj_dirs)} trajectories "
              f"(start_from_zero={start_from_zero})")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_camera_stats_np(extrinsics):
        """Compute camera normalization stats from extrinsics (numpy).

        Same logic as train.py _compute_camera_stats but in numpy.
        For DynPose, call with the full 81-frame extrinsic to match
        the old pipeline behavior.

        Returns:
            dict with 'mean': (1, 3) np.array, 'max': float
        """
        c2w = extrinsics.astype(np.float32)
        c2w_0_inv = np.linalg.inv(c2w[0])
        c2w_norm = c2w_0_inv[None] @ c2w  # (T, 4, 4)
        centers = c2w_norm[:, :3, 3]       # (T, 3)
        mean_center = centers.mean(axis=0, keepdims=True)  # (1, 3)
        max_dist = float(np.max(np.linalg.norm(centers - mean_center, axis=-1)))
        return {
            'mean': mean_center,   # (1, 3) np.array
            'max': max_dist,       # float
        }

    # -----------------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------------

    def __len__(self):
        return len(self.clips) * self.repeat

    def __getitem__(self, idx):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                clip_idx = (idx + attempt) % len(self.clips)
                clip = self.clips[clip_idx]
                return self._load_item_from_clip(clip)
            except Exception as e:
                if attempt == 0:
                    print(f"[SyntheticDataset] Failed idx={idx}: {e}")
        raise RuntimeError(f"Failed after {max_retries} retries for idx={idx}")

    def _apply_eraser(self, images, depths, traj3d, vis, intrinsics, extrinsics):
        """Eraser augmentation following TAPiP3D (data_ops.py:54-123).

        For frames t=1..T-1, randomly erase rectangles in RGB and depth,
        then project the dense traj3d grid to frame-t pixel space to find
        which frame-0 grid cells have their tracked point inside the erased
        region, and mark those as invisible at frame t.

        Args:
            images:     list of T np.ndarray (H, W, 3) uint8 — augmented
            depths:     np.ndarray (T, H, W) float32 — augmented
            traj3d:     np.ndarray (T, H, W, 3) world coords — dense grid
            vis:        np.ndarray (T, H, W) — per-frame visibility
            intrinsics: np.ndarray (T, 3, 3) — augmented (scale+crop+flip)
            extrinsics: np.ndarray (T, 4, 4) c2w — flip-modified

        Returns:
            images, depths, vis (copies, modified)
        """
        T, H, W = depths.shape

        prob = 0.5
        bounds = [20, 300]
        max_n = 3

        images = [img.copy() for img in images]
        depths = depths.copy()
        vis = vis.copy()

        for t in range(1, T):
            if random.random() >= prob:
                continue

            # Project traj3d[t] to frame-t 2D pixel space
            w2c_t = np.linalg.inv(extrinsics[t])
            proj_x, proj_y = SyntheticAugmentation._eraser_project_to_2d(
                traj3d[t], w2c_t, intrinsics[t]
            )

            n_rects = random.randint(1, max_n)
            for _ in range(n_rects):
                xc = random.randint(0, W - 1)
                yc = random.randint(0, H - 1)
                dx = random.randint(bounds[0], bounds[1])
                dy = random.randint(bounds[0], bounds[1])
                x0 = int(np.clip(xc - dx / 2, 0, W - 1).round())
                x1 = int(np.clip(xc + dx / 2, 0, W - 1).round())
                y0 = int(np.clip(yc - dy / 2, 0, H - 1).round())
                y1 = int(np.clip(yc + dy / 2, 0, H - 1).round())

                if x0 >= x1 or y0 >= y1:
                    continue

                # Erase RGB with mean color of the region
                region = images[t][y0:y1, x0:x1, :]
                if region.size > 0:
                    mean_color = region.astype(np.float32).reshape(-1, 3).mean(axis=0)
                    images[t][y0:y1, x0:x1, :] = mean_color.astype(np.uint8)

                # Erase depth with mean valid depth of the region
                depth_region = depths[t, y0:y1, x0:x1]
                if depth_region.size > 0:
                    valid_d = depth_region[depth_region > 0]
                    mean_d = valid_d.mean() if len(valid_d) > 0 else 0.0
                    depths[t, y0:y1, x0:x1] = mean_d

                # Mark vis=0 for grid cells whose frame-t projection falls in erased rect
                occ_mask = (
                    (proj_x >= x0) & (proj_x < x1) &
                    (proj_y >= y0) & (proj_y < y1)
                )
                vis[t][occ_mask] = 0

        return images, depths, vis

    def _apply_replace(self, images, depths, traj3d, vis, intrinsics, extrinsics):
        """Replace augmentation following TAPiP3D (data_ops.py:125-188) / DeltaV2.

        Similar to eraser, but instead of filling with mean color, pastes a
        photometric-augmented patch from a random frame. Depth is scaled to
        match the target region (simulating a foreground occluder).

        Should be called AFTER _apply_eraser (DeltaV2 order).

        Args / Returns: same as _apply_eraser.
        """
        T, H, W = depths.shape

        prob = 0.5
        bounds = [20, 300]
        max_n = 3

        images = [img.copy() for img in images]
        depths = depths.copy()
        vis = vis.copy()

        # Create photometric-augmented versions (DeltaV2: apply ColorJitter twice)
        jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        images_alt = []
        for img in images:
            pil_img = Image.fromarray(img)
            pil_img = jitter(pil_img)
            pil_img = jitter(pil_img)
            images_alt.append(np.array(pil_img))

        for t in range(1, T):
            if random.random() >= prob:
                continue

            # Project traj3d[t] to frame-t 2D pixel space
            w2c_t = np.linalg.inv(extrinsics[t])
            proj_x, proj_y = SyntheticAugmentation._eraser_project_to_2d(
                traj3d[t], w2c_t, intrinsics[t]
            )

            n_rects = random.randint(1, max_n)
            for _ in range(n_rects):
                xc = random.randint(0, W - 1)
                yc = random.randint(0, H - 1)
                dx = random.randint(bounds[0], bounds[1])
                dy = random.randint(bounds[0], bounds[1])
                x0 = int(np.clip(xc - dx / 2, 0, W - 1).round())
                x1 = int(np.clip(xc + dx / 2, 0, W - 1).round())
                y0 = int(np.clip(yc - dy / 2, 0, H - 1).round())
                y1 = int(np.clip(yc + dy / 2, 0, H - 1).round())

                if x0 >= x1 or y0 >= y1:
                    continue

                wid = x1 - x0
                hei = y1 - y0
                if hei >= H or wid >= W:
                    continue

                # Random source frame and crop position
                y00 = random.randint(0, H - hei - 1) if H - hei > 1 else 0
                x00 = random.randint(0, W - wid - 1) if W - wid > 1 else 0
                fr = random.randint(0, T - 1)

                # Replace RGB with photometric-augmented patch
                images[t][y0:y1, x0:x1, :] = images_alt[fr][y00:y00 + hei, x00:x00 + wid, :]

                # Replace depth: scale source depth to target region (TAPiP3D style)
                rep_depth = depths[fr, y00:y00 + hei, x00:x00 + wid].copy()
                rep_max = rep_depth.max()
                target_valid = depths[t, y0:y1, x0:x1][depths[t, y0:y1, x0:x1] > 0]
                if rep_max > 0 and len(target_valid) > 0:
                    rep_depth = rep_depth / rep_max * target_valid.min()
                    rep_depth = random.uniform(0.8, 1.0) * rep_depth
                depths[t, y0:y1, x0:x1] = rep_depth

                # Mark vis=0 for grid cells whose frame-t projection falls in rect
                occ_mask = (
                    (proj_x >= x0) & (proj_x < x1) &
                    (proj_y >= y0) & (proj_y < y1)
                )
                vis[t][occ_mask] = 0

        return images, depths, vis

    def _load_item_from_clip(self, clip):
        dtype, scene_info, frame_idx = clip

        if dtype == 'kubric':
            raw = self._load_kubric_clip(scene_info, frame_idx)
        elif dtype == 'pointodyssey':
            raw = self._load_pointodyssey_clip(scene_info, frame_idx)
        elif dtype == 'dynamic_replica':
            raw = self._load_dynamic_replica_clip(scene_info, frame_idx)
        elif dtype == 'dynpose':
            raw = self._load_dynpose_clip(scene_info, frame_idx)
        elif dtype == 'tartanair':
            raw = self._load_tartanair_clip(scene_info, frame_idx)
        else:
            raise ValueError(f"Unknown type: {dtype}")

        # Project frame-0 3D tracks to 2D for trajectory-centered crop.
        # Only for datasets with 3D tracks + camera params (not DynPose).
        traj_2d_f0 = None
        if dtype != 'dynpose' and raw.get('traj3d') is not None:
            K0 = raw['intrinsic'][0]       # (3, 3)
            c2w_0 = raw['extrinsic'][0]    # (4, 4)
            w2c_0 = np.linalg.inv(c2w_0)
            pts = raw['traj3d'][0]         # (N, 3) world coords at frame 0
            vis0 = raw['vis'][0]           # (N,) visibility at frame 0
            if len(pts) > 0:
                # Project world → camera → image
                pts_cam = pts @ w2c_0[:3, :3].T + w2c_0[:3, 3]  # (N, 3)
                uv_h = (K0 @ pts_cam.T).T                         # (N, 3)
                z = uv_h[:, 2]
                uv = uv_h[:, :2] / (z[:, None] + 1e-8)           # (N, 2)
                valid = (vis0 > 0) & (z > 0) & np.isfinite(uv).all(axis=-1)
                if valid.sum() > 0:
                    traj_2d_f0 = uv[valid]  # (M, 2) visible tracks in original image space

        # Apply augmentation (spatial → eraser/replace → photometric)
        images, depths, intrinsics, crop_info = self.aug(
            raw['images'], raw['depth'], raw['intrinsic'],
            traj_2d_frame0=traj_2d_f0,
        )

        # Modify extrinsics for flip (c2w: negate camera axis columns)
        extrinsics = raw['extrinsic'].copy()
        do_h_flip = crop_info.get('do_h_flip', False)
        do_v_flip = crop_info.get('do_v_flip', False)
        if do_h_flip:
            extrinsics[:, :3, 0] *= -1  # negate camera x-axis
        if do_v_flip:
            extrinsics[:, :3, 1] *= -1  # negate camera y-axis

        # Apply same pad+scale+crop+flip to fg_mask (always create, default all-ones)
        T = len(images)
        H, W = self.height, self.width
        fg_mask_raw = raw.get('fg_mask', None)
        if fg_mask_raw is not None:
            sx_list   = crop_info['sx_list']
            sy_list   = crop_info['sy_list']
            left_list = crop_info['left_list']
            top_list  = crop_info['top_list']
            pad_x0 = crop_info.get('pad_x0', 0)
            pad_x1 = crop_info.get('pad_x1', 0)
            pad_y0 = crop_info.get('pad_y0', 0)
            pad_y1 = crop_info.get('pad_y1', 0)
            fg_mask_aug = np.ones((T, H, W), dtype=np.float32)
            for t in range(T):
                m = fg_mask_raw[t]
                if pad_x0 or pad_x1 or pad_y0 or pad_y1:
                    m = np.pad(m, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode='constant')
                h_src, w_src = m.shape
                h_scaled = max(int(round(h_src * sy_list[t])), H)
                w_scaled = max(int(round(w_src * sx_list[t])), W)
                m = cv2.resize(m, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)
                fg_mask_aug[t] = m[top_list[t]:top_list[t] + H, left_list[t]:left_list[t] + W]
            if do_h_flip:
                fg_mask_aug = fg_mask_aug[:, :, ::-1].copy()
            if do_v_flip:
                fg_mask_aug = fg_mask_aug[:, ::-1, :].copy()
        else:
            fg_mask_aug = np.ones((T, H, W), dtype=np.float32)

        # Build dense H×W traj3d grid
        conf = None
        if dtype == 'dynpose':
            # DynPose tracks are already dense (N = H_src × W_src pixel grid).
            # Apply the same spatial transform as images instead of re-projecting.
            # camera_stats must be from UN-FLIPPED 81-frame extrinsics (same normalization
            # used when generating traj3d). Pass here so _make_dense_grid_direct can
            # apply the correct 3D coordinate transform after spatial flip.
            extr_full = np.load(scene_info['extrinsic_path']).astype(np.float32)
            dynpose_camera_stats = self._compute_camera_stats_np(extr_full)

            H_src, W_src = raw['images'][0].shape[:2]
            traj3d, vis, conf = self._make_dense_grid_direct(
                raw['traj3d'], raw['vis'], crop_info, H_src, W_src,
                conf_flat=raw.get('conf', None),
                camera_stats=dynpose_camera_stats,
            )
        else:
            dynpose_camera_stats = None
            # Sparse tracks: re-project to augmented grid
            traj3d, vis = self._make_dense_grid_sparse(
                raw['traj3d'], raw['vis'],
                intrinsics, extrinsics, crop_info
            )

        # Eraser + Replace augmentation (synthetic datasets only, not DynPose)
        # Order follows DeltaV2: eraser first, then replace.
        # Save pre-eraser vis (natural occlusion only) separately from eraser-modified vis.
        eraser_vis = None
        if self.eraser_augmentation and dtype != 'dynpose':
            vis_before_eraser = vis.copy()
            images, depths, vis = self._apply_eraser(
                images, depths, traj3d, vis, intrinsics, extrinsics
            )
            images, depths, vis = self._apply_replace(
                images, depths, traj3d, vis, intrinsics, extrinsics
            )
            # eraser_vis: per-frame mask for eraser/replace augmented regions only
            # (1 = not erased, 0 = erased at this frame)
            eraser_vis = (vis > 0.5) | (vis_before_eraser < 0.5)
            eraser_vis = eraser_vis.astype(np.float32)

        # Build output dict (pass modified extrinsics)
        result = self._to_output_dict(
            images, depths, intrinsics,
            extrinsics, traj3d, vis, fg_mask=fg_mask_aug, conf=conf
        )
        if eraser_vis is not None:
            T_ev, H_ev, W_ev = eraser_vis.shape
            result['eraser_vis'] = torch.from_numpy(
                eraser_vis.reshape(T_ev, H_ev * W_ev)
            )
        else:
            # No eraser augmentation (DynPose or eraser_augmentation=False):
            # still provide all-ones eraser_vis for consistent batch collation
            result['eraser_vis'] = torch.ones_like(result['vis'])
        result['dataset_type'] = dtype

        # Compute camera_stats for result.
        # DynPose: always UN-FLIPPED 81-frame extrinsics (traj3d was generated with these).
        #          The coordinate transform in _make_dense_grid_direct ensures that
        #          flipped traj3d projects correctly using flipped cameras + UN-FLIPPED stats.
        # Synthetic: use the available (already flip-modified) sampled frames.
        if dtype == 'dynpose':
            result['camera_stats'] = dynpose_camera_stats  # already computed, UN-FLIPPED
        else:
            result['camera_stats'] = self._compute_camera_stats_np(extrinsics)

        return result

    # -----------------------------------------------------------------------
    # Per-dataset loaders
    # -----------------------------------------------------------------------

    def _load_kubric_clip(self, scene_info, frame_idx):
        """Load a Kubric clip.

        Kubric tracks.npz contains:
            coords: (N, T, 2) — pixel coords of tracked points (N = H_img * W_img)
            abs_depth: (N, T) — euclidean depth per track per frame
            occluded: (N, T) — occlusion flag
            intrinsics: (T, 3, 3) — pixel-space
            cam_rot: (T, 3, 3) — w2c rotation
            cam_trans: (T, 3) — w2c translation

        Returns sparse traj3d (same format as PO/DR) for unified grid construction.
        Note: Kubric tracks are NOT on a regular pixel grid at any frame, so we
        treat them as sparse tracks and use _make_dense_grid_sparse.
        """
        raw_dir = scene_info['raw_dir']
        track_file = scene_info['track_file']

        T = len(frame_idx)

        # Load images
        images = []
        for fi in frame_idx:
            img_path = os.path.join(raw_dir, f'rgba_{fi:05d}.png')
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        H_img, W_img = images[0].shape[:2]

        # Load tracks
        tracks = np.load(track_file)
        intrinsics_all = tracks['intrinsics']       # (T_all, 3, 3)
        cam_rot_all = tracks['cam_rot']             # (T_all, 3, 3)
        cam_trans_all = tracks['cam_trans']          # (T_all, 3)
        coords_all = tracks['coords']               # (N, T_all, 2)
        abs_depth_all = tracks['abs_depth']          # (N, T_all)
        occluded_all = tracks['occluded']            # (N, T_all)

        N = coords_all.shape[0]

        # Slice to clip frames
        intrinsics = intrinsics_all[frame_idx].copy()  # (T, 3, 3)

        # Kubric intrinsic bug fix: tracks.npz stores fy as (focal/sensor_width)*H instead
        # of (focal/sensor_width)*W. Kubric/Blender uses square pixels (sensor_fit='AUTO')
        # so fx must equal fy. Stored coords, abs_depth, world_coords were generated with
        # the correct K internally, so overriding fy=fx here makes loader unprojection
        # match stored world_coords to machine precision. See extractor_utils.py:265.
        if self.kubric_fix:
            intrinsics[:, 1, 1] = intrinsics[:, 0, 0]
        cam_rot = cam_rot_all[frame_idx]             # (T, 3, 3)
        cam_trans = cam_trans_all[frame_idx]          # (T, 3)
        coords = coords_all[:, frame_idx, :]          # (N, T, 2)
        abs_depth = abs_depth_all[:, frame_idx]        # (N, T)
        occluded = occluded_all[:, frame_idx]          # (N, T)

        # Build c2w extrinsics
        extrinsics = np.zeros((T, 4, 4), dtype=np.float32)
        for t in range(T):
            extrinsics[t] = kubric_cam_to_c2w(cam_rot[t], cam_trans[t])

        # Unproject all tracks to 3D world coords at each frame
        traj3d = np.zeros((T, N, 3), dtype=np.float32)
        for t in range(T):
            K = intrinsics[t]
            c2w = extrinsics[t]
            u_t = coords[:, t, 0]  # (N,)
            v_t = coords[:, t, 1]  # (N,)
            euc_d = abs_depth[:, t]  # (N,)

            # Euclidean → z-depth
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x_fac = (u_t - cx) / fx
            y_fac = (v_t - cy) / fy
            z_depth = euc_d / np.sqrt(1 + x_fac ** 2 + y_fac ** 2)

            # Unproject to world coords
            X_cam = x_fac * z_depth
            Y_cam = y_fac * z_depth
            P_cam = np.stack([X_cam, Y_cam, z_depth, np.ones(N, dtype=np.float32)], axis=-1)  # (N, 4)
            P_world = (c2w @ P_cam.T).T[:, :3]  # (N, 3)
            traj3d[t] = P_world

        vis = (~occluded).astype(np.float32).T  # (N, T) → (T, N)

        # Load depth maps and undistort (for the model's depth input)
        depths = np.zeros((T, H_img, W_img), dtype=np.float32)
        for i, fi in enumerate(frame_idx):
            depth_path = os.path.join(raw_dir, f'depth_{fi:05d}.tiff')
            d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if d is None:
                raise FileNotFoundError(f"Cannot read {depth_path}")
            depths[i] = undistort_depthmap(d, intrinsics[i])

        # Load segmentation maps for foreground mask
        # Kubric segmentation: pixel value = object_id * 10
        # object_id 0 = background (ground/walls/sky)
        fg_mask = np.ones((T, H_img, W_img), dtype=np.float32)
        for i, fi in enumerate(frame_idx):
            seg_path = os.path.join(raw_dir, f'segmentation_{fi:05d}.png')
            if os.path.exists(seg_path):
                seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                if seg is not None:
                    obj_id = seg // 10  # recover object ID
                    fg_mask[i] = (obj_id > 0).astype(np.float32)

        return {
            'images': images,
            'depth': depths,
            'intrinsic': intrinsics,
            'extrinsic': extrinsics,
            'traj3d': traj3d,     # (T, N, 3) world coords
            'vis': vis,           # (T, N) visibility
            'fg_mask': fg_mask,   # (T, H, W) foreground mask (1=object, 0=background)
        }

    def _load_pointodyssey_clip(self, scene_info, frame_idx):
        """Load a Point Odyssey clip.

        anno.npz contains:
            trajs_3d: (T_all, N, 3) — world coords
            visibs: (T_all, N) — visibility
            intrinsics: (T_all, 3, 3) — pixel-space
            extrinsics: (T_all, 4, 4) — w2c
        """
        seq_dir = scene_info['seq_dir']
        anno_path = scene_info['anno_path']

        T = len(frame_idx)

        # Load images
        images = []
        for fi in frame_idx:
            img_path = os.path.join(seq_dir, 'rgbs', f'rgb_{fi:05d}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        H_img, W_img = images[0].shape[:2]

        # Load annotation
        anno = np.load(anno_path, allow_pickle=True)
        trajs_3d = anno['trajs_3d'][frame_idx]        # (T, N, 3)
        visibs = anno['visibs'][frame_idx]             # (T, N)
        intrinsics = anno['intrinsics'][frame_idx]     # (T, 3, 3)
        extr_w2c = anno['extrinsics'][frame_idx]       # (T, 4, 4) w2c

        # Convert w2c → c2w
        extrinsics = np.zeros((T, 4, 4), dtype=np.float32)
        for t in range(T):
            R = extr_w2c[t, :3, :3]
            tvec = extr_w2c[t, :3, 3]
            extrinsics[t] = w2c_to_c2w(R, tvec)

        # Load depth maps
        depths = np.zeros((T, H_img, W_img), dtype=np.float32)
        for i, fi in enumerate(frame_idx):
            depth_path = os.path.join(seq_dir, 'depths', f'depth_{fi:05d}.png')
            depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth16 is None:
                raise FileNotFoundError(f"Cannot read {depth_path}")
            depths[i] = depth16.astype(np.float32) / 65535.0 * 1000.0

        # Load foreground masks (instance segmentation → binary fg)
        fg_mask = np.ones((T, H_img, W_img), dtype=np.float32)
        for i, fi in enumerate(frame_idx):
            mask_path = os.path.join(seq_dir, 'masks', f'mask_{fi:05d}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    fg_mask[i] = (mask > 0).astype(np.float32)

        return {
            'images': images,
            'depth': depths,
            'intrinsic': intrinsics.astype(np.float32),
            'extrinsic': extrinsics,
            'traj3d': trajs_3d.astype(np.float32),
            'vis': visibs.astype(np.float32),
            'fg_mask': fg_mask,
        }

    def _load_dynamic_replica_clip(self, scene_info, frame_idx):
        """Load a Dynamic Replica clip.

        Annotations contain per-frame:
            image.path, depth.path, trajectories.path, viewpoint (R, T, focal_length, ...)
        """
        root = scene_info['dataset_root']
        annotations = scene_info['annotations']

        T = len(frame_idx)

        images = []
        depths = np.zeros((T,), dtype=object)
        intrinsics = np.zeros((T, 3, 3), dtype=np.float32)
        extrinsics = np.zeros((T, 4, 4), dtype=np.float32)
        traj3d_list = []
        vis_list = []

        for t_idx, anno in enumerate(annotations):
            # Load image
            img_path = os.path.join(root, anno['image']['path'])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            H_img, W_img = img.shape[:2]
            img_size = anno['image']['size']  # [H, W]

            # Intrinsics from viewpoint
            vp = anno['viewpoint']
            K = convert_ndc_to_pixel_intrinsics(
                vp['focal_length'], vp['principal_point'],
                img_size[1], img_size[0],  # width, height
                vp.get('intrinsics_format', 'ndc_isotropic')
            )
            intrinsics[t_idx] = K

            # Extrinsics: PyTorch3D convention → OpenCV → c2w
            # PyTorch3D: X_cam = X_world @ R_pt3d + T_pt3d (right-multiply, row vectors)
            # PyTorch3D axes: X-left, Y-up, Z-forward
            # OpenCV axes:   X-right, Y-down, Z-forward
            # opencv_from_cameras_projection does:
            #   T[:2] *= -1;  R[:, :2] *= -1;  R = R.T
            # → R_opencv = diag(-1,-1,1) @ R_pt3d^T,  T_opencv = diag(-1,-1,1) @ T_pt3d
            R_pt3d = np.array(vp['R'], dtype=np.float32)
            T_pt3d = np.array(vp['T'], dtype=np.float32)
            flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
            R_opencv = flip @ R_pt3d.T
            T_opencv = flip @ T_pt3d
            extrinsics[t_idx] = w2c_to_c2w(R_opencv, T_opencv)

            # Depth
            depth_path = os.path.join(root, anno['depth']['path'])
            d = load_16big_png_depth(depth_path)
            depths[t_idx] = d

            # Trajectories
            traj_path = anno.get('trajectories', {}).get('path')
            if traj_path:
                traj_full_path = os.path.join(root, traj_path)
                if os.path.isfile(traj_full_path):
                    traj_data = torch.load(traj_full_path, weights_only=False)
                    traj3d_list.append(traj_data['traj_3d_world'].numpy())
                    vis_list.append(traj_data['verts_inds_vis'].numpy())
                else:
                    traj3d_list.append(None)
                    vis_list.append(None)
            else:
                traj3d_list.append(None)
                vis_list.append(None)

        # Stack depths
        H_img, W_img = images[0].shape[:2]
        depth_arr = np.zeros((T, H_img, W_img), dtype=np.float32)
        for t in range(T):
            if depths[t] is not None:
                depth_arr[t] = depths[t]

        # Build traj3d: all frames must share the same tracked points
        # Use frame 0's point set, track through all frames
        if traj3d_list[0] is not None:
            # All frames should have the same N points
            N = traj3d_list[0].shape[0]
            traj3d = np.zeros((T, N, 3), dtype=np.float32)
            vis = np.zeros((T, N), dtype=np.float32)
            for t in range(T):
                if traj3d_list[t] is not None:
                    n_t = traj3d_list[t].shape[0]
                    n_use = min(N, n_t)
                    traj3d[t, :n_use] = traj3d_list[t][:n_use]
                    vis[t, :n_use] = vis_list[t][:n_use].astype(np.float32)
        else:
            # No trajectory data — create dummy from depth
            traj3d = np.zeros((T, 1, 3), dtype=np.float32)
            vis = np.zeros((T, 1), dtype=np.float32)

        return {
            'images': images,
            'depth': depth_arr,
            'intrinsic': intrinsics,
            'extrinsic': extrinsics,
            'traj3d': traj3d,
            'vis': vis,
        }

    def _load_dynpose_clip(self, scene_info, frame_idx):
        """Load a DynPose (real data) clip.

        Real data format:
            video: MP4 (81 frames, 480×832)
            depth: .npy (81, 1, H, W) float32
            intrinsic: .npy (81, 3, 3) float32
            extrinsic: .npy (81, 4, 4) float32 (c2w)
            track: .npz with traj3d (81, N, 3) float16, vis (81, N) float16
        """
        video_path = scene_info['video_path']
        depth_path = scene_info['depth_path']
        intrinsic_path = scene_info['intrinsic_path']
        extrinsic_path = scene_info['extrinsic_path']
        track_path = scene_info['track_path']

        T = len(frame_idx)

        # Load video frames at specific indices
        import imageio
        reader = imageio.get_reader(video_path)
        images = []
        for fi in frame_idx:
            try:
                frame = reader.get_data(fi)  # (H, W, 3) RGB uint8
            except Exception:
                raise RuntimeError(f"Cannot read frame {fi} from {video_path}")
            images.append(frame)
        reader.close()

        H_img, W_img = images[0].shape[:2]

        # Load and slice auxiliary data
        depth_all = np.load(depth_path)                  # (81, 1, H, W)
        depths = depth_all[frame_idx, 0].astype(np.float32)  # (T, H, W)

        intrinsics = np.load(intrinsic_path)[frame_idx].astype(np.float32)  # (T, 3, 3)
        extrinsics = np.load(extrinsic_path)[frame_idx].astype(np.float32)  # (T, 4, 4)

        # Load tracks (dense: N = H_img × W_img)
        track_data = np.load(track_path)
        traj3d = track_data['traj3d'][frame_idx].astype(np.float32)  # (T, N, 3)
        vis = track_data['vis'][frame_idx].astype(np.float32)        # (T, N)
        conf = track_data['conf'][frame_idx].astype(np.float32) if 'conf' in track_data else None  # (T, N)

        # No segmentation available for real data
        fg_mask = np.ones((T, H_img, W_img), dtype=np.float32)

        result = {
            'images': images,
            'depth': depths,
            'intrinsic': intrinsics,
            'extrinsic': extrinsics,
            'traj3d': traj3d,
            'vis': vis,
            'fg_mask': fg_mask,
        }
        if conf is not None:
            result['conf'] = conf
        return result

    def _load_tartanair_clip(self, scene_info, frame_idx):
        """Load a TartanAir v1 clip (static scene, camera-only motion).

        TartanAir data:
            image_left: 640×480 PNG (RGB)
            depth_left: 640×480 NPY float32 (z-depth in meters, sky ≈ 10000)
            pose_left.txt: per-line 'tx ty tz qx qy qz qw'
                - Position: camera optical center in world frame
                - Quaternion: camera orientation, NED body frame (x-fwd, y-right, z-down)
                - This is c2w in NED convention
            Intrinsics: fixed fx=fy=320, cx=320, cy=240

        Static scene → traj3d: unproject frame-0 depth to world coords,
        same world coords at all frames. Visibility from depth consistency check.
        """
        traj_dir = scene_info['traj_dir']
        pose_file = scene_info['pose_file']
        img_dir = scene_info['img_dir']
        depth_dir = scene_info['depth_dir']
        max_depth = scene_info.get('max_depth', 80.0)

        T = len(frame_idx)

        # Fixed intrinsics
        fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # Load all poses, slice to clip frames
        all_poses = np.loadtxt(pose_file)  # (N_total, 7)

        # Build c2w matrices (NED → OpenCV)
        extrinsics = np.zeros((T, 4, 4), dtype=np.float32)
        for i, fi in enumerate(frame_idx):
            p = all_poses[fi]
            extrinsics[i] = tartanair_pose_to_c2w(*p)

        intrinsics = np.tile(K, (T, 1, 1))  # (T, 3, 3)

        # Load images and depth maps
        images = []
        depths = np.zeros((T, 480, 640), dtype=np.float32)
        for i, fi in enumerate(frame_idx):
            img_path = os.path.join(img_dir, f'{fi:06d}_left.png')
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            depth_path = os.path.join(depth_dir, f'{fi:06d}_left_depth.npy')
            d = np.load(depth_path).astype(np.float32)
            d = np.clip(d, 0, max_depth)  # clamp sky
            depths[i] = d

        H_img, W_img = 480, 640

        # --- Build traj3d from depth (static scene) ---
        # Unproject ALL frame-0 pixels to world coords
        c2w_0 = extrinsics[0]
        uu, vv = np.meshgrid(np.arange(W_img), np.arange(H_img))
        uu_flat = uu.ravel().astype(np.float32)  # (H*W,)
        vv_flat = vv.ravel().astype(np.float32)
        z0_flat = depths[0].ravel()               # (H*W,)

        # Valid: reasonable depth (not sky, not zero)
        valid_depth = (z0_flat > 0.1) & (z0_flat < max_depth - 0.1)

        X_cam = (uu_flat - cx) / fx * z0_flat
        Y_cam = (vv_flat - cy) / fy * z0_flat
        ones = np.ones_like(z0_flat)
        P_cam = np.stack([X_cam, Y_cam, z0_flat, ones], axis=-1)  # (H*W, 4)
        P_world_all = (c2w_0 @ P_cam.T).T[:, :3]  # (H*W, 3)

        # Subsample tracks: take every Kth pixel on a regular grid for efficiency
        # Target ~4000-8000 tracks (matches PO/DR sparse track density)
        N_total = H_img * W_img  # 307200
        grid_step = max(1, int(np.sqrt(N_total / 6000)))  # ~6000 tracks
        sample_v = np.arange(0, H_img, grid_step)
        sample_u = np.arange(0, W_img, grid_step)
        sample_vv, sample_uu = np.meshgrid(sample_v, sample_u, indexing='ij')
        sample_idx = (sample_vv.ravel() * W_img + sample_uu.ravel())

        # Filter to valid depth
        sample_valid = valid_depth[sample_idx]
        sample_idx = sample_idx[sample_valid]
        N = len(sample_idx)

        if N < 10:
            raise ValueError(f"Too few valid depth pixels in {traj_dir}")

        P_world = P_world_all[sample_idx]  # (N, 3) world coords of tracked points

        # traj3d: static scene → same world coords at all frames
        traj3d = np.tile(P_world[None], (T, 1, 1))  # (T, N, 3)

        # Visibility: project world points into each frame, check depth consistency
        vis = np.zeros((T, N), dtype=np.float32)
        depth_tol = 0.05  # 5% relative tolerance

        for t in range(T):
            w2c_t = np.linalg.inv(extrinsics[t])
            ones_n = np.ones((N, 1), dtype=np.float32)
            P_homo = np.concatenate([P_world, ones_n], axis=-1)  # (N, 4)
            P_cam_t = (w2c_t @ P_homo.T).T[:, :3]  # (N, 3)

            z_t = P_cam_t[:, 2]
            u_t = fx * P_cam_t[:, 0] / (z_t + 1e-8) + cx
            v_t = fy * P_cam_t[:, 1] / (z_t + 1e-8) + cy

            # NaN/Inf can arise from points behind camera or at infinity; clamp before cast
            u_t = np.nan_to_num(u_t, nan=-1.0, posinf=-1.0, neginf=-1.0)
            v_t = np.nan_to_num(v_t, nan=-1.0, posinf=-1.0, neginf=-1.0)
            u_t = np.clip(u_t, -1.0, W_img + 1.0)
            v_t = np.clip(v_t, -1.0, H_img + 1.0)
            u_int = np.round(u_t).astype(np.int32)
            v_int = np.round(v_t).astype(np.int32)

            in_bounds = (u_int >= 0) & (u_int < W_img) & (v_int >= 0) & (v_int < H_img) & (z_t > 0.1)

            # Depth consistency check
            depth_at_proj = np.zeros(N, dtype=np.float32)
            valid_px = in_bounds
            depth_at_proj[valid_px] = depths[t, v_int[valid_px], u_int[valid_px]]

            depth_match = np.abs(depth_at_proj - z_t) / (z_t + 1e-6) < depth_tol
            vis[t] = (in_bounds & depth_match).astype(np.float32)

        return {
            'images': images,
            'depth': depths,
            'intrinsic': intrinsics,
            'extrinsic': extrinsics,
            'traj3d': traj3d,     # (T, N, 3) world coords (static)
            'vis': vis,           # (T, N) visibility
        }

    # -----------------------------------------------------------------------
    # Dense H×W grid construction
    # -----------------------------------------------------------------------


    def _make_dense_grid_sparse(self, traj3d_sparse, vis_sparse,
                                 intrinsics_aug, extrinsics, crop_info):
        """Create dense H×W traj3d grid from sparse tracks (PO/DR).

        Project sparse 3D tracks to 2D using augmented intrinsics at frame 0,
        scatter onto H×W grid. Grid positions are fixed by frame-0 projection.
        At each frame t, the same grid cell stores the tracked point's world coords.

        Args:
            traj3d_sparse: (T, N_sparse, 3) world coords
            vis_sparse:    (T, N_sparse) visibility
            intrinsics_aug: (T, 3, 3) augmented intrinsics (after scale+crop)
            extrinsics: (T, 4, 4) c2w
            crop_info: dict with 'scale', 'left', 'top'

        Returns:
            traj3d: (T, H, W, 3) dense grid
            vis:    (T, H, W) visibility
        """
        T, N_sparse, _ = traj3d_sparse.shape
        H, W = self.height, self.width

        traj3d = np.zeros((T, H, W, 3), dtype=np.float32)
        vis = np.zeros((T, H, W), dtype=np.float32)

        # Determine grid positions from frame-0 projection
        K_aug_0 = intrinsics_aug[0]     # (3, 3)
        c2w_0 = extrinsics[0]           # (4, 4)
        w2c_0 = np.linalg.inv(c2w_0)    # (4, 4)

        pts_3d_0 = traj3d_sparse[0]     # (N, 3)

        # Project all tracks to 2D at frame 0
        ones = np.ones((N_sparse, 1), dtype=np.float32)
        pts_homo = np.concatenate([pts_3d_0, ones], axis=-1)  # (N, 4)
        pts_cam = (w2c_0 @ pts_homo.T).T[:, :3]               # (N, 3)

        uv_homo = (K_aug_0 @ pts_cam.T).T                     # (N, 3)
        z_cam = uv_homo[:, 2]
        uv = uv_homo[:, :2] / (uv_homo[:, 2:3] + 1e-8)       # (N, 2)

        # Filter NaN/inf before int cast (can occur with real data tracks)
        finite_mask = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        uv[~finite_mask] = -1.0

        u_px = np.round(uv[:, 0]).astype(np.int32)
        v_px = np.round(uv[:, 1]).astype(np.int32)

        # Valid: in bounds + in front of camera + visible at any frame + finite
        in_bounds = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H) & (z_cam > 0) & finite_mask
        # Also require at least some visibility across frames
        any_vis = vis_sparse.max(axis=0) > 0.5  # (N,)
        valid = in_bounds & any_vis

        valid_indices = np.where(valid)[0]
        u_valid = u_px[valid]
        v_valid = v_px[valid]

        # Handle duplicate pixel assignments (multiple tracks → same pixel):
        # keep the one closest to camera at frame 0
        if len(valid_indices) > 0:
            depths_valid = pts_cam[valid, 2]
            # Create flat pixel index
            flat_px = v_valid * W + u_valid
            # For each unique pixel, keep the track with smallest depth
            unique_px, first_idx = np.unique(flat_px, return_index=True)
            # Actually, build a mapping: for each pixel, store the best track
            best_track_for_px = {}
            for i, (fp, ti, d) in enumerate(zip(flat_px, valid_indices, depths_valid)):
                if fp not in best_track_for_px or d < best_track_for_px[fp][1]:
                    best_track_for_px[fp] = (ti, d)

            # Extract final mapping
            final_track_indices = []
            final_v = []
            final_u = []
            for fp, (ti, _) in best_track_for_px.items():
                final_track_indices.append(ti)
                final_v.append(fp // W)
                final_u.append(fp % W)

            track_indices = np.array(final_track_indices, dtype=np.int64)
            v_grid = np.array(final_v, dtype=np.int32)
            u_grid = np.array(final_u, dtype=np.int32)

            # Fill the grid for all frames
            for t in range(T):
                traj3d[t, v_grid, u_grid] = traj3d_sparse[t, track_indices]
                vis[t, v_grid, u_grid] = vis_sparse[t, track_indices]

        # NOTE: No explicit flip needed here. Unlike _make_dense_grid_direct (which
        # receives a pre-formed dense grid in original pixel order), sparse tracks are
        # re-projected through K_aug_0 which already incorporates pad/scale/crop/flip.
        # The resulting (u_grid, v_grid) are already in the augmented (flipped) pixel
        # space, and the 3D coordinate values (world coords) will be correctly
        # transformed to camera_0 space during training via the flip-modified extrinsics.

        return traj3d, vis

    def _make_dense_grid_direct(self, traj3d_flat, vis_flat, crop_info, H_src, W_src,
                                conf_flat=None, camera_stats=None):
        """Convert dense tracks from source resolution to augmented resolution.

        For datasets where tracks are already in a dense H_src×W_src pixel grid
        (e.g., DynPose with N = H_src*W_src). Applies the same spatial transform
        (scale + crop + flip) as images, avoiding 3D re-projection.

        Args:
            traj3d_flat: (T, H_src*W_src, 3) coords in source pixel grid
            vis_flat:    (T, H_src*W_src) visibility
            crop_info: dict with 'scale', 'left', 'top', 'do_h_flip', 'do_v_flip'
            H_src, W_src: source image dimensions
            conf_flat:   (T, H_src*W_src) confidence, optional
            camera_stats: dict with 'mean' (1,3) and 'max' float, from UN-FLIPPED
                          81-frame extrinsics. When provided, corrects the 3D
                          coordinates after flip so that projection through the
                          flipped (but UN-FLIPPED-stats-normalized) cameras gives
                          the correct pixel positions.

        Returns:
            traj3d: (T, H_out, W_out, 3) dense grid
            vis:    (T, H_out, W_out) visibility
            conf:   (T, H_out, W_out) confidence (or None)
        """
        T = traj3d_flat.shape[0]
        H_out, W_out = self.height, self.width

        # Per-frame spatial params (from MV-TAP-style per-frame crop_info)
        sx_list   = crop_info.get('sx_list',   [crop_info.get('scale', 1.0)] * T)
        sy_list   = crop_info.get('sy_list',   [crop_info.get('scale', 1.0)] * T)
        left_list = crop_info.get('left_list', [crop_info.get('left',  0)]   * T)
        top_list  = crop_info.get('top_list',  [crop_info.get('top',   0)]   * T)

        # Reshape to spatial grid
        traj3d_grid = traj3d_flat.reshape(T, H_src, W_src, 3)
        vis_grid = vis_flat.reshape(T, H_src, W_src)
        conf_grid = conf_flat.reshape(T, H_src, W_src) if conf_flat is not None else None

        # Zero out inf/nan tracks (can occur from float16 storage)
        bad_mask = ~np.isfinite(traj3d_grid).all(axis=-1)  # (T, H_src, W_src)
        traj3d_grid[bad_mask] = 0.0
        vis_grid[bad_mask] = 0.0
        if conf_grid is not None:
            conf_grid[bad_mask] = 0.0

        # Apply padding: sx_list[t] = W_int / W_pad where W_pad = W_src + pad_x0 + pad_x1.
        # Must pad the grid so that H_src * sy = H_int (correct scale target).
        # Padded regions get traj=0, vis=0 (excluded from loss).
        pad_x0 = crop_info.get('pad_x0', 0)
        pad_x1 = crop_info.get('pad_x1', 0)
        pad_y0 = crop_info.get('pad_y0', 0)
        pad_y1 = crop_info.get('pad_y1', 0)
        if pad_x0 or pad_x1 or pad_y0 or pad_y1:
            traj3d_grid = np.pad(traj3d_grid,
                                 ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
                                 mode='constant', constant_values=0.0)
            vis_grid = np.pad(vis_grid,
                              ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)),
                              mode='constant', constant_values=0.0)
            if conf_grid is not None:
                conf_grid = np.pad(conf_grid,
                                   ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)),
                                   mode='constant', constant_values=0.0)
            H_src = H_src + pad_y0 + pad_y1  # H_pad: base for sy scaling
            W_src = W_src + pad_x0 + pad_x1  # W_pad: base for sx scaling

        # Use FRAME-0 crop parameters to fix track identity across all frames.
        # Per-frame scale drift (sx_list[t] ≠ sx_list[0]) would cause different source
        # pixels to be selected at each frame, making output pixel (y,x) track different
        # surface points per frame → bleeding into background.
        # Sparse datasets (Kubric etc.) don't have this problem because their tracks have
        # explicit per-frame (u_t, v_t) positions; scale/crop is just a coordinate transform.
        # Here, track identity = grid cell, so we must fix the source pixel mapping using
        # frame-0 parameters for ALL frames.
        sx_0 = sx_list[0]; sy_0 = sy_list[0]
        left_0 = left_list[0]; top_0 = top_list[0]
        H_scaled_0 = max(int(round(H_src * sy_0)), H_out)
        W_scaled_0 = max(int(round(W_src * sx_0)), W_out)

        v_out_idx, u_out_idx = np.mgrid[0:H_out, 0:W_out]
        # Inverse of scale+crop at frame 0: output (y,x) → source (v_src, u_src)
        v_src = np.clip(
            ((v_out_idx + top_0).astype(np.float32) * H_src / H_scaled_0).astype(np.int32),
            0, H_src - 1
        )
        u_src = np.clip(
            ((u_out_idx + left_0).astype(np.float32) * W_src / W_scaled_0).astype(np.int32),
            0, W_src - 1
        )

        # Gather all frames using the same frame-0 source mapping
        traj3d_out = traj3d_grid[:, v_src, u_src, :]   # (T, H_out, W_out, 3)
        vis_out = vis_grid[:, v_src, u_src]              # (T, H_out, W_out)
        conf_out = conf_grid[:, v_src, u_src] if conf_grid is not None else None

        # Apply flips (matching image transform order)
        # For DynPose: traj3d is in 81-frame normalized space, so after spatial flip
        # we must also transform the coordinates to the flipped normalized space.
        # Derivation: traj_needed_x = -traj_unflipped_x[v, W-1-u] - 2*mean_x/max_dist
        #             After spatial flip traj_flip[v,u] = traj_unflipped[v,W-1-u], so:
        #             traj_flip_x_new = -traj_flip_x - 2 * mean_x * s  (s = 1/max_dist)
        if camera_stats is not None:
            mean = np.asarray(camera_stats['mean']).reshape(3)
            s = 1.0 / (float(camera_stats['max']) + 1e-6)

        if crop_info.get('do_h_flip', False):
            traj3d_out = traj3d_out[:, :, ::-1, :].copy()
            vis_out = vis_out[:, :, ::-1].copy()
            if conf_out is not None:
                conf_out = conf_out[:, :, ::-1].copy()
            if camera_stats is not None:
                traj3d_out[:, :, :, 0] = -traj3d_out[:, :, :, 0] - 2.0 * mean[0] * s

        if crop_info.get('do_v_flip', False):
            traj3d_out = traj3d_out[:, ::-1, :, :].copy()
            vis_out = vis_out[:, ::-1, :].copy()
            if conf_out is not None:
                conf_out = conf_out[:, ::-1, :].copy()
            if camera_stats is not None:
                traj3d_out[:, :, :, 1] = -traj3d_out[:, :, :, 1] - 2.0 * mean[1] * s

        return traj3d_out, vis_out, conf_out

    # -----------------------------------------------------------------------
    # Output formatting
    # -----------------------------------------------------------------------

    def _to_output_dict(self, images, depths, intrinsics, extrinsics, traj3d, vis, fg_mask=None, conf=None):
        """Convert augmented data to final output dict.

        traj3d is (T, H, W, 3) dense grid → flatten to (T, H*W, 3) for model.
        vis is (T, H, W) → flatten to (T, H*W).
        All samples have the same N = H*W for easy collation with torch.stack.
        """
        T = len(images)

        # Flatten spatial dims: (T, H, W, 3) → (T, H*W, 3)
        traj3d_flat = traj3d.reshape(T, -1, 3)
        vis_flat = vis.reshape(T, -1)

        # Convert to output format
        video = [Image.fromarray(img) for img in images]
        depth_tensor = torch.from_numpy(depths).unsqueeze(1)         # (T, 1, H, W)
        intrinsic_tensor = torch.from_numpy(intrinsics)              # (T, 3, 3)
        extrinsic_tensor = torch.from_numpy(extrinsics.copy())       # (T, 4, 4)
        traj3d_tensor = torch.from_numpy(traj3d_flat)                # (T, H*W, 3)
        vis_tensor = torch.from_numpy(vis_flat)                      # (T, H*W)
        if conf is not None:
            conf_flat = conf.reshape(T, -1)
            conf_tensor = torch.from_numpy(conf_flat)                # (T, H*W)
        else:
            conf_tensor = torch.ones_like(vis_tensor)                # (T, H*W)

        result = {
            'video': video,
            'prompt': '',
            'depth': depth_tensor,
            'intrinsic': intrinsic_tensor,
            'extrinsic': extrinsic_tensor,
            'traj3d': traj3d_tensor,
            'vis': vis_tensor,
            'conf': conf_tensor,
        }

        # Foreground mask: (T, H, W) → (T, 1, H, W) for Kubric
        # PO/DR don't have segmentation, so fg_mask=None → all foreground
        if fg_mask is not None:
            result['fg_mask'] = torch.from_numpy(fg_mask).unsqueeze(1)  # (T, 1, H, W)
        else:
            H, W = depths.shape[1], depths.shape[2]
            result['fg_mask'] = torch.ones(T, 1, H, W, dtype=torch.float32)

        return result
