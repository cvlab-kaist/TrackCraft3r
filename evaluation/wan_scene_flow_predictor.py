"""TrackCraft3R inference / evaluation predictor.

Wan2.1-T2V-1.3B + LoRA + diagonal-condition-row + pixel-delta with
predicted visibility. Loads a stage-2 release checkpoint and produces
(T, M, 3) sparse 3D tracks in the frame-0 camera space.
"""

import copy

import torch
import numpy as np
from PIL import Image

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models import load_state_dict
from peft import LoraConfig, inject_adapter_in_model

from .base_predictor import BasePredictor


class WanSceneFlowPredictor(BasePredictor):
    """Sparse 3D track predictor.

    Forward path per `predict()` call:
        1. Resize RGB frames to model resolution (stretch or AR-preserving pad).
        2. Per-frame VAE-encode RGB frames -> clean latents.
        3. Compute Pj(tj) by unprojecting depth into frame-0 camera space,
           normalize (z-inlier percentile + max-distance), VAE-encode through
           the separate pointmap VAE -> Pj latents.
        4. DiT forward (RGB | Pj on the input stream) -> doubled-channel
           query latent (xyz | vis).
        5. Decode xyz half through VAE -> (3, T, H, W) normalized P0(tj) delta.
           Decode vis half through vae_vis -> sigmoid -> (T, H, W) visibility.
        6. Reconstruct P0(tj) = (delta + P0(t0)_norm) * pj_scale + pj_mean.
        7. Sample at query UV (frame-0 visibility-positive points) -> (T, M, 3).
    """

    @property
    def is_dense(self):
        return False

    def __init__(self,
                 checkpoint_path,
                 model_id="Wan-AI/Wan2.1-T2V-1.3B",
                 lora_rank=1024,
                 lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                 height=480, width=832,
                 device="cuda",
                 regression_timestep=-1,
                 diagonal_condition_row=True,
                 pj_norm_inlier=True,
                 diag_max_depth=80.0,
                 pj_norm_percentile_lo=2.0,
                 pj_norm_percentile_hi=98.0,
                 pixel_delta=True,
                 track_latent_length=12,
                 resize_mode="stretch",
                 predict_vis=True,
                 vis_separate_decoder=True,
                 pj_separate_encoder=True,
                 parallel_vae_decode=True,
                 apply_speed_opts=True):
        self.height = height
        self.parallel_vae_decode = parallel_vae_decode

        # Process-wide speed optimizations (idempotent; safe to call repeatedly).
        if apply_speed_opts:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.width = width
        self.device = device
        self.regression_timestep = regression_timestep
        self.diagonal_condition_row = diagonal_condition_row
        self.pj_norm_inlier = pj_norm_inlier
        self.diag_max_depth = diag_max_depth
        self.pj_norm_percentile_lo = pj_norm_percentile_lo
        self.pj_norm_percentile_hi = pj_norm_percentile_hi
        self.pixel_delta = pixel_delta
        self.resize_mode = resize_mode
        self.predict_vis = predict_vis
        self.vis_separate_decoder = vis_separate_decoder
        self.pj_separate_encoder = pj_separate_encoder

        # 1. Load base Wan2.1 pipeline (DiT + VAE + T5)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu",
            model_configs=[
                ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id=model_id, origin_file_pattern="Wan2*_VAE.pth"),
            ],
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # 2. LoRA adapters
        self.use_lora = lora_rank > 0
        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_rank,
                target_modules=lora_target_modules.split(","),
            )
            self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
            print(f"  LoRA injected: rank={lora_rank}, targets={lora_target_modules}")

        # 3. Expand patch_embedding 16 -> 32 ch (RGB | Pj concatenated input)
        if getattr(self.pipe.dit, 'has_image_input', False):
            self.pipe.dit.has_image_input = False
        old_pe = self.pipe.dit.patch_embedding
        vae_z_dim = getattr(self.pipe.vae, 'z_dim', old_pe.in_channels)
        new_ch = vae_z_dim * 2
        new_pe = torch.nn.Conv3d(
            new_ch, old_pe.out_channels,
            kernel_size=old_pe.kernel_size, stride=old_pe.stride,
        )
        with torch.no_grad():
            src_ch = min(vae_z_dim, old_pe.in_channels)
            new_pe.weight[:, :src_ch] = old_pe.weight[:, :src_ch].clone()
            new_pe.weight[:, vae_z_dim:vae_z_dim + src_ch] = old_pe.weight[:, :src_ch].clone()
            if old_pe.bias is not None:
                new_pe.bias.copy_(old_pe.bias)
        self.pipe.dit.patch_embedding = new_pe.to(dtype=torch.bfloat16)
        print(f"  patch_embedding expanded: {old_pe.in_channels}->{new_ch} channels")

        # 4. Set DiT mode flags (must match training)
        self.pipe.dit.concat_mode = "scene_flow"
        self.pipe.dit.scene_flow_query_mode = "frame0_latent"
        self.pipe.dit.scene_flow_query_rope_mode = "image"
        self.pipe.regress_row = False
        self.pipe.regress_diagonal = False
        self.pipe.dit.regress_row = False
        self.pipe.dit.regress_diagonal = False
        self.pipe.diagonal_condition_row = diagonal_condition_row
        self.pipe.dit.diagonal_condition_row = diagonal_condition_row
        self.pipe.pj_norm_inlier = pj_norm_inlier
        self.pipe.dit.p0_rope_frame0 = False
        self.pipe.predict_delta = False
        self.pipe.dit.predict_delta = False
        self.pipe.pixel_delta = pixel_delta
        self.pipe.dit.pixel_delta = pixel_delta
        self.pipe.image_only_row = False
        self.pipe.dit.image_only_row = False
        self.pipe.generation_mode = False
        self.pipe.dit.generation_mode = False
        self.pipe.num_inference_steps = 1
        self.pipe.regression_timestep = regression_timestep
        self.pipe.dit.track_modality_embedding = None
        self.pipe.dit.image_modality_embedding = None
        self.pipe.pj_token_prepend = False
        self.pipe.dit.pj_token_prepend = False
        self.pipe.vae_temporal_compression = False

        # 5. Expand DiT head to 2x output (xyz | vis)
        if predict_vis:
            assert hasattr(self.pipe.dit.head, 'expand_output_for_vis'), \
                "predict_vis requires a DiT Head with expand_output_for_vis()"
            self.pipe.dit.head.expand_output_for_vis()
            print("  DiT head expanded to 2x output (xyz | vis)")
        self.pipe.predict_vis = predict_vis

        # 6. Separate vae_vis / vae_pj (deep-copy from primary VAE)
        self.pipe.vis_separate_decoder = vis_separate_decoder
        self.pipe.pj_separate_encoder = pj_separate_encoder
        self.pipe.vae_vis = (copy.deepcopy(self.pipe.vae)
                             if predict_vis and vis_separate_decoder else None)
        self.pipe.vae_pj = (copy.deepcopy(self.pipe.vae)
                            if pj_separate_encoder else None)

        # 7. Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # 8. Move to device. VAE encoders/decoders use channels_last on
        #    Ampere+ for ~5-30% speedup (DiT stays default — channels_last
        #    typically hurts transformers).
        for m in (self.pipe.dit, self.pipe.vae, self.pipe.vae_vis, self.pipe.vae_pj):
            if m is not None:
                m.to(device=device, dtype=torch.bfloat16).eval()
        if apply_speed_opts:
            for vae_attr in ('vae', 'vae_vis', 'vae_pj'):
                vae = getattr(self.pipe, vae_attr, None)
                if vae is None:
                    continue
                for sub in ('encoder', 'decoder'):
                    sub_m = getattr(vae, sub, None)
                    if sub_m is not None:
                        sub_m.to(memory_format=torch.channels_last)

        # 9. Cache null text context
        self.pipe.prompter.text_encoder.to(device=device)
        with torch.no_grad():
            self._null_context = self.pipe.prompter.encode_prompt(
                "", positive=True, device=device,
            ).to(dtype=torch.bfloat16, device=device)
        self.pipe.prompter.text_encoder.to(device="cpu")
        torch.cuda.empty_cache()

        print(f"WanSceneFlowPredictor loaded from {checkpoint_path}")

    # ------------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_path):
        ckpt = load_state_dict(checkpoint_path, torch_dtype=torch.bfloat16)
        dit_sd, vae_sd, vae_vis_sd, vae_pj_sd = {}, {}, {}, {}
        for key, val in ckpt.items():
            # vae_vis / vae_pj must be checked BEFORE vae (substring match).
            if key.startswith("pipe.vae_vis."):
                vae_vis_sd[key[len("pipe.vae_vis."):]] = val
            elif key.startswith("pipe.vae_pj."):
                vae_pj_sd[key[len("pipe.vae_pj."):]] = val
            elif key.startswith("pipe.vae."):
                vae_sd[key[len("pipe.vae."):]] = val
            elif key.startswith("pipe.dit."):
                dit_sd[key[len("pipe.dit."):]] = val

        if dit_sd:
            r = self.pipe.dit.load_state_dict(dit_sd, strict=False)
            print(f"  DiT loaded: {len(dit_sd)} keys "
                  f"(unexpected={len(r.unexpected_keys)}, missing={len(r.missing_keys)})")
        if vae_sd:
            self.pipe.vae.load_state_dict(vae_sd, strict=False)
            print(f"  VAE loaded: {len(vae_sd)} keys")
        if vae_vis_sd and self.pipe.vae_vis is not None:
            self.pipe.vae_vis.load_state_dict(vae_vis_sd, strict=False)
            print(f"  VAE_vis loaded: {len(vae_vis_sd)} keys")
        if vae_pj_sd and self.pipe.vae_pj is not None:
            self.pipe.vae_pj.load_state_dict(vae_pj_sd, strict=False)
            print(f"  VAE_pj loaded: {len(vae_pj_sd)} keys")

    # ------------------------------------------------------------------
    def _resize(self, images_pil):
        orig_w, orig_h = images_pil[0].width, images_pil[0].height
        if self.resize_mode == "pad":
            ar_scale = min(self.width / orig_w, self.height / orig_h)
            new_w = int(round(orig_w * ar_scale))
            new_h = int(round(orig_h * ar_scale))
            pad_left = (self.width - new_w) // 2
            pad_top = (self.height - new_h) // 2
            resized = []
            for img in images_pil:
                canvas = Image.new("RGB", (self.width, self.height), (0, 0, 0))
                canvas.paste(img.resize((new_w, new_h), Image.BILINEAR), (pad_left, pad_top))
                resized.append(canvas)
            return resized, ar_scale, new_w, new_h, pad_left, pad_top, orig_w, orig_h
        ar_scale = None
        new_w, new_h = self.width, self.height
        return [img.resize((new_w, new_h), Image.BILINEAR) for img in images_pil], \
               ar_scale, new_w, new_h, 0, 0, orig_w, orig_h

    @staticmethod
    def _scale_intrinsics(intr, ar_scale, pad_left, pad_top, width, orig_w,
                          height, orig_h):
        """Scale a 4-vec [fx, fy, cx, cy] from original-image space to model space."""
        fx, fy, cx, cy = intr
        if ar_scale is not None:
            return fx * ar_scale, fy * ar_scale, cx * ar_scale + pad_left, cy * ar_scale + pad_top
        return fx * (width / orig_w), fy * (height / orig_h), \
               cx * (width / orig_w),  cy * (height / orig_h)

    @torch.no_grad()
    def predict(self, images_pil, query_uv, visibility, intrinsics,
                depth_map=None, extrinsics_w2c=None):
        """Predict 3D tracks at `query_uv` in original-image pixel space.

        `intrinsics` is the GT 4-vec [fx, fy, cx, cy]; the paper protocol uses
        GT intrinsics paired with predicted depth+pose.

        Returns (T, M', 3) where M' <= M (out-of-bounds queries are filtered).
        """
        assert depth_map is not None and extrinsics_w2c is not None, (
            "TrackCraft3R requires depth_map and extrinsics_w2c — provide GT, "
            "DepthAnything-3, or ViPE outputs.")

        resized, ar_scale, new_w, new_h, pad_left, pad_top, orig_w, orig_h = \
            self._resize(images_pil)

        # 1. VAE-encode RGB
        video_tensor = self.pipe.preprocess_video(resized)  # (1, 3, T, H, W)
        B, C, T, H, W = video_tensor.shape
        flat = video_tensor.transpose(1, 2).reshape(B * T, C, 1, H, W)
        input_latents = self.pipe.vae.encode(
            flat.to(device=self.device, dtype=torch.bfloat16),
            device=self.device, tiled=False,
        ).to(dtype=torch.bfloat16, device=self.device)
        _, C_l, _, H_l, W_l = input_latents.shape
        input_latents = input_latents.reshape(B, T, C_l, H_l, W_l).transpose(1, 2)

        # 2. Compute Pj(tj) in frame-0 camera space and VAE-encode it
        fx_s, fy_s, cx_s, cy_s = self._scale_intrinsics(
            intrinsics, ar_scale, pad_left, pad_top,
            self.width, orig_w, self.height, orig_h)

        c2w = np.linalg.inv(extrinsics_w2c)
        w2c_0 = extrinsics_w2c[0]

        T_dm = min(depth_map.shape[0], T)
        depth_resized = []
        for t_idx in range(T_dm):
            d_pil = Image.fromarray(depth_map[t_idx].astype(np.float32), mode='F')
            d_resized = d_pil.resize((new_w, new_h), Image.NEAREST)
            d_canvas = np.zeros((H, W), dtype=np.float32)
            d_canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = np.array(d_resized)
            depth_resized.append(d_canvas)
        depth_resized = np.stack(depth_resized)

        vv, uu = np.meshgrid(np.arange(H, dtype=np.float32),
                             np.arange(W, dtype=np.float32), indexing='ij')
        pts_all_frames = []
        for j in range(T_dm):
            d_j = depth_resized[j]
            pts = np.stack([(uu - cx_s) / fx_s * d_j,
                            (vv - cy_s) / fy_s * d_j,
                            d_j, np.ones_like(d_j)], axis=-1).reshape(-1, 4)
            pts_cam0 = (pts @ c2w[j].T) @ w2c_0.T
            pts_all_frames.append(pts_cam0[:, :3].reshape(H, W, 3))
        pts_all = np.stack(pts_all_frames)  # (T, H, W, 3)

        if self.diag_max_depth > 0:
            pts_all[..., 2] = np.clip(pts_all[..., 2], 1e-6, self.diag_max_depth)

        # z-inlier (percentile) -> mean center -> max-distance scale.
        z_all = pts_all[..., 2].reshape(-1)
        z_lo = np.percentile(z_all, self.pj_norm_percentile_lo)
        z_hi = np.percentile(z_all, self.pj_norm_percentile_hi)
        inlier_mask = (pts_all[..., 2] >= z_lo) & (pts_all[..., 2] <= z_hi)
        inlier_pts = pts_all[inlier_mask]
        pj_mean = inlier_pts.mean(axis=0)
        centered = pts_all - pj_mean
        pj_scale = np.linalg.norm(centered[inlier_mask], axis=-1).max()
        normalized_pj = centered / (pj_scale + 1e-6)

        pj_tensor = torch.from_numpy(normalized_pj).permute(0, 3, 1, 2).float().unsqueeze(2)
        pj_vae = self.pipe.vae_pj if self.pj_separate_encoder else self.pipe.vae
        pj_latents = pj_vae.encode(
            pj_tensor.to(device=self.device, dtype=torch.bfloat16),
            device=self.device, tiled=False,
        ).to(dtype=torch.bfloat16, device=self.device)
        _, C_pj, _, H_pj, W_pj = pj_latents.shape
        pj_latents = pj_latents.reshape(1, T_dm, C_pj, H_pj, W_pj).transpose(1, 2)

        self._last_pj_input = pts_all  # raw Pj(tj) in cam_0 space, for --save_dense

        # 3. DiT forward (single-step regression)
        context = self._null_context.expand(B, -1, -1)
        timestep = self.pipe.scheduler.timesteps[self.regression_timestep].unsqueeze(0).to(
            dtype=torch.bfloat16, device=self.device)
        query_latent = self.pipe.model_fn(
            dit=self.pipe.dit,
            latents=input_latents,
            timestep=timestep,
            context=context,
            cfg_merge=False,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            pj_latents=pj_latents,
        )

        # 4. Decode xyz / vis halves (optionally on parallel CUDA streams)
        if self.predict_vis and query_latent.shape[1] == 2 * self.pipe.vae.z_dim:
            xyz_lat, vis_lat = query_latent.chunk(2, dim=1)
            if self.parallel_vae_decode:
                s_xyz, s_vis = torch.cuda.Stream(), torch.cuda.Stream()
                torch.cuda.synchronize()
                with torch.cuda.stream(s_xyz):
                    traj3d_pred = self._decode_latents(xyz_lat)
                with torch.cuda.stream(s_vis):
                    vis_decoded = self._decode_latents(vis_lat, vae_module=self.pipe.vae_vis)
                torch.cuda.synchronize()
            else:
                traj3d_pred = self._decode_latents(xyz_lat)
                vis_decoded = self._decode_latents(vis_lat, vae_module=self.pipe.vae_vis)
            vis_logit = vis_decoded.mean(dim=1, keepdim=True)
            vis_pred_np = torch.sigmoid(vis_logit[0, 0]).cpu().float().numpy()
            del vis_decoded, vis_logit
        else:
            traj3d_pred = self._decode_latents(query_latent)
            vis_pred_np = None
        traj3d_np = traj3d_pred[0].cpu().float().numpy()  # (3, T, H, W)
        del traj3d_pred

        _, _, H_out, W_out = traj3d_np.shape

        # 5. pixel_delta + diagonal_condition_row reconstruction:
        #    delta -> add P0(t0)_norm -> denormalize.
        if self.pixel_delta and self.diagonal_condition_row:
            p0_t0_norm = normalized_pj[0]                                   # (H, W, 3)
            p0_t0_3hw = np.transpose(p0_t0_norm, (2, 0, 1))[:, np.newaxis]  # (3, 1, H, W)
            traj3d_np = traj3d_np + p0_t0_3hw
            traj3d_np = traj3d_np * (pj_scale + 1e-6) + pj_mean.reshape(3, 1, 1, 1)

        # Cache dense outputs for --save_dense
        self._last_row_dense = traj3d_np.transpose(1, 2, 3, 0)
        self._last_rgb_frames = np.stack([np.array(img) for img in resized], axis=0)
        self._last_vis_dense = vis_pred_np

        # 6. Scale query UV to model output resolution
        if self.resize_mode == "pad":
            query_uv_scaled = query_uv * ar_scale + np.array([pad_left, pad_top], dtype=np.float64)
        else:
            query_uv_scaled = query_uv * np.array(
                [W_out / orig_w, H_out / orig_h], dtype=np.float64)
        self._last_query_uv_model = query_uv_scaled.copy()

        # OOB filter
        oob_mask = (
            (query_uv_scaled[:, 0] >= 0) & (query_uv_scaled[:, 0] < W_out) &
            (query_uv_scaled[:, 1] >= 0) & (query_uv_scaled[:, 1] < H_out)
        )
        if oob_mask.sum() < len(query_uv_scaled):
            print(f"[predict] {len(query_uv_scaled) - oob_mask.sum()} OOB points filtered")
            query_uv_scaled = query_uv_scaled[oob_mask]

        u_q = query_uv_scaled[:, 0].astype(int)
        v_q = query_uv_scaled[:, 1].astype(int)
        pred_tracks = traj3d_np[:, :, v_q, u_q].transpose(1, 2, 0)  # (T, M', 3)
        self._last_oob_mask = oob_mask

        del input_latents, query_latent
        torch.cuda.empty_cache()
        return pred_tracks

    # ------------------------------------------------------------------
    def _decode_latents(self, latents, mini_batch=12, vae_module=None):
        """Decode (B, C, T_lat, H_l, W_l) -> (B, 3, T, H, W) per-frame."""
        vae = vae_module if vae_module is not None else self.pipe.vae
        b, c, t, h, w = latents.shape
        flat = latents.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        chunks = []
        for i in range(0, flat.shape[0], mini_batch):
            chunk = flat[i:i + mini_batch].unsqueeze(2)
            decoded = vae.decode(chunk, device=self.device, tiled=False).squeeze(2)
            chunks.append(decoded)
            del decoded, chunk
            torch.cuda.empty_cache()
        decoded_flat = torch.cat(chunks, dim=0)
        _, c_out, h_out, w_out = decoded_flat.shape
        return decoded_flat.view(b, t, c_out, h_out, w_out).permute(0, 2, 1, 3, 4)
