import torch, os, json, time
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        track_latent_length=None,
        regression_timestep=None,
        height=480,
        width=832,
        traj3d_coord_space="world",
        point_norm_mode="mean_scale",
        diag_max_depth=80.0,
        vae_train_parts="decoder",
        predict_vis=False,
        vis_loss_weight=1.0,
        vis_separate_decoder=False,
        pj_separate_encoder=False,
    ):
        super().__init__()
        self.traj3d_coord_space = traj3d_coord_space
        self.track_latent_length = track_latent_length

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Expand patch_embedding from C_z (RGB only) to 2*C_z (RGB + Pj).
        if getattr(self.pipe.dit, 'has_image_input', False):
            self.pipe.dit.has_image_input = False
        old_pe = self.pipe.dit.patch_embedding
        vae_z_dim = getattr(self.pipe.vae, 'z_dim', old_pe.in_channels)
        new_ch = vae_z_dim * 2
        new_pe = torch.nn.Conv3d(new_ch, old_pe.out_channels,
                                 kernel_size=old_pe.kernel_size, stride=old_pe.stride)
        with torch.no_grad():
            src_ch = min(vae_z_dim, old_pe.in_channels)
            new_pe.weight[:, :src_ch] = old_pe.weight[:, :src_ch].clone()
            new_pe.weight[:, vae_z_dim:vae_z_dim + src_ch] = old_pe.weight[:, :src_ch].clone()
            if old_pe.bias is not None:
                new_pe.bias.copy_(old_pe.bias)
        self.pipe.dit.patch_embedding = new_pe.to(dtype=self.pipe.torch_dtype)
        self.pipe.dit.patch_embedding.requires_grad_(True)

        self.pipe.regression_timestep = regression_timestep if regression_timestep is not None else 0
        self.pipe.point_norm_mode = point_norm_mode
        self.diag_max_depth = diag_max_depth
        self.pipe.diag_max_depth = diag_max_depth
        self.pixel_delta = True
        self.pj_norm_inlier = True

        # Optional: per-frame visibility prediction via dual-latent DiT output.
        # When enabled, DiT's output head is expanded to 2x channels. The output
        # is split into [xyz_latent | vis_latent]; both halves are decoded
        # independently through the shared (frozen) VAE decoder. The first half
        # reconstructs the xyz delta (existing behavior, preserved weights); the
        # second half is zero-initialized and learns per-frame visibility via
        # balanced BCE on the decoded 3-ch output (averaged to 1 ch).
        self.predict_vis = predict_vis
        self.pipe.predict_vis = predict_vis
        self.vis_loss_weight = vis_loss_weight
        self.pipe.vis_loss_weight = vis_loss_weight
        if predict_vis:
            dit_head = getattr(self.pipe.dit, 'head', None)
            if dit_head is None or not hasattr(dit_head, 'expand_output_for_vis'):
                raise RuntimeError(
                    "--predict_vis requires a DiT Head with expand_output_for_vis(); "
                    "the current DiT does not support it.")
            dit_head.expand_output_for_vis()
            # The newly-expanded head contains pretrained xyz weights + zero-init
            # vis weights. Make it trainable so vis weights can learn.
            for p in dit_head.parameters():
                p.requires_grad_(True)
            print(f"✓ predict_vis=True: DiT head output doubled (xyz|vis), vis weights zero-init, "
                  f"loss_weight={vis_loss_weight}")

            # Optional: separate VAE decoder for vis prediction.
            # Useful when --trainable_models includes "vae" so xyz and vis decoders
            # can specialize without cross-task interference.
            self.vis_separate_decoder = vis_separate_decoder
            self.pipe.vis_separate_decoder = vis_separate_decoder
            if vis_separate_decoder:
                import copy
                # Deep-copy the VAE. Both VAEs start with identical pretrained weights.
                vae_vis = copy.deepcopy(self.pipe.vae)
                # Register as a child of pipe (so freeze_except/named_children see it).
                self.pipe.vae_vis = vae_vis
                # Mirror gradient checkpointing setting on the new VAE.
                self.pipe.vae_vis.use_gradient_checkpointing = use_gradient_checkpointing
                # Apply the same VAE training rules as pipe.vae (decoder/encoder parts).
                if "vae" in (trainable_models or ""):
                    self.pipe.vae_vis.requires_grad_(False)
                    parts = [p.strip() for p in vae_train_parts.split(",")]
                    for part in parts:
                        if part == "decoder":
                            if hasattr(self.pipe.vae_vis.model, "decoder"):
                                self.pipe.vae_vis.model.decoder.requires_grad_(True)
                            if hasattr(self.pipe.vae_vis.model, "conv2"):
                                self.pipe.vae_vis.model.conv2.requires_grad_(True)
                        elif part == "encoder":
                            if hasattr(self.pipe.vae_vis.model, "encoder"):
                                self.pipe.vae_vis.model.encoder.requires_grad_(True)
                            if hasattr(self.pipe.vae_vis.model, "conv1"):
                                self.pipe.vae_vis.model.conv1.requires_grad_(True)
                    n_vis_trainable = sum(p.numel() for p in self.pipe.vae_vis.parameters() if p.requires_grad)
                    print(f"✓ vis_separate_decoder=True: vae_vis deep-copied from pipe.vae "
                          f"({n_vis_trainable:,} trainable params via vae_train_parts={vae_train_parts})")
                else:
                    # If vae is not in trainable_models, vae_vis stays frozen (same as vae).
                    self.pipe.vae_vis.requires_grad_(False)
                    print(f"✓ vis_separate_decoder=True: vae_vis deep-copied, frozen (vae not in trainable_models)")
            else:
                self.pipe.vae_vis = None

        # Optional: separate VAE encoder for pointmap (Pj(tj)) encoding.
        # Without this, RGB and pointmap streams share pipe.vae.encoder.
        # With this, pointmap encoding is routed through pipe.vae_pj.encode
        # so the two encoders can specialize under fine-tuning.
        self.pj_separate_encoder = pj_separate_encoder
        self.pipe.pj_separate_encoder = pj_separate_encoder
        if pj_separate_encoder:
            import copy
            vae_pj = copy.deepcopy(self.pipe.vae)
            self.pipe.vae_pj = vae_pj
            self.pipe.vae_pj.use_gradient_checkpointing = use_gradient_checkpointing
            if "vae" in (trainable_models or ""):
                self.pipe.vae_pj.requires_grad_(False)
                parts = [p.strip() for p in vae_train_parts.split(",")]
                for part in parts:
                    if part == "decoder":
                        if hasattr(self.pipe.vae_pj.model, "decoder"):
                            self.pipe.vae_pj.model.decoder.requires_grad_(True)
                        if hasattr(self.pipe.vae_pj.model, "conv2"):
                            self.pipe.vae_pj.model.conv2.requires_grad_(True)
                    elif part == "encoder":
                        if hasattr(self.pipe.vae_pj.model, "encoder"):
                            self.pipe.vae_pj.model.encoder.requires_grad_(True)
                        if hasattr(self.pipe.vae_pj.model, "conv1"):
                            self.pipe.vae_pj.model.conv1.requires_grad_(True)
                n_pj_trainable = sum(p.numel() for p in self.pipe.vae_pj.parameters() if p.requires_grad)
                print(f"✓ pj_separate_encoder=True: vae_pj deep-copied from pipe.vae "
                      f"({n_pj_trainable:,} trainable params via vae_train_parts={vae_train_parts})")
            else:
                self.pipe.vae_pj.requires_grad_(False)
                print(f"✓ pj_separate_encoder=True: vae_pj deep-copied, frozen (vae not in trainable_models)")
        else:
            self.pipe.vae_pj = None

        # VAE training: selectively unfreeze encoder/decoder/both + bridge convs
        if "vae" in trainable_models:
            # 1. Freeze all VAE first
            self.pipe.vae.requires_grad_(False)
            total_params = sum(p.numel() for p in self.pipe.vae.parameters())

            # 2. Unfreeze requested parts (+ corresponding bridge conv)
            parts = [p.strip() for p in vae_train_parts.split(",")]
            for part in parts:
                if part == "decoder":
                    if hasattr(self.pipe.vae.model, "decoder"):
                        self.pipe.vae.model.decoder.requires_grad_(True)
                    # conv2: bridge latent → decoder input
                    if hasattr(self.pipe.vae.model, "conv2"):
                        self.pipe.vae.model.conv2.requires_grad_(True)
                    n = sum(p.numel() for m in [self.pipe.vae.model.decoder, self.pipe.vae.model.conv2]
                            for p in m.parameters() if p.requires_grad)
                    print(f"✓ VAE Decoder + conv2 trainable ({n:,} params)")
                elif part == "encoder":
                    if hasattr(self.pipe.vae.model, "encoder"):
                        self.pipe.vae.model.encoder.requires_grad_(True)
                    # conv1: bridge encoder output → mu/log_var
                    if hasattr(self.pipe.vae.model, "conv1"):
                        self.pipe.vae.model.conv1.requires_grad_(True)
                    n = sum(p.numel() for m in [self.pipe.vae.model.encoder, self.pipe.vae.model.conv1]
                            for p in m.parameters() if p.requires_grad)
                    print(f"✓ VAE Encoder + conv1 trainable ({n:,} params)")
                elif part == "":
                    pass  # empty string from split, skip
                else:
                    print(f"✗ Warning: VAE part '{part}' not found!")

            trainable_vae = sum(p.numel() for p in self.pipe.vae.parameters() if p.requires_grad)
            print(f"  Total trainable VAE params: {trainable_vae:,} / {total_params:,}")

        # Propagate gradient checkpointing flag to VAE
        self.pipe.vae.use_gradient_checkpointing = use_gradient_checkpointing


        # Print all trainable parameters summary
        print("\n=== Trainable Parameters Summary ===")
        for name in ["dit", "vae", "text_encoder", "image_encoder"]:
            model = getattr(self.pipe, name, None)
            if model is not None:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                status = "✓" if trainable > 0 else "✗"
                print(f"{status} {name}: {trainable:,} / {total:,} trainable")

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
    
    def _pixel_frames(self, tl):
        """Pixel frames per clip (no VAE temporal compression)."""
        return tl

    def _preprocess_raw_data(self, data):
        """Online preprocessing for raw video data (replaces offline preprocess.py).
        VAE encodes frames on-the-fly, slices to track_latent_length.

        """
        device = next(self.pipe.dit.parameters()).device
        tl = self.track_latent_length or 25
        B = len(data['video'])  # actual batch size

        # Online VAE encode (batch all B samples together for GPU efficiency)
        pf = self._pixel_frames(tl)  # pixel frames (1:1 with latent frames, no temporal compression)
        _t_preproc_start = time.time()
        # Check if VAE encoder is being trained (has any requires_grad=True params)
        vae_encoder_trainable = hasattr(self.pipe.vae.model, 'encoder') and \
            any(p.requires_grad for p in self.pipe.vae.model.encoder.parameters())

        # preprocess_video is always no_grad (PIL → tensor, no learnable params)
        with torch.no_grad():
            video_tensors = []
            for i in range(B):
                vt = self.pipe.preprocess_video(data['video'][i][:pf])  # (1, 3, pf, H, W)
                video_tensors.append(vt)
            video_tensor = torch.cat(video_tensors, dim=0)  # (B, 3, pf, H, W)
            _, C, T_pix, H, W = video_tensor.shape
            video_reshaped = video_tensor.transpose(1, 2).reshape(B * T_pix, C, 1, H, W)

        # Per-frame VAE encode: (B*T, C, 1, H, W) → reshape to (B, C_l, T, H_l, W_l)
        encode_ctx = torch.enable_grad() if vae_encoder_trainable else torch.no_grad()
        with encode_ctx:
            input_latents = self.pipe.vae.encode(
                video_reshaped.to(device=device, dtype=self.pipe.torch_dtype),
                device=device, tiled=False
            ).to(dtype=self.pipe.torch_dtype, device=device)  # (B*T, C_l, 1, H', W')
            _, C_l, _, H_l, W_l = input_latents.shape
            input_latents = input_latents.reshape(B, T_pix, C_l, H_l, W_l).transpose(1, 2)  # (B, C_l, T, H', W')

        with torch.no_grad():
            noise = torch.randn_like(input_latents)

        torch.cuda.synchronize()
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Timing VAE] encode={time.time() - _t_preproc_start:.2f}s  "
                  f"B={B} T={T_pix} ({B*T_pix} frames)")

        # Build inputs dict
        inputs = {
            'input_latents': input_latents,
            'latents': noise,
            'noise': noise,
            'height': H,
            'width': W,
            'num_frames': T_pix,
            'cfg_scale': 1,
            'tiled': False,
            'cfg_merge': False,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'use_gradient_checkpointing_offload': self.use_gradient_checkpointing_offload,
            'max_timestep_boundary': self.max_timestep_boundary,
            'min_timestep_boundary': self.min_timestep_boundary,
            'traj3d': data['traj3d'][:, :pf].to(device),         # (B, T_pixel, N, 3)
        }
        # vis: frame-0 visibility used to exclude unfilled sparse grid cells (PO/DR/Kubric)
        if 'vis' in data and data['vis'] is not None:
            inputs['vis'] = data['vis'][:, :pf].to(device)       # (B, T_pixel, N)
        # eraser_vis: per-frame mask for eraser/replace augmented regions
        if 'eraser_vis' in data and data['eraser_vis'] is not None:
            inputs['eraser_vis'] = data['eraser_vis'][:, :pf].to(device)  # (B, T_pixel, N)
        # Add conf if available (needed for confidence filtering)
        if 'conf' in data and data['conf'] is not None:
            inputs['conf'] = data['conf'][:, :pf].to(device)    # (B, T_pixel, N)
        # Add fg_mask if available (Kubric: foreground segmentation)
        if 'fg_mask' in data and data['fg_mask'] is not None:
            inputs['fg_mask'] = data['fg_mask'][:, :pf].to(device)  # (B, T_pixel, 1, H, W)
        # For camera_first coord transform: pass extrinsic, camera_stats, dataset_type
        if self.traj3d_coord_space == 'camera_first':
            if 'extrinsic' in data and data['extrinsic'] is not None:
                inputs['extrinsic'] = data['extrinsic'][:, :pf].to(device)  # (B, T, 4, 4)
            if 'camera_stats' in data and data['camera_stats'] is not None:
                inputs['camera_stats'] = {
                    'mean': data['camera_stats']['mean'].to(device),  # (B, 1, 3)
                    'max':  data['camera_stats']['max'].to(device),   # (B, 1)
                }
            if 'dataset_type' in data:
                inputs['dataset_type'] = data['dataset_type']  # list of B strings

        # Compute Pj(tj) from depth + cameras and encode to latents.
        if True:
            depth = data['depth'][:, :pf].to(device).float()          # (B, T, 1, H, W)
            intrinsic = data['intrinsic'][:, :pf].to(device).float()  # (B, T, 3, 3)
            extrinsic = data['extrinsic'][:, :pf].to(device).float()  # (B, T, 4, 4) c2w
            diag_max_depth = self.diag_max_depth

            # Ensure extrinsic is in inputs for camera_first transform in forward
            if 'extrinsic' not in inputs:
                inputs['extrinsic'] = data['extrinsic'][:, :pf].to(device)

            B_dc, T_dc, _, H_dc, W_dc = depth.shape
            vv, uu = torch.meshgrid(
                torch.arange(H_dc, device=device, dtype=torch.float32),
                torch.arange(W_dc, device=device, dtype=torch.float32), indexing='ij')

            pj_norm_params = []   # list of (mean, max_dist) per sample
            pj_frames_list = []   # (B,) list of (3, T, H, W)

            for b in range(B_dc):
                c2w_0 = extrinsic[b, 0]                     # (4, 4)
                w2c_0 = torch.linalg.inv(c2w_0)             # (4, 4)
                pts_all_frames = []
                for j in range(T_dc):
                    depth_j = depth[b, j, 0]                # (H, W)
                    K_j = intrinsic[b, j]                    # (3, 3)
                    c2w_j = extrinsic[b, j]                  # (4, 4)
                    fx, fy = K_j[0, 0], K_j[1, 1]
                    cx, cy = K_j[0, 2], K_j[1, 2]
                    # Unproject to frame-j camera space
                    x_cam = (uu - cx) / fx * depth_j
                    y_cam = (vv - cy) / fy * depth_j
                    z_cam = depth_j
                    ones = torch.ones_like(z_cam)
                    pts = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1).reshape(-1, 4)
                    # cam_j → world → cam_0
                    pts_cam0 = (pts @ c2w_j.T) @ w2c_0.T    # (N, 4)
                    pts_cam0_xyz = pts_cam0[:, :3].reshape(H_dc, W_dc, 3)
                    pts_all_frames.append(pts_cam0_xyz)

                pts_all = torch.stack(pts_all_frames, dim=0)  # (T, H, W, 3)
                # Z clamp [1e-6, diag_max_depth]
                pts_all[..., 2] = pts_all[..., 2].clamp(min=1e-6, max=diag_max_depth)

                if self.pj_norm_inlier:
                    # z-inlier normalization (0.02-0.98 percentile + max_dist)
                    z_all = pts_all[..., 2].reshape(-1)
                    z_q02 = torch.quantile(z_all, 0.02)
                    z_q98 = torch.quantile(z_all, 0.98)
                    inlier_mask = (pts_all[..., 2] >= z_q02) & (pts_all[..., 2] <= z_q98)
                    inlier_pts = pts_all[inlier_mask]
                    mean = inlier_pts.mean(dim=0)
                    centered = pts_all - mean
                    inlier_centered = centered[inlier_mask]
                    max_dist = inlier_centered.norm(dim=-1).max()
                    normalized_pj = centered / (max_dist + 1e-6)
                else:
                    # Legacy mode: mean + p99 normalization
                    all_points = pts_all.reshape(-1, 3)
                    mean = all_points.mean(dim=0)
                    centered = pts_all - mean
                    distances = centered.reshape(-1, 3).norm(dim=-1)
                    p99_dist = torch.quantile(distances, 0.99)
                    max_dist = p99_dist
                    normalized_pj = centered / (p99_dist + 1e-6)

                pj_norm_params.append((mean, max_dist))
                pj_frames_list.append(normalized_pj.permute(3, 0, 1, 2))  # (3, T, H, W)

            inputs['pj_norm_params'] = pj_norm_params

            # VAE-encode Pj(tj) 3D point maps.
            pj_maps = torch.stack(pj_frames_list, dim=0)  # (B, 3, T, H, W)
            vae_input = pj_maps.transpose(1, 2).reshape(B_dc * T_dc, 3, 1, H_dc, W_dc)

            # Route pointmap encoding through vae_pj when --pj_separate_encoder,
            # otherwise through the shared pipe.vae.
            pj_vae = self.pipe.vae_pj if getattr(self, 'pj_separate_encoder', False) else self.pipe.vae
            pj_encoder_trainable = hasattr(pj_vae.model, 'encoder') and \
                any(p.requires_grad for p in pj_vae.model.encoder.parameters())
            pj_encode_ctx = torch.enable_grad() if pj_encoder_trainable else torch.no_grad()
            with pj_encode_ctx:
                # Per-frame VAE encode
                pj_latents = pj_vae.encode(
                    vae_input.to(dtype=self.pipe.torch_dtype), device=device, tiled=False
                ).to(dtype=self.pipe.torch_dtype, device=device)
                _, C_pj, _, H_pj, W_pj = pj_latents.shape
                pj_latents = pj_latents.reshape(B_dc, T_dc, C_pj, H_pj, W_pj).transpose(1, 2)

            inputs['pj_latents'] = pj_latents           # (B, 16, T, H_l, W_l)

            torch.cuda.synchronize()
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"[Timing Pj(tj)] B={B_dc} T={T_dc}, pj_latents={pj_latents.shape}")

        return inputs

    def forward(self, data, inputs=None):
        # Online preprocessing of synthetic data
        if inputs is None:
            inputs = self._preprocess_raw_data(data)

        # Null text conditioning: compute once and cache
        if not hasattr(self, '_null_text_context') or self._null_text_context is None:
            with torch.no_grad():
                self._null_text_context = self.pipe.prompter.encode_prompt(
                    "", positive=True, device=next(self.pipe.dit.parameters()).device
                ).to(dtype=self.pipe.torch_dtype)
        device = inputs['input_latents'].device
        B = inputs['input_latents'].shape[0]
        inputs['context'] = self._null_text_context.to(device=device).expand(B, -1, -1)


        tl = self.track_latent_length
        pf = self._pixel_frames(tl) if tl is not None else tl  # pixel frames for aux data
        # Latent data (5D: B, C, F, H, W) — always slice to tl (latent frames)
        if tl is not None:
            if "input_latents" in inputs and inputs["input_latents"].dim() == 5:
                inputs["input_latents"] = inputs["input_latents"][:, :, :tl, :, :]
            if "latents" in inputs and inputs["latents"].dim() == 5:
                inputs["latents"] = inputs["latents"][:, :, :tl, :, :]
            if "noise" in inputs and inputs["noise"].dim() == 5:
                inputs["noise"] = inputs["noise"][:, :, :tl, :, :]

        H, W = inputs['height'], inputs['width']

        if True:  # track output (always)
            # Build conf_mask: all-ones × per-frame vis.
            # Per-frame vis properly handles eraser occlusions + natural occlusions.
            traj3d_sf = inputs["traj3d"][:, :pf, :, :].float()   # (B, T, H*W, 3)
            B_sf, T_sf, N_sf, _ = traj3d_sf.shape

            conf_mask_sf = torch.ones(B_sf, T_sf, N_sf, device=device)

            if 'vis' in inputs:
                vis_sf = inputs['vis'].float()                                  # (B, T, N)
                # Frame-0 visibility: points visible at frame 0 get loss at ALL frames
                # (amodal supervision — train through natural occlusion)
                conf_mask_sf = conf_mask_sf * (vis_sf[:, 0:1, :] > 0.5)       # frame-0 vis only

            # Eraser/replace augmentation mask: exclude artificially erased regions per-frame
            # (erased input → model can't predict → don't penalize)
            if 'eraser_vis' in inputs:
                eraser_vis_sf = inputs['eraser_vis'].float()                   # (B, T, N)
                conf_mask_sf = conf_mask_sf * eraser_vis_sf[:, :T_sf, :]

            # × fg_mask[frame-0] (Kubric: filter background pixels)
            if 'fg_mask' in inputs:
                fg = inputs['fg_mask'][:, 0, 0, :, :].reshape(B_sf, 1, -1).float()  # (B, 1, H*W)
                conf_mask_sf = conf_mask_sf * (fg > 0.5)

            # Transform traj3d to frame-0 camera space (like St4RTrack) if requested.
            if self.traj3d_coord_space == 'camera_first' and 'extrinsic' in inputs:
                extrinsic = inputs['extrinsic'].float()          # (B, T, 4, 4) c2w
                camera_stats = inputs.get('camera_stats', None)
                dataset_types = inputs.get('dataset_type', ['synthetic'] * B_sf)
                traj3d_cam_list = []
                for b in range(B_sf):
                    t3d = traj3d_sf[b]  # (T, N, 3)
                    dtype_b = dataset_types[b] if isinstance(dataset_types, list) else 'synthetic'
                    if dtype_b == 'dynpose' and camera_stats is not None:
                        # DynPose traj3d is in normalize_c2w space:
                        #   c2w_norm translations = (c2w_rel_t - mean) / max_dist
                        #   p_norm = R_j @ p_cam_j + t_norm_j
                        # Inverse: p_cam0_metric = p_norm - t_norm_j + t_rel_j  (per-frame)
                        mean_b = camera_stats['mean'][b, 0].float()  # (3,)
                        max_b  = camera_stats['max'][b, 0].float()   # scalar
                        c2w_0_inv = torch.linalg.inv(extrinsic[b, 0:1])  # (1, 4, 4)
                        c2w_rel = c2w_0_inv @ extrinsic[b]               # (T, 4, 4)
                        t_rel = c2w_rel[:, :3, 3]                        # (T, 3)
                        t_norm = (t_rel - mean_b) / (max_b + 1e-6)      # (T, 3)
                        t3d = t3d - t_norm.unsqueeze(1) + t_rel.unsqueeze(1)  # (T, N, 3)
                    else:
                        # PO/DR/Kubric: world coords → frame-0 camera coords
                        c2w_0 = extrinsic[b, 0]                      # (4, 4)
                        w2c_0 = torch.linalg.inv(c2w_0)
                        R = w2c_0[:3, :3]                            # (3, 3)
                        t = w2c_0[:3, 3]                             # (3,)
                        t3d = t3d @ R.T + t                          # (T, N, 3)
                    traj3d_cam_list.append(t3d)
                traj3d_sf = torch.stack(traj3d_cam_list, dim=0)      # (B, T, N, 3)

            # × traj3d inlier: filter points where any frame has |xyz| > threshold
            # Applied AFTER cam0 transform so thresholds are in metric cam0 space.
            # DynPose: 50m (real-world scenes), Synthetic: 50m (world→cam0 is rigid, same scale)
            TRAJ3D_ABS_THRESH = 50.0
            max_abs = traj3d_sf.abs().max(dim=1).values.max(dim=-1).values  # (B, N)
            conf_mask_sf = conf_mask_sf * (max_abs < TRAJ3D_ABS_THRESH).float().unsqueeze(1)  # (B, 1, N)

            # Reshape to spatial maps: (B, T, H*W, 3) → (B, 3, T, H, W)
            gt_traj3d = traj3d_sf.permute(0, 3, 1, 2).reshape(B_sf, 3, T_sf, H, W)
            gt_vis = conf_mask_sf.reshape(B_sf, T_sf, H, W)

            inputs["gt_traj3d"] = gt_traj3d.to(self.pipe.torch_dtype)
            inputs["gt_vis"] = gt_vis

            # NEW (isolated from delta loss): per-frame visibility target + loss mask
            # for the optional vis-pred head. Mask mirrors the delta conf_mask_sf
            # (including conf threshold for DynPose) so both losses train on the
            # same "trusted" pixels.
            if getattr(self, 'predict_vis', False):
                vis_target_sf = inputs['vis'][:, :pf, :].float()   # (B, T, N) per-frame
                vis_loss_mask_sf = torch.ones(B_sf, T_sf, N_sf, device=device)
                # frame-0 vis: only annotate at pixels where a track exists at frame 0
                vis_loss_mask_sf = vis_loss_mask_sf * (vis_target_sf[:, 0:1, :] > 0.5)
                # eraser: per-frame exclusion of artificially erased regions
                if 'eraser_vis' in inputs:
                    vis_loss_mask_sf = vis_loss_mask_sf * inputs['eraser_vis'][:, :T_sf, :].float()
                # fg_mask (Kubric): restrict to foreground
                if 'fg_mask' in inputs:
                    fg_vm = inputs['fg_mask'][:, 0, 0, :, :].reshape(B_sf, 1, -1).float()
                    vis_loss_mask_sf = vis_loss_mask_sf * (fg_vm > 0.5)
                # traj3d inlier (reuse already-computed max_abs)
                vis_loss_mask_sf = vis_loss_mask_sf * (max_abs < TRAJ3D_ABS_THRESH).float().unsqueeze(1)

                inputs['gt_vis_per_frame'] = vis_target_sf.reshape(B_sf, T_sf, H, W)
                inputs['vis_loss_mask'] = vis_loss_mask_sf.reshape(B_sf, T_sf, H, W)

            # Normalize P0(tj) GT with Pj(tj)'s normalization params, add depth-validity mask.
            if 'pj_norm_params' in inputs:
                pj_norm_params = inputs['pj_norm_params']
                diag_max_depth = self.diag_max_depth
                gt_norm_dc = []
                depth_valid_masks = []
                for b in range(B_sf):
                    gt_b = gt_traj3d[b].clone().float()  # (3, T, H, W) in camera_0

                    # P0(tj) depth validity: z must be in (0, diag_max_depth]
                    z_valid = (gt_b[2] > 0) & (gt_b[2] <= diag_max_depth)  # (T, H, W)
                    depth_valid_masks.append(z_valid)

                    # Z clamp for normalization (same as Pj(tj))
                    gt_b[2] = gt_b[2].clamp(min=1e-6, max=diag_max_depth)

                    mean_b, max_dist_b = pj_norm_params[b]
                    gt_norm_dc.append((gt_b - mean_b.view(3, 1, 1, 1)) / (max_dist_b + 1e-6))

                # Intersect existing mask with depth validity
                depth_valid = torch.stack(depth_valid_masks)  # (B, T, H, W)
                inputs['gt_vis'] = gt_vis * depth_valid.float()
                if getattr(self, 'predict_vis', False) and 'vis_loss_mask' in inputs:
                    inputs['vis_loss_mask'] = inputs['vis_loss_mask'] * depth_valid.float()

                # Replace gt_traj3d with pre-normalized version
                inputs['gt_traj3d'] = torch.stack(gt_norm_dc).to(self.pipe.torch_dtype)

                # pixel_delta: GT target = P0(tj)_norm - P0(t0)_norm
                if self.pixel_delta:
                    gt_dc = inputs['gt_traj3d']  # (B, 3, T, H, W) already normalized
                    inputs['gt_traj3d'] = gt_dc - gt_dc[:, :, 0:1, :, :]

            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss = self.pipe.training_loss(**models, **inputs)
            return loss

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.init(project="trackcraft3r", config=vars(args))

    # Build the synthetic dataset
    from diffsynth.trainers.synthetic_dataset import SyntheticDataset
    synth_configs = json.loads(args.synthetic_config)
    dataset = SyntheticDataset(
        dataset_configs=synth_configs,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        use_augmentation=getattr(args, 'synthetic_augmentation', False),
        depth_augmentation=getattr(args, 'depth_augmentation', False),
        eraser_augmentation=getattr(args, 'eraser_augmentation', False),
        repeat=args.dataset_repeat,
        kubric_fix=getattr(args, 'kubric_fix', False),
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        track_latent_length=args.track_latent_length,
        regression_timestep=getattr(args, 'regression_timestep', None),
        height=getattr(args, 'height', 480),
        width=getattr(args, 'width', 832),
        traj3d_coord_space=getattr(args, 'traj3d_coord_space', 'world'),
        point_norm_mode=getattr(args, 'point_norm_mode', 'mean_scale'),
        diag_max_depth=getattr(args, 'diag_max_depth', 80.0),
        vae_train_parts=getattr(args, 'vae_train_parts', 'decoder'),
        predict_vis=getattr(args, 'predict_vis', False),
        vis_loss_weight=getattr(args, 'vis_loss_weight', 1.0),
        vis_separate_decoder=getattr(args, 'vis_separate_decoder', False),
        pj_separate_encoder=getattr(args, 'pj_separate_encoder', False),
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    launch_training_task(dataset, model, model_logger, args=args, wandb=wandb)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.finish()
