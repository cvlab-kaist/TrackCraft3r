import os, torch, argparse, json, math, time
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
import numpy as np

# -----------------------------------------------------------------------------
# Monkey-patch: accelerate 1.10.1 + torchdata StatefulDataLoader in DDP mode
# raises KeyError('_sampler_iter_yielded') on the very first next() call
# because the underlying StatefulDataLoader's state_dict() is still "empty"
# (no batch yielded yet) and the adjust_state_dict_for_prefetch helper
# indexes that key directly. Replace the direct indexing with .get() so
# missing keys are treated as 0 (correct initial state).
# -----------------------------------------------------------------------------
def _patched_adjust_state_dict_for_prefetch(self):
    from accelerate.state import PartialState
    from accelerate.utils import DistributedType
    if PartialState().distributed_type == DistributedType.NO:
        return
    factor = PartialState().num_processes - 1
    if self.dl_state_dict.get("_sampler_iter_yielded", 0) > 0:
        self.dl_state_dict["_sampler_iter_yielded"] -= factor
    if self.dl_state_dict.get("_num_yielded", 0) > 0:
        self.dl_state_dict["_num_yielded"] -= factor
    _iss = self.dl_state_dict.get("_index_sampler_state")
    if _iss is not None and "samples_yielded" in _iss and _iss["samples_yielded"] > 0:
        _iss["samples_yielded"] -= self.batch_size * factor


from accelerate.data_loader import DataLoaderAdapter as _DLA
_DLA.adjust_state_dict_for_prefetch = _patched_adjust_state_dict_for_prefetch
del _DLA

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None, global_step=0):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.save_training_state(accelerator, global_step)


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


    def save_training_state(self, accelerator, global_step):
        """Save full training state including optimizer, scheduler, and RNG state."""
        state_dir = os.path.join(self.output_path, f"state-{global_step}")
        accelerator.save_state(state_dir)
        if accelerator.is_main_process:
            print(f"[Checkpoint] Saved full training state to {state_dir}")


def raw_batch_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [d[key] for d in batch]

        # Special handling for camera_stats (dict with 'mean' and 'max')
        if key == 'camera_stats':
            mean_centers = []
            max_dists = []
            for stats in values:
                mean = stats['mean']  # (1, 3) np.array or tensor
                if isinstance(mean, torch.Tensor):
                    mean_centers.append(mean.squeeze(0).float())
                else:
                    mean_centers.append(torch.from_numpy(np.asarray(mean)).squeeze(0).float())
                max_dists.append(float(stats['max']))
            collated[key] = {
                'mean': torch.stack(mean_centers, dim=0).unsqueeze(1),  # (B, 1, 3)
                'max': torch.tensor(max_dists).unsqueeze(1),            # (B, 1)
            }
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, 0)  # (B, ...)
        elif isinstance(values[0], list):
            # PIL image lists: keep as list of lists (can't stack)
            collated[key] = values  # [[frames_1], [frames_2], ...]
        elif isinstance(values[0], str):
            collated[key] = values  # [str_1, str_2, ...]
        else:
            collated[key] = values

    return collated


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
    wandb = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters

    lr_decay_steps = getattr(args, 'lr_decay_steps', None) if args is not None else None
    min_lr = getattr(args, 'min_lr', None) if args is not None else None
    batch_size = getattr(args, 'batch_size', 1) if args is not None else 1

    # Auto-scale lr_decay_steps for multi-GPU training
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    if lr_decay_steps is not None and num_processes > 1:
        lr_decay_steps_original = lr_decay_steps
        lr_decay_steps = lr_decay_steps * num_processes
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Multi-GPU] Scaled lr_decay_steps: {lr_decay_steps_original} -> {lr_decay_steps} (x{num_processes} GPUs)")

    # Seed for reproducibility
    import random
    seed = getattr(args, 'seed', 42) if args is not None else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Performance: TF32 for matmul/conv
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Separate param groups for VAE decoder (lower lr) vs everything else.
    # Include all VAE deep copies (pipe.vae, pipe.vae_vis, pipe.vae_pj) so any
    # unfrozen params across them use vae_lr rather than the DiT LoRA lr.
    vae_lr = getattr(args, 'vae_lr', None) if args is not None else None
    if vae_lr is not None and hasattr(model, 'pipe') and hasattr(model.pipe, 'vae'):
        vae_param_ids = set()
        for attr in ('vae', 'vae_vis', 'vae_pj'):
            module = getattr(model.pipe, attr, None)
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad:
                        vae_param_ids.add(id(p))
        group_vae = [p for p in model.parameters() if p.requires_grad and id(p) in vae_param_ids]
        group_other = [p for p in model.parameters() if p.requires_grad and id(p) not in vae_param_ids]
        param_groups = [
            {"params": group_other, "lr": learning_rate},
            {"params": group_vae, "lr": vae_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[optimizer] Separate param groups: other lr={learning_rate}, VAE lr={vae_lr}")
            print(f"  other params: {sum(p.numel() for p in group_other):,}")
            print(f"  VAE params:   {sum(p.numel() for p in group_vae):,} "
                  f"(vae + vae_vis + vae_pj, where present)")
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    if lr_decay_steps is not None and min_lr is not None:
        # Custom LambdaLR for cosine decay followed by constant LR
        def lr_lambda_main(step):
            if step < lr_decay_steps:
                progress = step / lr_decay_steps
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr / learning_rate + (1.0 - min_lr / learning_rate) * cosine_factor
            else:
                return min_lr / learning_rate

        # Build per-group lambda list (VAE group uses same decay ratio as main)
        num_groups = len(optimizer.param_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda_main] * num_groups)
        print(f"[scheduler] CosineAnnealingLR: {learning_rate} -> {min_lr} over {lr_decay_steps} steps, then constant at {min_lr}")
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    loader_kwargs = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=(num_workers > 0), persistent_workers=(num_workers > 0))
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=raw_batch_collate_fn, **loader_kwargs)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_stateful_dataloader=True),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters),
            InitProcessGroupKwargs(timeout=timedelta(minutes=30)),
        ]
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Pre-cache null T5 context on CPU before training starts (avoids T5 call under zombie-memory pressure during validation)
    if accelerator.is_main_process:
        _raw = accelerator.unwrap_model(model)
        _pipe = _raw.pipe
        _t5_dev = next(_pipe.dit.parameters()).device
        print(f"[pre-cache] Pre-encoding null T5 context on {_t5_dev} ...")
        try:
            with torch.no_grad():
                _null_ctx = _pipe.prompter.encode_prompt("", positive=True, device=_t5_dev)
            _raw._cached_null_context = _null_ctx.cpu()
            del _null_ctx
            torch.cuda.empty_cache()
            print(f"[pre-cache] null T5 context cached on CPU, shape={_raw._cached_null_context.shape}")
        except Exception as _e:
            print(f"[pre-cache] Warning: failed to pre-cache T5 context: {_e}. Will re-encode during validation.")

    # Resume from full training state if provided
    resume_from_state = getattr(args, 'resume_from_state', None) if args is not None else None
    global_step = 0
    if resume_from_state is not None and os.path.exists(resume_from_state):
        try:
            accelerator.load_state(resume_from_state)
        except ValueError as e:
            if "different number of parameter groups" in str(e) or "doesn't match the size of optimizer's group" in str(e):
                # Optimizer param groups changed (e.g. VAE encoder newly unfrozen).
                # accelerator.load_state() loads models BEFORE optimizers (see accelerate/checkpointing.py),
                # so the model weights are already loaded successfully when the optimizer error occurs.
                # Do NOT re-load model weights — the double load via raw load_state_dict can
                # subtly corrupt LoRA/DDP-wrapped weights. Just reset optimizer & scheduler.
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[Resume] Optimizer param-group mismatch — model weights already loaded, optimizer/scheduler reset.")
            else:
                raise
        except RuntimeError as e:
            # Missing keys in checkpoint (e.g. pipe.vae_vis / pipe.vae_pj just
            # added and not in old checkpoint). Retry with strict=False so the
            # extra modules stay at their initial (deep-copied) weights.
            if "Missing key" in str(e) and ("vae_vis" in str(e) or "vae_pj" in str(e)):
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[Resume] Checkpoint lacks vae_vis/vae_pj keys — reloading with "
                          f"strict=False; new modules stay at their deep-copied init weights.")
                try:
                    accelerator.load_state(resume_from_state, strict=False)
                except ValueError as e2:
                    if "different number of parameter groups" in str(e2) or "doesn't match the size of optimizer's group" in str(e2):
                        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            print(f"[Resume] Optimizer param-group mismatch after strict=False reload — resetting optimizer/scheduler.")
                    else:
                        raise
            else:
                raise
        # Extract step number from state directory name (e.g., "state-1000" -> 1000)
        # Strip trailing slashes to handle paths like "state-799/"
        try:
            global_step = int(os.path.basename(resume_from_state.rstrip('/')).split('-')[-1])
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"[Resume] Loaded full training state from {resume_from_state}, resuming from step {global_step}")
        except:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"[Resume] Loaded state from {resume_from_state}, but could not parse step number. Starting from step 0.")
    elif resume_from_state is not None:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Warning] Resume state path provided but not found: {resume_from_state}")

    # Sync model_logger.num_steps with global_step so checkpoint filenames continue correctly
    if global_step > 0:
        model_logger.num_steps = global_step
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Resume] model_logger.num_steps set to {global_step}")

    _is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0
    _t_iter_end = time.time()
    for epoch_id in range(num_epochs):
        progress_bar = tqdm(dataloader)
        for data in progress_bar:
            with accelerator.accumulate(model):
                _t_data = time.time() - _t_iter_end
                optimizer.zero_grad(set_to_none=True)
                _t0 = time.time()
                loss = model(data)
                torch.cuda.synchronize()
                _t_forward = time.time() - _t0
                _t0 = time.time()
                accelerator.backward(loss)
                torch.cuda.synchronize()
                _t_backward = time.time() - _t0

                # NOTE: NaN gradient protection
                # nan_to_num on loss keeps DDP sync alive, but backward through NaN activations
                # still produces NaN gradients (0 * NaN = NaN in IEEE 754).
                # If NaN grads detected, zero them out and skip optimizer step to protect weights.
                # NOTE: has_nan_grad must be all-reduced across ranks so all ranks agree on
                # whether to skip the optimizer step — otherwise DDP weight divergence occurs.
                has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break

                # Synchronize across all DDP ranks (max = logical OR)
                if accelerator.num_processes > 1:
                    import torch.distributed as dist
                    has_nan_tensor = torch.tensor(float(has_nan_grad), device=accelerator.device)
                    dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.MAX)
                    has_nan_grad = has_nan_tensor.item() > 0

                if has_nan_grad:
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"\nWarning: NaN/Inf gradients at step {global_step}, skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                else:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    _t0 = time.time()
                    optimizer.step()
                    torch.cuda.synchronize()
                    _t_opt = time.time() - _t0

                if _is_rank0 and global_step % 1 == 0:
                    _t_total = _t_data + _t_forward + _t_backward + _t_opt
                    print(f"\n[Timing step={global_step}] "
                          f"data={_t_data:.1f}s  forward={_t_forward:.1f}s  "
                          f"backward={_t_backward:.1f}s  optim={_t_opt:.1f}s  "
                          f"total={_t_total:.1f}s")

                model_logger.on_step_end(accelerator, model, save_steps, global_step=global_step)
                scheduler.step()

                # Calculate average loss across all GPUs first
                avg_loss = accelerator.gather(loss.detach()).mean().item()

                if wandb is not None and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    log_dict = {"train_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
                    pipe = accelerator.unwrap_model(model).pipe
                    if getattr(pipe, 'last_row_loss', None) is not None:
                        log_dict["train_loss/row_P0_tj"] = pipe.last_row_loss
                    if getattr(pipe, 'last_vis_bce', None) is not None:
                        log_dict["train_loss/vis_bce"] = pipe.last_vis_bce
                    wandb.log(log_dict)


                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                global_step += 1
                _t_iter_end = time.time()

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=True, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=True, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--resume_from_state", type=str, default=None, help="Path to a full training state directory to resume (includes optimizer, scheduler, step count). If provided, resumes complete training state.")
    parser.add_argument("--track_latent_length", type=int, default=None, help="Number of frames to use for track latent training. If None, use all frames.")
    parser.add_argument("--lr_decay_steps", type=int, default=None,
                        help="Number of steps for cosine LR decay. If None, use constant LR.")
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Minimum learning rate for cosine decay schedule.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size per GPU.")
    # Synthetic dataset arguments
    parser.add_argument("--synthetic_config", type=str, default="[]",
                        help='JSON list of synthetic dataset configs, e.g., [{"type":"kubric","raw_path":"...","track_path":"..."}]')
    parser.add_argument("--synthetic_augmentation", default=True, action="store_true",
                        help="Enable augmentation (random crop + color jitter + blur + grayscale + noise) for synthetic data.")
    parser.add_argument("--depth_augmentation", default=False, action="store_true",
                        help="Enable depth augmentation (scale+shift, blur, multi-res noise) to simulate estimated depth inaccuracy. "
                             "Only affects depth maps; Pj(tj) is recomputed from augmented depth on-the-fly.")
    parser.add_argument("--eraser_augmentation", default=False, action="store_true",
                        help="Enable eraser augmentation (TAPiP3D-style) for synthetic data. "
                             "Randomly erases rectangles in RGB+depth at frames t>0 and marks "
                             "affected tracked points as invisible via per-frame vis masking.")
    parser.add_argument("--kubric_fix", default=False, action="store_true",
                        help="Fix Kubric intrinsic bug: tracks.npz stores fy miscomputed as "
                             "(focal/sensor_width)*H instead of (focal/sensor_width)*W. "
                             "For square-pixel Kubric/Blender cameras fx should equal fy. "
                             "When enabled, sets intrinsics[:,1,1] = intrinsics[:,0,0] at load time. "
                             "Source bug: koo/MV-Kubric/extractor_utils.py:265.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--regression_timestep", type=int, default=-1,
                        help="scheduler.timesteps index for regression training+validation. "
                             "0=noisiest (default, same as original), -1=cleanest. "
                             "Training and validation always use the same index.")
    parser.add_argument("--traj3d_coord_space", type=str, default="camera_first",
                        choices=["world", "camera_first"],
                        help="Coordinate space for traj3d GT: 'world' (original world coords, default) or "
                             "'camera_first' (transform to frame-0 camera space before loss).")
    parser.add_argument("--point_norm_mode", type=str, default="mean_scale",
                        choices=["mean_scale", "max", "avg_dis"],
                        help="Point map normalization before loss: "
                             "'mean_scale' (subtract centroid, divide by mean dist, default), "
                             "'max' (subtract centroid, divide by max abs → [-1, 1]), or "
                             "'avg_dis' (no centering, divide by mean dist from origin).")
    parser.add_argument("--diag_max_depth", type=float, default=80.0,
                        help="Maximum depth (meters) used to clamp the input pointmap.")
    parser.add_argument("--predict_vis", action="store_true", default=True,
                        help="Add a 1-channel per-frame visibility prediction head to the VAE "
                             "decoder (parallel to the 3-ch xyz head, zero-initialized). Target = "
                             "per-frame vis from dataset (1=visible, 0=occluded). Trained with "
                             "balanced BCE on valid-annotation pixels only. Does NOT modify the "
                             "existing delta loss / vis-mask pathway.")
    parser.add_argument("--vis_loss_weight", type=float, default=0.1,
                        help="Weight for the per-frame visibility balanced-BCE loss. Only used "
                             "when --predict_vis is set.")
    parser.add_argument("--vis_separate_decoder", action="store_true", default=False,
                        help="Use a SEPARATE VAE decoder for visibility prediction (deep-copied "
                             "from the shared VAE at init). Recommended when unfreezing VAE so "
                             "each decoder can specialize (xyz vs vis) without cross-task "
                             "interference. Requires --predict_vis. The second decoder is stored "
                             "under pipe.vae_vis and is saved/loaded as part of the checkpoint.")
    parser.add_argument("--pj_separate_encoder", action="store_true", default=False,
                        help="Use a SEPARATE VAE encoder for pointmap (Pj(tj)) encoding, "
                             "deep-copied from the shared VAE at init. Without this flag the "
                             "RGB and pointmap streams share pipe.vae.encoder. With this flag, "
                             "pointmap encoding is routed through pipe.vae_pj.encode so the two "
                             "encoders can specialize. vae_train_parts applies to pipe.vae_pj "
                             "the same way it does to pipe.vae. The encoder (and decoder, if "
                             "listed) of pipe.vae_pj are saved/loaded as part of the checkpoint.")
    parser.add_argument("--vae_lr", type=float, default=None,
                        help="Separate learning rate for VAE encoder/decoder. If None, uses --learning_rate. "
                             "Recommended: 1e-5 ~ 1e-6 (10-100x lower than LoRA lr) to preserve "
                             "pretrained spatial structure while adapting to 3D output.")
    parser.add_argument("--vae_train_parts", type=str, default="decoder",
                        help="Which VAE parts to train: 'decoder', 'encoder', or 'encoder,decoder' for both. "
                             "Only effective when --trainable_models includes 'vae'.")
    # Iteration tracking mode
    return parser



