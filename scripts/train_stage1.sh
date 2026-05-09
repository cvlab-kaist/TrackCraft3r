#!/bin/bash
# Stage 1: train DiT LoRA + I/O projections; VAE is frozen.

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export UCX_HANDLE_ERRORS=none
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-./checkpoints/wan_models}
export MODELSCOPE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=4

# ---- Paths to datasets (edit before running) ----
KUBRIC_ROOT=${KUBRIC_ROOT:-/path/to/kubric}
DYNAMIC_REPLICA_ROOT=${DYNAMIC_REPLICA_ROOT:-/path/to/dynamic_replica}
POINTODYSSEY_ROOT=${POINTODYSSEY_ROOT:-/path/to/point_odyssey/train}
TARTANAIR_ROOT=${TARTANAIR_ROOT:-/path/to/tartanair}

OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/stage1}

accelerate launch --num_processes 8 examples/wanvideo/model_training/train.py \
  --height 480 \
  --width 832 \
  --dataset_repeat 10 \
  --save_steps 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --min_lr 3e-5 \
  --lr_decay_steps 10000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --trainable_models "" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 1024 \
  --track_latent_length 12 \
  --batch_size 10 \
  --dataset_num_workers 8 \
  --synthetic_config "[
    {\"type\": \"kubric\",
      \"root_path\": \"$KUBRIC_ROOT\",
      \"S\": 12, \"strides\": [3,4,5,6,7], \"start_from_zero\": true},
    {\"type\": \"dynamic_replica\",
      \"path\": \"$DYNAMIC_REPLICA_ROOT\",
      \"S\": 12, \"strides\": [5,6,7,8,9], \"start_from_zero\": false},
    {\"type\": \"pointodyssey\",
      \"path\": \"$POINTODYSSEY_ROOT\",
      \"S\": 12, \"strides\": [2,3,4,5,6], \"start_from_zero\": false},
    {\"type\": \"tartanair\",
      \"path\": \"$TARTANAIR_ROOT\",
      \"S\": 12, \"strides\": [1,2,3], \"start_from_zero\": false}
  ]" \
  --kubric_fix
