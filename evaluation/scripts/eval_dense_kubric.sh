#!/bin/bash
# Dense 3D tracking evaluation on the kubric_da3 / kubric_vipe variants.
#
# Each NPZ contains: predicted depth+camera (DA3 or ViPE), GT extrinsics for the
# GT-camera metric (currently unused in this script), dense GT tracks, GT
# intrinsics. The model is run dense (per-pixel of the prediction grid) and
# evaluated against the dense GT tracks.
#
# Usage:
#   bash evaluation/scripts/eval_dense_kubric.sh
#   bash evaluation/scripts/eval_dense_kubric.sh --checkpoint_path <path>

export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-./checkpoints/wan_models}
export MODELSCOPE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

CKPT=${CKPT:-./checkpoints/trackcraft3r/model.safetensors}
DATA_ROOT=${DATA_ROOT:-./eval_dataset}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_results/kubric}

NUM_FRAMES=24
FRAME_STRIDE=1
NUM_SAMPLES=50
EVAL_STRIDE=1
HEIGHT=480
WIDTH=832
EVAL_HEIGHT=480
EVAL_WIDTH=832

LORA_RANK=1024
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"
REGRESSION_TIMESTEP=-1
TRACK_LATENT_LENGTH=12
RESIZE_MODE="stretch"
DIAG_MAX_DEPTH=80.0
PJ_NORM_PERCENTILE_LO=2.0
PJ_NORM_PERCENTILE_HI=98.0

DATA_TYPES_OVERRIDE=""
GPUS=(0 1)

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)       CKPT="$2"; shift 2 ;;
        --data_root)             DATA_ROOT="$2"; shift 2 ;;
        --output_dir)            OUTPUT_DIR="$2"; shift 2 ;;
        --num_frames)            NUM_FRAMES="$2"; shift 2 ;;
        --num_samples)           NUM_SAMPLES="$2"; shift 2 ;;
        --eval_stride)           EVAL_STRIDE="$2"; shift 2 ;;
        --data_types)            DATA_TYPES_OVERRIDE="$2"; shift 2 ;;
        --gpus)                  GPUS=($2); shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "${CKPT}" ]]; then
    echo "Usage: bash $0 --checkpoint_path <path>"; exit 1
fi

if [[ -n "${DATA_TYPES_OVERRIDE}" ]]; then
    DATA_TYPES=(${DATA_TYPES_OVERRIDE})
else
    DATA_TYPES=("kubric_da3" "kubric_vipe")
fi
NUM_GPUS=${#GPUS[@]}

echo "============================================"
echo " Dense Kubric Eval"
echo " Checkpoint: ${CKPT}"
echo " data_types: ${DATA_TYPES[*]}"
echo " GPUs: ${GPUS[*]}"
echo "============================================"

for ((i=0; i<${#DATA_TYPES[@]}; i++)); do
    dt="${DATA_TYPES[$i]}"
    gpu="${GPUS[$((i % NUM_GPUS))]}"
    out_dir="${OUTPUT_DIR}/${dt}"
    echo "[${dt}] GPU=${gpu} -> ${out_dir}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m evaluation.eval_dense_kubric \
        --checkpoint_path "${CKPT}" \
        --data_root "${DATA_ROOT}/${dt}" \
        --data_type "${dt}" \
        --num_frames ${NUM_FRAMES} \
        --frame_stride ${FRAME_STRIDE} \
        --num_samples ${NUM_SAMPLES} \
        --eval_stride ${EVAL_STRIDE} \
        --eval_height ${EVAL_HEIGHT} \
        --eval_width ${EVAL_WIDTH} \
        --height ${HEIGHT} --width ${WIDTH} \
        --lora_rank ${LORA_RANK} \
        --lora_target_modules "${LORA_TARGET_MODULES}" \
        --regression_timestep ${REGRESSION_TIMESTEP} \
        --track_latent_length ${TRACK_LATENT_LENGTH} \
        --resize_mode ${RESIZE_MODE} \
        --diag_max_depth ${DIAG_MAX_DEPTH} \
        --pj_norm_percentile_lo ${PJ_NORM_PERCENTILE_LO} \
        --pj_norm_percentile_hi ${PJ_NORM_PERCENTILE_HI} \
        --output_dir "${out_dir}" &
    if (( (i+1) % NUM_GPUS == 0 )); then wait; fi
done
wait

echo "============================================"
echo " All dense kubric evaluations complete!"
echo "============================================"
