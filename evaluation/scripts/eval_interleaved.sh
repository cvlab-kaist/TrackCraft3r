#!/bin/bash
# Interleaved-stride evaluation on the four WorldTrack mini benchmarks
# (both DA3 and ViPE depth+camera variants).
#
# Each sequence is split into `interleave_stride` interleaved sub-sequences,
# each fed to the model as a single 12-frame clip with a shared frame-0
# anchor. Predictions are then merged (overlapping frames averaged) and
# metrics are computed on the first --eval_frames frames.
#
# Usage:
#   bash evaluation/scripts/eval_interleaved.sh
#   bash evaluation/scripts/eval_interleaved.sh --checkpoint_path <path>

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
OUTPUT_DIR=${OUTPUT_DIR:-./eval_results/interleaved}

EVAL_FRAMES_LIST=(84)
MODEL_FRAMES=12
NUM_SAMPLES=50
HEIGHT=480
WIDTH=832

LORA_RANK=1024
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"
REGRESSION_TIMESTEP=-1
TRACK_LATENT_LENGTH=12
RESIZE_MODE="stretch"
DIAG_MAX_DEPTH=80.0
PJ_NORM_PERCENTILE_LO=2.0
PJ_NORM_PERCENTILE_HI=98.0
SAVE_PREDICTIONS="false"
SAVE_DENSE="false"

DATA_TYPES_OVERRIDE=""
GPUS=(0 1 2 3)

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)       CKPT="$2"; shift 2 ;;
        --data_root)             DATA_ROOT="$2"; shift 2 ;;
        --output_dir)            OUTPUT_DIR="$2"; shift 2 ;;
        --eval_frames_list)      EVAL_FRAMES_LIST=($2); shift 2 ;;
        --num_samples)           NUM_SAMPLES="$2"; shift 2 ;;
        --height)                HEIGHT="$2"; shift 2 ;;
        --width)                 WIDTH="$2"; shift 2 ;;
        --resize_mode)           RESIZE_MODE="$2"; shift 2 ;;
        --lora_rank)             LORA_RANK="$2"; shift 2 ;;
        --diag_max_depth)        DIAG_MAX_DEPTH="$2"; shift 2 ;;
        --pj_norm_percentile_lo) PJ_NORM_PERCENTILE_LO="$2"; shift 2 ;;
        --pj_norm_percentile_hi) PJ_NORM_PERCENTILE_HI="$2"; shift 2 ;;
        --save_predictions)      SAVE_PREDICTIONS="true"; shift ;;
        --no-save_predictions)   SAVE_PREDICTIONS="false"; shift ;;
        --save_dense)            SAVE_DENSE="true"; shift ;;
        --no-save_dense)         SAVE_DENSE="false"; shift ;;
        --data_types)            DATA_TYPES_OVERRIDE="$2"; shift 2 ;;
        --gpus)                  GPUS=($2); shift 2 ;;
        *) shift ;;
    esac
done

if [[ "${SAVE_PREDICTIONS}" == "true" ]]; then
    SAVE_PRED_FLAG="--save_predictions"
else
    SAVE_PRED_FLAG="--no-save_predictions"
fi
if [[ "${SAVE_DENSE}" == "true" ]]; then
    SAVE_DENSE_FLAG="--save_dense"
else
    SAVE_DENSE_FLAG="--no-save_dense"
fi

if [[ -z "${CKPT}" ]]; then
    echo "Usage: bash $0 --checkpoint_path <path>"
    exit 1
fi

if [[ -n "${DATA_TYPES_OVERRIDE}" ]]; then
    DATA_TYPES=(${DATA_TYPES_OVERRIDE})
else
    DATA_TYPES=(
        "ds_mini_da3"  "pstudio_mini_da3"  "adt_mini_da3"  "po_mini_da3"
        "ds_mini_vipe" "pstudio_mini_vipe" "adt_mini_vipe" "po_mini_vipe"
    )
fi
NUM_GPUS=${#GPUS[@]}

# Compute (eval_frames, stride, num_frames) triples from EVAL_FRAMES_LIST.
TRIPLES=()
for EVAL in "${EVAL_FRAMES_LIST[@]}"; do
    STRIDE=$(( (EVAL - 1 + MODEL_FRAMES - 2) / (MODEL_FRAMES - 1) ))
    (( STRIDE < 1 )) && STRIDE=1
    NUM=$(( STRIDE * (MODEL_FRAMES - 1) + 1 ))
    TRIPLES+=("${EVAL}:${STRIDE}:${NUM}")
done

echo "============================================"
echo " Interleaved Stride Eval"
echo " Checkpoint: ${CKPT}"
echo " Resolution: ${HEIGHT}x${WIDTH}  model_frames=${MODEL_FRAMES} (fixed)"
printf "   %-5s | %-6s | %-10s | %s\n" "EVAL" "stride" "num_frames" "output_suffix"
for triple in "${TRIPLES[@]}"; do
    IFS=':' read -r EVAL STRIDE NUM <<< "${triple}"
    printf "   %-5d | %-6d | %-10d | _eval%d_s%d_f%d\n" "$EVAL" "$STRIDE" "$NUM" "$EVAL" "$STRIDE" "$NUM"
done
echo " GPUs: ${GPUS[*]}  (${NUM_GPUS} parallel)"
echo "============================================"

JOBS=()
for triple in "${TRIPLES[@]}"; do
    for dt in "${DATA_TYPES[@]}"; do
        JOBS+=("${triple}:${dt}")
    done
done
echo "Total jobs: ${#JOBS[@]}"

for ((batch=0; batch<${#JOBS[@]}; batch+=NUM_GPUS)); do
    for ((j=0; j<NUM_GPUS && batch+j<${#JOBS[@]}; j++)); do
        i=$((batch + j))
        IFS=':' read -r EVAL STRIDE NUM dt <<< "${JOBS[$i]}"
        gpu="${GPUS[$j]}"
        RUN_OUT="${OUTPUT_DIR}_eval${EVAL}_s${STRIDE}_f${NUM}"
        echo "[$((i+1))/${#JOBS[@]}] ${dt} on GPU ${gpu}  (eval=${EVAL}, stride=${STRIDE}, num=${NUM})"
        CUDA_VISIBLE_DEVICES=${gpu} python -m evaluation.eval_worldtrack \
            --model_type wan_scene_flow \
            --checkpoint_path "${CKPT}" \
            --diagonal_condition_row \
            --pixel_delta \
            --pj_norm_inlier \
            --predict_vis \
            --vis_separate_decoder \
            --pj_separate_encoder \
            --lora_rank ${LORA_RANK} \
            --lora_target_modules "${LORA_TARGET_MODULES}" \
            --data_root "${DATA_ROOT}" \
            --data_types ${dt} \
            --num_frames ${NUM} \
            --num_samples ${NUM_SAMPLES} \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            --output_dir "${RUN_OUT}" \
            --regression_timestep ${REGRESSION_TIMESTEP} \
            --track_latent_length ${TRACK_LATENT_LENGTH} \
            --resize_mode ${RESIZE_MODE} \
            --diag_max_depth ${DIAG_MAX_DEPTH} \
            --pj_norm_percentile_lo ${PJ_NORM_PERCENTILE_LO} \
            --pj_norm_percentile_hi ${PJ_NORM_PERCENTILE_HI} \
            --interleaved_stride \
            --interleave_stride ${STRIDE} \
            --eval_frames ${EVAL} \
            ${SAVE_PRED_FLAG} \
            ${SAVE_DENSE_FLAG} &
    done
    wait
done

echo "============================================"
echo " All evaluations complete!"
echo "============================================"
