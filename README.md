# TrackCraft3R: Repurposing Video Diffusion Transformers for Dense 3D Tracking

[Paper (TBD)]() &nbsp;|&nbsp; [arXiv (TBD)]() &nbsp;|&nbsp; [Project Page](https://cvlab-kaist.github.io/TrackCraft3r)

This repository contains the official training code for **TrackCraft3R**, the first method that repurposes a pre-trained video diffusion transformer (Wan2.1-T2V-1.3B) as a single-pass dense 3D tracker. Given a monocular video together with its predicted depth and camera, TrackCraft3R predicts dense 3D trajectories in a single forward pass.

---

## 1. Environment

We tested training on **8 × NVIDIA H200 (141 GB)** GPUs with CUDA 12.1, Python 3.10, and PyTorch 2.4.

```bash
# 1. Create a fresh conda env
conda create -n trackcraft3r python=3.10 -y
conda activate trackcraft3r

# 2. Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest of the dependencies
pip install -e .
pip install -r requirements.txt
pip install accelerate huggingface_hub wandb imageio[ffmpeg]
```

Configure `accelerate` once:

```bash
accelerate config   # or: accelerate config default
```

---

## 2. Pre-trained Weights

### 2.1 Wan2.1-T2V-1.3B base model

The training code initializes the DiT, VAE, and T5 text encoder from the public Wan2.1-T2V-1.3B checkpoint. Download it once with:

```bash
python scripts/download_wan_1.3B.py --target ./checkpoints/wan_models
```

This pulls only the files we need (`diffusion_pytorch_model*.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth`) into `./checkpoints/wan_models/Wan-AI/Wan2.1-T2V-1.3B/`. The training scripts read this location through the `MODELSCOPE_CACHE` environment variable (default `./checkpoints/wan_models`).

### 2.2 TrackCraft3R checkpoint

Trained weights are released on the Hugging Face Hub at
[`trackcraft3r/checkpoint`](https://huggingface.co/trackcraft3r/checkpoint):

```bash
huggingface-cli download trackcraft3r/checkpoint --local-dir ./checkpoints/trackcraft3r
# → ./checkpoints/trackcraft3r/model.safetensors
```

---

## 3. Training Datasets

We train on four synthetic datasets: [Kubric](https://github.com/google-research/kubric), Dynamic Replica and PointOdyssey (downloaded via the scripts in [St4RTrack](https://github.com/HavenFeng/St4RTrack)), and [TartanAir](https://theairlab.org/tartanair-dataset/). For Kubric we render 6K sequences (480×832, 81 frames).

Once downloaded/rendered, set the four environment variables that the training scripts read:

```bash
export KUBRIC_ROOT=/path/to/kubric
export DYNAMIC_REPLICA_ROOT=/path/to/dynamic_replica
export POINTODYSSEY_ROOT=/path/to/point_odyssey/train
export TARTANAIR_ROOT=/path/to/tartanair
```

See `diffsynth/trainers/synthetic_dataset.py` for the exact directory structure each loader expects.

---

## 4. Training

Training proceeds in two stages.

### 4.1 Stage 1: DiT LoRA + I/O projections

Train the DiT with LoRA together with the input/output projection layers. The VAE encoders/decoders are frozen.

```bash
bash scripts/train_stage1.sh
```

Checkpoints are saved every 100 steps to `./checkpoints/stage1/`.

### 4.2 Stage 2: DiT LoRA + I/O projections + VAE

Continue from a Stage-1 checkpoint and additionally unfreeze the VAE encoder/decoder. The pointmap encoder and visibility decoder are deep-copied to give two independent encoders (RGB / pointmap) and two independent decoders (residual track / visibility), all trained jointly with the DiT LoRA and the input/output projection layers.

```bash
# Pick the Stage-1 state directory you want to resume from
export RESUME_FROM=./checkpoints/stage1/state-XXXX

bash scripts/train_stage2.sh
```

Checkpoints are saved to `./checkpoints/stage2/`.

### 4.3 W&B logging

Run `wandb login` once before training. To skip W&B entirely, `export WANDB_MODE=disabled`.

---

## 5. Evaluation

We provide two evaluation scripts that reproduce the numbers reported in
the paper: an **interleaved eval** for long-video inference and a
**stride eval** for large-motion inference. See
[`evaluation/README.md`](evaluation/README.md) for details.

```bash
# 1) Download eval dataset and checkpoint
huggingface-cli download trackcraft3r/trackcraft3r-eval --repo-type dataset --local-dir ./eval_dataset
huggingface-cli download trackcraft3r/checkpoint --local-dir ./checkpoints/trackcraft3r

# 2) Run
bash evaluation/scripts/eval_interleaved.sh \
    --checkpoint_path ./checkpoints/trackcraft3r/model.safetensors \
    --data_root ./eval_dataset --output_dir ./eval_results/interleaved

bash evaluation/scripts/eval_stride.sh \
    --checkpoint_path ./checkpoints/trackcraft3r/model.safetensors \
    --data_root ./eval_dataset --output_dir ./eval_results/stride
```

---

## 6. Run on your own video

End-to-end pipeline:

```
your_video → preprocess (DA3 or ViPE) → build_user_npz → inference → visualize
```

The walkthrough below uses the included sample
[`assets/example/breakdance.mp4`](assets/example/breakdance.mp4).


### 6.1 Extract depth + camera

You can use either DA3 or ViPE. Both produce z-depth + per-frame
camera and feed into the same downstream NPZ. Each is installed
in its own conda env to avoid clashing with `trackcraft3r`.

**Option A: Depth-Anything-V3** ([repo](https://github.com/ByteDance-Seed/depth-anything-3)):

```bash
# 1. Set up a dedicated env for DA3.
#    Pinning torch + xformers up-front prevents `pip install -e .` from
#    pulling the latest torch (which may not match your CUDA driver).
conda create -n da3 python=3.10 -y
conda activate da3
pip install torch==2.5.1 torchvision==0.20.1 xformers==0.0.28.post3 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Clone + install DA3
git clone https://github.com/ByteDance-Seed/depth-anything-3 ./Depth-Anything-V3
cd ./Depth-Anything-V3 && pip install -e . && cd ..

# 3. Run on the example video (use --frame_dir for an image directory)
python scripts/preprocess_da3.py \
    --video_path ./assets/example/breakdance.mp4 \
    --output_dir ./preproc/breakdance_da3/ \
    --da3_root ./Depth-Anything-V3 \
    --model_name "depth-anything/DA3NESTED-GIANT-LARGE"

conda deactivate
```

Writes `depth.npy` (T,H,W float32 z-depth), `extrinsics.npy`
(T,4,4 **W2C**), `intrinsics.npy` (T,3,3, rescaled to original image
res).

**Option B: ViPE** ([repo](https://github.com/nv-tlabs/vipe)):

```bash
# 1. Set up a dedicated env for ViPE (follow ViPE's README for the exact
#    torch/CUDA versions it expects)
conda create -n vipe python=3.10 -y
conda activate vipe

# 2. Clone + install ViPE
git clone https://github.com/nv-tlabs/vipe ./ViPE
cd ./ViPE && pip install -e . && cd ..

# 3. Run on the example video
python scripts/preprocess_vipe.py \
    --video_path ./assets/example/breakdance.mp4 \
    --output_dir ./preproc/breakdance_vipe/ \
    --vipe_root ./ViPE

conda deactivate
```

Writes `depth.npy` (z-depth), `extrinsics.npy` (T,4,4 **C2W** per
`vipe/utils/io.py "cam2world matrices"`), `intrinsics.npy`. ViPE's
intrinsics are constant across frames in a single clip.

After §6.1, `conda activate trackcraft3r` to run §6.2 onwards.

### 6.2 Build the TrackCraft3R-format NPZ

DA3 path:
```bash
python scripts/build_user_npz.py \
    --video_path     ./assets/example/breakdance.mp4 \
    --depth_npy      ./preproc/breakdance_da3/depth.npy \
    --extrinsics_npy ./preproc/breakdance_da3/extrinsics.npy \
    --intrinsics_npy ./preproc/breakdance_da3/intrinsics.npy \
    --depth_convention z --extrinsics_convention w2c \
    --output_npz ./breakdance_user.npz
```

ViPE path:
```bash
python scripts/build_user_npz.py \
    --video_path     ./assets/example/breakdance.mp4 \
    --depth_npy      ./preproc/breakdance_vipe/depth.npy \
    --extrinsics_npy ./preproc/breakdance_vipe/extrinsics.npy \
    --intrinsics_npy ./preproc/breakdance_vipe/intrinsics.npy \
    --depth_convention z --extrinsics_convention c2w \
    --output_npz ./breakdance_user.npz
```

The output NPZ contains:

* `images_jpeg_bytes` — JPEG-encoded RGB frames
* `depth_map` (T, H, W) — z-depth from DA3 / ViPE
* `extrinsics_w2c` (T, 4, 4) — frame-0-normalized world-to-camera
* `fx_fy_cx_cy` (4,) — predicted intrinsics from DA3 / ViPE

### 6.3 Run inference + save the dense prediction

`--num_frames` × `--frame_stride` decides which frames the model sees. The NPZ from §6.2
keeps all frames so you can re-run inference at different settings
without re-building.

```bash
MODELSCOPE_CACHE=./checkpoints/wan_models \
python scripts/inference_user_video.py \
    --checkpoint_path ./checkpoints/trackcraft3r/model.safetensors \
    --input_npz  ./breakdance_user.npz \
    --output_npz ./breakdance_dense.npz \
    --num_frames 12 --frame_stride 5
```

Saves `track_map` (T,H,W,3) per-pixel 3D tracks in frame-0 cam space (used
for the overlaid track trails), `recon_map` (T,H,W,3) per-frame depth
back-projection in the same frame-0 cam space (used for the per-frame RGB
point cloud), and `rgb` (T,H,W,3) for point-cloud coloring.

### 6.4 Visualize with Viser

```bash
python scripts/visualize_dense.py --dense_npz ./breakdance_dense.npz --port 8080
```

---

## 7. Acknowledgements

- [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1): base video backbone
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): training pipeline framework
- [St4RTrack](https://github.com/HavenFeng/St4RTrack): evaluation code
- [Any4D](https://github.com/Any-4D/Any4D): visualization code

