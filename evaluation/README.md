# TrackCraft3R Evaluation

Evaluation code for **TrackCraft3R**:
* Sparse 3D tracking on the four mini WorldTrack benchmarks
  (ADT, PO, PStudio, DS).
* Dense 3D tracking on the Kubric test split.

Each dataset NPZ contains predicted depth (`depth_map`) and camera
(`extrinsics_w2c`) from ViPE or DA3. GT trajectories (`tracks_XYZ` /
`world_coords`) and GT camera (`extrinsics_w2c_gt`) are also included
to compute the final metric. The metric protocol follows
[St4RTrack](https://github.com/HavenFeng/St4RTrack).

## 1. Evaluation Data

Download the evaluation dataset from the Hugging Face Hub and point the
scripts at it:

```bash
huggingface-cli download trackcraft3r/trackcraft3r-eval --repo-type dataset --local-dir ./eval_dataset
```

The resulting layout is described in [`../eval_dataset/README.md`](../eval_dataset/README.md).

## 2. Checkpoint

Download the released TrackCraft3R checkpoint from the Hugging Face Hub:

```bash
huggingface-cli download trackcraft3r/checkpoint --local-dir ./checkpoints/trackcraft3r
```

You should end up with `./checkpoints/trackcraft3r/model.safetensors`.

The Wan2.1-T2V-1.3B base checkpoint (DiT + VAE + T5) is also required;
see the top-level `README.md` for the download command.

## 3. Run

Three evaluation modes are provided.

### 3.1 Interleaved eval (long-video inference)

The model always processes 12 frames per run. To cover a longer span
(default 84 frames), `eval_interleaved.sh` slices the video into
interleaved 12-frame sub-sequences sharing a frame-0 anchor and runs
each through the model.

```bash
bash evaluation/scripts/eval_interleaved.sh \
    --checkpoint_path ./checkpoints/trackcraft3r/model.safetensors \
    --data_root      ./eval_dataset \
    --output_dir     ./eval_results/interleaved
```

### 3.2 Stride eval (large-motion inference)

Fixes `num_frames=12` and sweeps the temporal spacing
(`stride ∈ {3,5,7,9,11}`). Larger stride indicates larger inter-frame
motion.

```bash
bash evaluation/scripts/eval_stride.sh \
    --checkpoint_path ./checkpoints/trackcraft3r/model.safetensors \
    --data_root      ./eval_dataset \
    --output_dir     ./eval_results/stride
```

Both scripts also accept `--data_types "<type1> <type2> ..."` to
evaluate any of the evaluation datasets: 4 DA3 variants
(`adt_mini_da3`, `po_mini_da3`, `pstudio_mini_da3`, `ds_mini_da3`) and
4 ViPE variants (`adt_mini_vipe`, `po_mini_vipe`, `pstudio_mini_vipe`,
`ds_mini_vipe`).

### 3.3 Dense kubric eval

Dense 3D tracking on the kubric 50 test split (`kubric_da3` /
`kubric_vipe`).

```bash
bash evaluation/scripts/eval_dense_kubric.sh \
    --checkpoint_path ./checkpoints/trackcraft3r/model.safetensors \
    --data_root      ./eval_dataset \
    --output_dir     ./eval_results/kubric
```

### 3.4 GPUs

Both scripts accept `--gpus "0 1 2 3"` (default). Jobs are dispatched
round-robin across the supplied GPU IDs.

## 4. Output

Each per-dataset run writes (under the run's `--output_dir`):

* `track_eval_<data_type>.txt`: final aggregated metrics (TAPVid3D
  AJ / OC_ACC / pts at multiple visibility thresholds, computed via
  Sim3 alignment between predictions and GT trajectories).
* `per_video_<data_type>.json`: per-sequence metric values.

## 5. Acknowledgements

* [St4RTrack](https://github.com/HavenFeng/St4RTrack): evaluation protocol,
  Sim(3) alignment + TAPVid3D metric utilities reused under
  `dust3r_eval_utils.py`.
