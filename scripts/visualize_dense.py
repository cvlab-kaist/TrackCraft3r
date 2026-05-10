"""Any4D-style 3D visualization of TrackCraft3R dense predictions via Viser.

Reads a `dense.npz` saved by `inference_user_video.py` and shows, all at once
in frame-0 cam space:
  - the frame-0 RGB point cloud (anchor, always visible)
  - the frame-t RGB point cloud (slider scrubs t; built from per-frame depth
    back-projection `recon_map[t]`, so camera panning reveals new scene content)
  - dynamic-pixel trajectories drawn as growing polylines [0..t]
    (built from the model's frame-0-anchored prediction `track_map`)

The dynamic-pixel selection (percentile + max-tracks) can be retuned live
from the GUI; trajectory polylines are rebuilt in place.

Usage:
    python scripts/visualize_dense.py --dense_npz my_video_dense.npz --port 8080
"""

import argparse
import time

import matplotlib
import matplotlib.colors
import numpy as np
import viser


def _edge_mask(depth_hw: np.ndarray, rtol: float = 0.05) -> np.ndarray:
    """Pixels at depth discontinuities (foreground/background bleed) → True."""
    safe_z = np.maximum(depth_hw, 1e-6)
    dx = np.abs(np.diff(depth_hw, axis=1, append=depth_hw[:, -1:])) / safe_z
    dy = np.abs(np.diff(depth_hw, axis=0, append=depth_hw[-1:, :])) / safe_z
    edge = (dx > rtol) | (dy > rtol)
    try:
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(edge, iterations=1)
    except ImportError:
        pass
    return edge


def _cc_clean(mask_hw: np.ndarray, min_cc_frac: float = 2e-5) -> np.ndarray:
    """Drop tiny connected components (isolated noise spikes)."""
    try:
        from scipy.ndimage import label as cc_label
    except ImportError:
        return mask_hw
    cc, n_cc = cc_label(mask_hw)
    if n_cc == 0:
        return mask_hw
    counts = np.bincount(cc.ravel())
    counts[0] = 0
    H, W = mask_hw.shape
    min_cc = max(20, int(H * W * min_cc_frac))
    return np.isin(cc, np.where(counts >= min_cc)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense_npz", type=str, required=True,
                   help="Path to a dense.npz with keys track_map + recon_map + rgb.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--downsample", type=int, default=1,
                   help="Spatial downsample for the per-frame point clouds.")
    p.add_argument("--traj_downsample", type=int, default=4,
                   help="Grid stride for picking trajectory candidates.")
    p.add_argument("--dynamic_percentile", type=float, default=90.5,
                   help="Show trajectories above this motion percentile.")
    p.add_argument("--max_tracks", type=int, default=2000,
                   help="Cap drawn trajectories.")
    p.add_argument("--traj_color", choices=["time", "query"], default="query",
                   help="'query' = each track keeps one color (HSV by image-y); "
                        "'time' = viridis along time, segments change color.")
    p.add_argument("--delta_zero", type=float, default=0.05,
                   help="Zero out per-pixel trajectory deltas smaller than this (metres). "
                        "Removes VAE/diffusion wobble on static pixels. 0 to disable.")
    args = p.parse_args()

    d = np.load(args.dense_npz)
    track_map = d["track_map"].astype(np.float32)   # (T, H, W, 3)  model tracks
    recon_map = d["recon_map"].astype(np.float32)   # (T, H, W, 3)  depth back-proj
    rgb       = d["rgb"]                            # (T, H, W, 3)  uint8
    T, H, W, _ = track_map.shape
    print(f"  loaded dense: T={T} H={H} W={W}")

    # Edge filter from frame-0 depth (z is the third coord in cam-0 space).
    edge0 = _edge_mask(recon_map[0, :, :, 2])
    keep_full = ~edge0
    print(f"  edge filter: {edge0.sum()}/{H*W} pixels masked "
          f"({100 * edge0.mean():.1f}%)")

    # Per-pixel peak displacement from frame 0 — drives dynamic selection.
    motion = np.linalg.norm(track_map - track_map[0:1], axis=-1).max(axis=0)  # (H, W)

    # Delta-zero: remove sub-threshold wobble from trajectory positions.
    if args.delta_zero > 0:
        delta = track_map - track_map[0:1]                        # (T, H, W, 3)
        mag   = np.linalg.norm(delta, axis=-1, keepdims=True)     # (T, H, W, 1)
        track_map = (track_map[0:1] + np.where(mag >= args.delta_zero, delta, 0.0)).astype(np.float32)
        print(f"  delta-zero (|δ|<{args.delta_zero}m): "
              f"{int((mag < args.delta_zero).sum())}/{mag.size} entries zeroed")

    # Trajectory candidate grid.
    h_g = np.arange(0, H, args.traj_downsample)
    w_g = np.arange(0, W, args.traj_downsample)
    Hg, Wg = np.meshgrid(h_g, w_g, indexing="ij")

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="floating",
                                control_width="large", show_logo=False)

    scene_std = float(recon_map.reshape(-1, 3).std())
    point_size = 0.04
    print(f"  scene std={scene_std:.4f}, point_size={point_size:.4f}")

    # ---- Frame 0 anchor cloud (always visible) -------------------------------
    keep_ds = keep_full[::args.downsample, ::args.downsample]
    keep_flat = keep_ds.reshape(-1)
    pts0 = recon_map[0, ::args.downsample, ::args.downsample].reshape(-1, 3)[keep_flat]
    rgb0 = rgb[0, ::args.downsample, ::args.downsample].reshape(-1, 3)[keep_flat]
    pc_anchor = server.scene.add_point_cloud(
        "/anchor", points=pts0, colors=rgb0,
        point_size=point_size, point_shape="rounded")

    # ---- Per-frame clouds (one per t, toggle visibility with slider) --------
    # Each frame uses its OWN depth edge mask so that object boundaries that
    # have moved since frame 0 are correctly excluded (prevents halo bleed).
    # keep_full (frame-0) is intentionally kept for trajectory query selection.
    pc_frame_t = []
    for t in range(T):
        keep_flat_t = (~_edge_mask(recon_map[t, :, :, 2]))[::args.downsample, ::args.downsample].reshape(-1)
        pts_t = recon_map[t, ::args.downsample, ::args.downsample].reshape(-1, 3)[keep_flat_t]
        rgb_t = rgb[t, ::args.downsample, ::args.downsample].reshape(-1, 3)[keep_flat_t]
        node = server.scene.add_point_cloud(
            f"/frame_t/{t}", points=pts_t, colors=rgb_t,
            point_size=point_size, point_shape="rounded")
        node.visible = (t == T - 1)
        pc_frame_t.append(node)
    cur_t = [T - 1]

    # ---- GUI -----------------------------------------------------------------
    g_play       = server.gui.add_button("Play")
    g_t          = server.gui.add_slider("Time t", min=0, max=T - 1, step=1,
                                         initial_value=T - 1)
    g_rgb_t      = server.gui.add_image(rgb[T - 1], label="Frame t")
    g_show_anchor = server.gui.add_checkbox("Show frame 0 cloud", initial_value=True)
    g_show_t     = server.gui.add_checkbox("Show frame t cloud", initial_value=True)
    g_show_traj  = server.gui.add_checkbox("Show trajectories", initial_value=True)
    g_ps         = server.gui.add_slider("Point size", min=0.0005, max=0.05,
                                         step=0.0005, initial_value=point_size)
    g_lw         = server.gui.add_slider("Line width", min=0.5, max=10.0,
                                         step=0.5, initial_value=0.5)
    g_pct        = server.gui.add_slider("Dynamic percentile", min=50.0, max=99.5,
                                         step=0.5, initial_value=args.dynamic_percentile)
    g_max        = server.gui.add_slider("Max tracks", min=50, max=10000,
                                         step=50, initial_value=args.max_tracks)
    g_fps        = server.gui.add_slider("Playback FPS", min=1.0, max=30.0,
                                         step=0.5, initial_value=6.5)

    is_playing = [False]
    last_time = [time.time()]
    frame_dt = [1.0 / 6.5]

    # ---- Trajectory polylines (rebuilt live on percentile / max_tracks change)
    seg_nodes = []                                   # (T-1,) line-segment nodes

    if args.traj_color == "time":
        cmap = matplotlib.colormaps["viridis"]
        seg_colors_T = np.array([cmap(i / max(T - 2, 1))[:3]
                                 for i in range(T - 1)], dtype=np.float32)

    def _rebuild_trajs():
        """Reselect dynamic pixels and rebuild trajectory line segments in place."""
        for n in seg_nodes:
            n.remove()
        seg_nodes.clear()

        thr = float(np.percentile(motion, g_pct.value))
        mask = _cc_clean((motion > thr) & keep_full)
        on_grid = mask[Hg, Wg]
        sh, sw = Hg[on_grid], Wg[on_grid]
        if len(sh) > int(g_max.value):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(sh), int(g_max.value), replace=False)
            sh, sw = sh[idx], sw[idx]
        N = len(sh)
        print(f"  [traj] p={g_pct.value:.1f}  thr={thr:.3f}  raw={int(mask.sum())}  "
              f"on grid={int(on_grid.sum())}  drawn={N}")
        if N == 0:
            return

        # Pull ~1% toward origin so lines win the depth test against
        # same-depth cloud sprites without visibly floating.
        trj = track_map[:, sh, sw, :] * 0.99                  # (T, N, 3)

        if args.traj_color == "query":
            hue = sh.astype(np.float32) / max(H - 1, 1)
            per_q = matplotlib.colormaps["hsv"](hue)[:, :3]
        for k in range(T - 1):
            segs = trj[k:k + 2].swapaxes(0, 1)                # (N, 2, 3)
            if args.traj_color == "time":
                cols = np.broadcast_to(seg_colors_T[k][None, None, :], (N, 2, 3)).copy()
            else:
                cols = np.broadcast_to(per_q[:, None, :], (N, 2, 3)).copy()
            node = server.scene.add_line_segments(
                f"/traj/{k}", segs, cols, line_width=g_lw.value)
            node.visible = (k < cur_t[0]) and g_show_traj.value
            seg_nodes.append(node)

    _rebuild_trajs()

    def _set_time(t_new):
        for i, n in enumerate(pc_frame_t):
            n.visible = (i == t_new) and g_show_t.value
        for k, n in enumerate(seg_nodes):
            n.visible = (k < t_new) and g_show_traj.value
        g_rgb_t.image = rgb[t_new]
        cur_t[0] = t_new

    @g_play.on_click
    def _(_):
        is_playing[0] = not is_playing[0]
        g_play.name = "Pause" if is_playing[0] else "Play"

    @g_t.on_update
    def _(_):
        _set_time(int(g_t.value))

    @g_fps.on_update
    def _(_):
        frame_dt[0] = 1.0 / max(0.1, g_fps.value)

    @g_show_anchor.on_update
    def _(_):
        pc_anchor.visible = g_show_anchor.value

    @g_show_t.on_update
    def _(_):
        for i, n in enumerate(pc_frame_t):
            n.visible = (i == cur_t[0]) and g_show_t.value

    @g_show_traj.on_update
    def _(_):
        for k, n in enumerate(seg_nodes):
            n.visible = (k < cur_t[0]) and g_show_traj.value

    @g_ps.on_update
    def _(_):
        pc_anchor.point_size = g_ps.value
        for n in pc_frame_t:
            n.point_size = g_ps.value

    @g_lw.on_update
    def _(_):
        for n in seg_nodes:
            n.line_width = g_lw.value

    @g_pct.on_update
    def _(_):
        _rebuild_trajs()

    @g_max.on_update
    def _(_):
        _rebuild_trajs()

    print(f"  Viser server running at http://localhost:{args.port}")
    print("  Scrub Time t: frame-0 cloud stays fixed, frame-t cloud + "
          "trajectories grow with t.")
    print("  'Dynamic percentile' / 'Max tracks' sliders rebuild trajectories live.")
    print("  Ctrl-C to exit.")
    try:
        while True:
            if is_playing[0] and time.time() - last_time[0] > frame_dt[0]:
                g_t.value = (g_t.value + 1) % T
                last_time[0] = time.time()
            time.sleep(1e-3)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
