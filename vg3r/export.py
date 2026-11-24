"""Writing predictions: a point cloud (PLY), the cameras (JSON), the depth maps (NPZ), a depth video (MP4) and a scene with the cameras
(GLB, needs trimesh).

    save_predictions(out_dir, pred, frames)
"""
import json
import os

import numpy as np

from .inference import resize


def depth_edges(D, rel=0.04):
    """Pixels of D [N, h, w] at depth discontinuities: the relative depth difference to a 4-neighbour exceeds rel. Points there lie
    between foreground and background ("flying pixels") and are left out of the point cloud by default."""
    E = np.zeros(D.shape, bool)
    for ax, sh in ((1, 1), (1, -1), (2, 1), (2, -1)):
        N = np.roll(D, sh, axis=ax)
        E |= np.abs(D - N) > rel * np.minimum(D, N)
    return E


def colors_of(pred, frames):
    """The frames at the model's input size, as point colors [N, h, w, 3] uint8."""
    return resize(np.asarray(frames), pred["hw"])


def point_cloud(pred, frames, every=1, max_points=2_000_000, edge_rel=0.04, seed=0):
    """All frames' points (every `every`-th frame), without depth discontinuities when edge_rel > 0, randomly subsampled to at
    most max_points => xyz [M, 3] float32, rgb [M, 3] uint8."""
    D, X = pred["depth"][::every], pred["points"][::every]
    C = colors_of(pred, frames)[::every]
    keep = np.isfinite(D) & (D > 0)
    if edge_rel and edge_rel > 0:
        keep &= ~depth_edges(D, edge_rel)
    xyz, rgb = X[keep], C[keep]
    if max_points and len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def write_ply(path, xyz, rgb):
    """Binary little-endian PLY with float xyz and uchar rgb."""
    n = len(xyz)
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(rec.tobytes())


def write_cameras(path, pred):
    h, w = pred["hw"]
    out = dict(convention="OpenCV camera-to-world (x right, y down, z forward); world = the camera of frame 0; translations and depth "
                          "share one unknown scale per video (the median depth of frame 0 is 1)",
               image_size=[int(h), int(w)],
               frames=[dict(index=i, c2w=pred["c2w"][i].tolist(), K=pred["K"][i].tolist()) for i in range(len(pred["c2w"]))])
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def inverse_depth_colors(D, lo=None, hi=None):
    """[N, h, w] depth => [N, h, w, 3] uint8, inverse depth with the turbo color map, one range per video (2nd to 98th percentile)."""
    from matplotlib import colormaps
    inv = 1.0 / np.maximum(D, 1e-6)
    lo = np.percentile(inv, 2) if lo is None else lo
    hi = np.percentile(inv, 98) if hi is None else hi
    t = np.clip((inv - lo) / max(hi - lo, 1e-12), 0, 1)
    return (colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)


def write_depth_video(path, pred, frames, fps=10):
    """The input frames and the inverse depth side by side (MP4, H.264; needs imageio-ffmpeg)."""
    import imageio.v2 as iio
    rgb = colors_of(pred, frames)
    dep = inverse_depth_colors(pred["depth"])
    h, w = pred["hw"]
    with iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as wr:
        for a, b in zip(rgb, dep):
            wr.append_data(np.concatenate([a, b], 1)[: h - h % 2, : (2 * w) - (2 * w) % 2])


def write_glb(path, xyz, rgb, pred, every=None, scale=None):
    """Points and camera frustums (colored from blue to yellow by time) as one GLB scene (needs trimesh)."""
    import trimesh
    N = len(pred["c2w"])
    every = every or max(1, N // 30)
    extent = np.percentile(np.linalg.norm(xyz - np.median(xyz, 0), axis=1), 90) if len(xyz) else 1.0
    s = scale or 0.05 * extent
    h, w = pred["hw"]
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(xyz, colors=np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1)))
    for i in range(0, N, every):
        K, C = pred["K"][i], pred["c2w"][i]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        rays = np.c_[(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)] * s
        v = np.vstack([np.zeros(3), rays]) @ C[:3, :3].T + C[:3, 3]
        f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3]])
        t = i / max(N - 1, 1)
        col = (np.array([62, 120, 178]) * (1 - t) + np.array([201, 162, 39]) * t).astype(np.uint8)
        m = trimesh.Trimesh(v, f, process=False)
        m.visual.face_colors = np.r_[col, 255]
        scene.add_geometry(m)
    scene.export(path)


def save_predictions(out_dir, pred, frames, max_points=2_000_000, edge_rel=0.04, glb=True, video=True, fps=10):
    """points.ply, cameras.json, depth.npz and, when the optional packages are there, depth.mp4 and scene.glb. => written paths."""
    os.makedirs(out_dir, exist_ok=True)
    xyz, rgb = point_cloud(pred, frames, max_points=max_points, edge_rel=edge_rel)
    out = {}
    write_ply(f"{out_dir}/points.ply", xyz, rgb)
    out["points"] = f"{out_dir}/points.ply"
    write_cameras(f"{out_dir}/cameras.json", pred)
    out["cameras"] = f"{out_dir}/cameras.json"
    np.savez_compressed(f"{out_dir}/depth.npz", depth=pred["depth"], K=pred["K"], c2w=pred["c2w"])
    out["depth"] = f"{out_dir}/depth.npz"
    if video:
        try:
            write_depth_video(f"{out_dir}/depth.mp4", pred, frames, fps=fps)
            out["video"] = f"{out_dir}/depth.mp4"
        except ImportError as e:
            print(f"depth.mp4 skipped ({e}); pip install imageio-ffmpeg matplotlib")
    if glb:
        try:
            k = min(len(xyz), 500_000)
            write_glb(f"{out_dir}/scene.glb", xyz[:k], rgb[:k], pred)
            out["scene"] = f"{out_dir}/scene.glb"
        except ImportError as e:
            print(f"scene.glb skipped ({e}); pip install trimesh")
    return out
