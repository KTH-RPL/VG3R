"""VG3R command-line demo: a video (file or folder of frames) => depth, cameras and a point cloud.

    python demo.py --video examples/tum_walking_xyz/frames --out outputs/tum_walking_xyz
    python demo.py --video my_video.mp4 --stride 2 --max-frames 200 --out outputs/my_video

Writes points.ply, cameras.json, depth.npz, depth.mp4 (input and inverse depth side by side) and scene.glb (points and cameras).
"""
import argparse
import os
import time

import numpy as np
import torch

from vg3r import VG3R, predict
from vg3r.export import save_predictions
from vg3r.io import read_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="a video file or a folder of frames (sorted by name)")
    ap.add_argument("--out", default=None, help="output folder (default outputs/<name of the video>)")
    ap.add_argument("--weights", default="weights", help="weights folder, see README.md")
    ap.add_argument("--stride", type=int, default=1, help="use every stride-th frame")
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--max-frames", type=int, default=None, help="at most this many frames (after the stride)")
    ap.add_argument("--max-points", type=int, default=2_000_000, help="points in points.ply")
    ap.add_argument("--edge-rel", type=float, default=0.04,
                    help="drop points at depth discontinuities above this relative depth difference (0 keeps every point)")
    ap.add_argument("--fps", type=int, default=10, help="frame rate of depth.mp4")
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(os.path.normpath(a.video)))[0]
    if name == "frames":
        name = os.path.basename(os.path.dirname(os.path.normpath(a.video)))
    out = a.out or os.path.join("outputs", name)

    frames = read_frames(a.video, stride=a.stride, max_frames=a.max_frames, start=a.start)
    print(f"{len(frames)} frames of {frames.shape[2]} x {frames.shape[1]} from {a.video}")
    t0 = time.time()
    model = VG3R.from_pretrained(a.weights, a.device)
    print(f"model loaded in {time.time() - t0:.0f} s")
    pred = predict(model, frames)
    h, w = pred["hw"]
    print(f"input {w} x {h}, {pred['passes']} forward pass(es) (windows at frames {pred['windows']}), "
          f"{pred['seconds']:.1f} s, {1000 * pred['seconds'] / len(frames):.0f} ms per frame")
    files = save_predictions(out, pred, frames, max_points=a.max_points, edge_rel=a.edge_rel, glb=not a.no_glb,
                             video=not a.no_video, fps=a.fps)
    path = np.linalg.norm(np.diff(pred["c2w"][:, :3, 3], axis=0), axis=1).sum()
    print(f"camera path length {path:.2f} (units of the median depth of frame 0)")
    for k, f in files.items():
        print(f"  {k:8s} {f}")


if __name__ == "__main__":
    main()
