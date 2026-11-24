"""VG3R browser demo: upload a video or pick an example, inspect the depth video and the point cloud with the cameras.

    pip install -e ".[demo]"
    python demo_gradio.py [--weights weights] [--port 7860]
"""
import argparse
import os
import tempfile

import torch

from vg3r import VG3R, predict
from vg3r.export import save_predictions
from vg3r.io import read_frames

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [os.path.join(HERE, "examples", d, "frames") for d in ("tum_walking_xyz", "kitti_0926_0002")]


def build(model):
    import gradio as gr

    def run(video, folder, stride, max_frames, edge_rel):
        src = video or folder
        if not src:
            raise gr.Error("upload a video or pick an example")
        frames = read_frames(src, stride=int(stride), max_frames=int(max_frames) or None)
        pred = predict(model, frames)
        out = tempfile.mkdtemp(prefix="vg3r_")
        files = save_predictions(out, pred, frames, max_points=500_000, edge_rel=float(edge_rel))
        h, w = pred["hw"]
        info = (f"{len(frames)} frames at {w} x {h}, {pred['passes']} forward pass(es), {pred['seconds']:.1f} s "
                f"({1000 * pred['seconds'] / len(frames):.0f} ms per frame)")
        return files.get("video"), files.get("scene"), [files[k] for k in ("points", "cameras", "depth") if k in files], info

    with gr.Blocks(title="VG3R") as demo:
        gr.Markdown("## VG3R: feed-forward 3D reconstruction from a video\nDepth, cameras and a point cloud from one video. "
                    "Depth and camera translations share one unknown scale per video.")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="video", sources=["upload"])
                folder = gr.Dropdown([e for e in EXAMPLES if os.path.isdir(e)], label="or an example (folder of frames)",
                                     value=None)
                stride = gr.Slider(1, 10, value=1, step=1, label="use every n-th frame")
                max_frames = gr.Slider(0, 600, value=200, step=1, label="at most this many frames (0 = all)")
                edge = gr.Slider(0.0, 0.2, value=0.04, step=0.01, label="drop points at depth edges (relative step; 0 keeps all)")
                go = gr.Button("Reconstruct", variant="primary")
            with gr.Column():
                info = gr.Markdown()
                depth = gr.Video(label="input and inverse depth")
                scene = gr.Model3D(label="points and cameras", clear_color=[1, 1, 1, 1])
                files = gr.File(label="points.ply, cameras.json, depth.npz", file_count="multiple")
        go.click(run, [video, folder, stride, max_frames, edge], [depth, scene, files, info])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    model = VG3R.from_pretrained(a.weights, "cuda" if torch.cuda.is_available() else "cpu")
    build(model).queue().launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
