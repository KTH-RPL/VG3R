"""Reconstruction from video on 7-Scenes (18 test sequences) and Neural RGB-D (9 scenes): the scenes, loaders and ground truth of the
pi^3 evaluation code (prepared as Spann3R does), but consecutive frames: the first 90 frames at a stride of 3 of every sequence (the
video sampling of the TUM-dynamics and ScanNet camera benchmarks) instead of keyframes 40 to 500 frames apart.

VG3R's points of all frames (vg3r.predict, windows of 37 frames) are resized to the ground-truth grid (bilinear, antialiased) and
kept on every 2nd pixel row and column. Then the metrics of the pi^3 code (mv_recon/eval.py, after CUT3R): a similarity transform
(Umeyama) on the valid pixels, ICP (point to point, threshold 0.1 m), normals, accuracy and completeness (mean distance to the nearest
point, meters) and normal consistency (NC = the mean of its two directions); each averaged over the sequences of a dataset.

    python eval/run_recon.py infer   --data-root data --weights weights --out results    (GPU)
    python eval/run_recon.py metrics --out results                                       (CPU; needs open3d, not torch)
=> results/recon/<dataset>/<sequence>.npz, results/recon/<dataset>.json
"""
import argparse
import json
import os
import sys

import numpy as np

import common as C

DATASETS = ("7scenes", "nrgbd")
SEQ_MAP = {"7scenes": "datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf40.json",       # the sequences of the dense setting
           "nrgbd": "datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf100.json"}
IDS = list(range(0, 90 * 3, 3))                  # the first 90 frames at a stride of 3
PIX_STRIDE = 2
LOAD_IMG_SIZE = 518                              # load_img_size of the pi^3 configuration (ground-truth grid 518 pixels wide)


def dataset(ds, data_root, cache_dir):
    """The pi^3 loaders import their siblings as top-level packages (utils, datasets): put their root first on sys.path. Done after
    the model is loaded, as in the pi^3 code."""
    if C.PI3 not in sys.path:
        sys.path.insert(0, C.PI3)
    os.makedirs(cache_dir, exist_ok=True)
    if ds == "7scenes":
        from datasets.sevenscenes import SevenScenes
        return SevenScenes(SEVENSCENES_DIR=os.path.join(data_root, "7scenes"), split="test", load_img_size=LOAD_IMG_SIZE,
                           cache_file=os.path.join(cache_dir, "7scenes_mv_recon_cache.npy"))
    from datasets.nrgbd import NRGBD
    return NRGBD(NRGBD_DIR=os.path.join(data_root, "nrgbd"), load_img_size=LOAD_IMG_SIZE,
                 cache_file=os.path.join(cache_dir, "nrgbd_mv_recon_cache.npy"))


def cmd_infer(a):
    import torch
    import torch.nn.functional as F
    model = C.load_model(a.weights)
    for ds in a.datasets.split(","):
        data = dataset(ds, os.path.abspath(a.data_root), f"{a.out}/recon/dataset_cache")
        seqs = list(json.load(open(os.path.join(C.PI3, SEQ_MAP[ds]))))
        od = f"{a.out}/recon/{ds}"
        os.makedirs(od, exist_ok=True)
        for i, seq in enumerate(seqs[: a.max_seqs or None], 1):
            f = f"{od}/{seq.replace('/', '-')}.npz"
            if os.path.exists(f):
                continue
            d = data.get_data(sequence_name=seq, ids=IDS)
            imgs = d["images"]                                         # (N, 3, H, W) in [0, 1] on the ground-truth grid
            H, W = imgs.shape[-2:]
            fr = C.load_frames(d["image_paths"])                       # the original frames are the model input
            pred = C.predict(model, fr)
            pts = torch.from_numpy(pred["points"]).permute(0, 3, 1, 2)
            pts = F.interpolate(pts, (H, W), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1).numpy()
            assert pts.shape == d["pointclouds"].shape, (pts.shape, d["pointclouds"].shape)
            s = PIX_STRIDE
            np.savez_compressed(f, pred=pts.astype(np.float32)[:, ::s, ::s], gt=np.asarray(d["pointclouds"])[:, ::s, ::s],
                                valid=np.asarray(d["valid_mask"], bool)[:, ::s, ::s],
                                img=(imgs.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)[:, ::s, ::s],
                                seconds=float(pred["seconds"]), model_hw=np.int32(pred["hw"]), ids=np.int64(IDS))
            print(f"[recon] {ds} [{i}/{len(seqs)}] {seq}: {len(fr)} frames, {pred['seconds']:.1f} s", flush=True)
        C.write_json(f"{od}/dataset.json", dict(n_sequences=len(data), sequences=seqs))


def cmd_metrics(a):
    import open3d as o3d
    U = C.pi3("mv_recon/utils.py")
    keys = ("Acc-mean", "Acc-med", "Comp-mean", "Comp-med", "NC-mean", "NC-med", "NC1-mean", "NC1-med", "NC2-mean", "NC2-med")
    for ds in a.datasets.split(","):
        od = f"{a.out}/recon/{ds}"
        if not os.path.exists(f"{od}/dataset.json"):
            print(f"[recon] {ds}: no inference output; run infer first")
            continue
        n_seq = json.load(open(f"{od}/dataset.json"))["n_sequences"]
        files = sorted(f for f in os.listdir(od) if f.endswith(".npz"))
        if len(files) != n_seq:
            print(f"[recon] {ds}: {len(files)} of {n_seq} sequences inferred; run infer first")
            continue
        per, tot = {}, {k: 0.0 for k in keys}
        for f in files:
            z = np.load(f"{od}/{f}")
            pred, gt, valid = z["pred"], z["gt"], z["valid"]
            colors = z["img"][valid].reshape(-1, 3) / 255.0
            c, R, t = U.umeyama(pred[valid].T, gt[valid].T)
            pred = c * np.einsum("nhwj, ij -> nhwi", pred, R) + t.T
            pp, gp = pred[valid].reshape(-1, 3), gt[valid].reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points, pcd.colors = o3d.utility.Vector3dVector(pp), o3d.utility.Vector3dVector(colors)
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points, pcd_gt.colors = o3d.utility.Vector3dVector(gp), o3d.utility.Vector3dVector(colors)
            reg = o3d.pipelines.registration.registration_icp(pcd, pcd_gt, 0.1, np.eye(4),
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint())
            pcd = pcd.transform(reg.transformation)
            pcd.estimate_normals()
            pcd_gt.estimate_normals()
            pn, gn = np.asarray(pcd.normals), np.asarray(pcd_gt.normals)
            acc, acc_med, nc1, nc1_med = U.accuracy(pcd_gt.points, pcd.points, gn, pn)
            comp, comp_med, nc2, nc2_med = U.completion(pcd_gt.points, pcd.points, gn, pn)
            r = {"Acc-mean": acc, "Acc-med": acc_med, "Comp-mean": comp, "Comp-med": comp_med, "NC-mean": (nc1 + nc2) / 2,
                 "NC-med": (nc1_med + nc2_med) / 2, "NC1-mean": nc1, "NC1-med": nc1_med, "NC2-mean": nc2, "NC2-med": nc2_med}
            per[f[:-4]] = {k: float(v) for k, v in r.items()}
            for k in keys:
                tot[k] += float(r[k])
        avg = {k: v / n_seq for k, v in tot.items()}
        avg["n_seq"] = n_seq
        C.write_json(f"{a.out}/recon/{ds}.json", dict(dataset=ds, avg=avg, per_seq=per))
        print(f"[recon] {ds}: Acc {avg['Acc-mean']:.4f} Comp {avg['Comp-mean']:.4f} NC {avg['NC-mean']:.4f} ({n_seq} sequences)",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["infer", "metrics"])
    ap.add_argument("--data-root", default=None, help="infer: folder with 7scenes/ and nrgbd/ (see eval/README.md)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--max-seqs", type=int, default=0, help="infer: only the first N sequences of each dataset (a quick check)")
    a = ap.parse_args()
    for k in ("out", "data_root", "weights"):
        if getattr(a, k, None):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    if a.cmd == "infer" and not a.data_root:
        ap.error("infer needs --data-root")
    dict(infer=cmd_infer, metrics=cmd_metrics)[a.cmd](a)


if __name__ == "__main__":
    main()
