"""Video depth on Sintel (14 sequences), Bonn (5 x 110 frames) and KITTI (13 x 110 frames), the sequences and ground truth of the
pi^3 evaluation code (prepared as MonST3R does), with its depth metrics.

Every sequence is one video for VG3R (vg3r.predict: the whole sequence, in windows of 37 frames when it is longer). VG3R predicts
depth up to one scale per video, so each sequence gets one scale, fitted by the pi^3 code's own scale alignment (depth_evaluation
with align_with_scale: 10 Weiszfeld iterations of the L1 error) on the valid ground truth (depth > 0, below 70 m on Sintel and Bonn;
Sintel predictions are also clipped at 70 m). The predicted depth is resized to the ground truth with bicubic interpolation. AbsRel
and delta < 1.25 are averaged over the sequences of a dataset, weighted by their valid pixels (as the pi^3 code does).

    python eval/run_videodepth.py infer   --data-root data --weights weights --out results     (GPU)
    python eval/run_videodepth.py metrics --data-root data --out results                       (GPU: the pi^3 metrics run on CUDA)
=> results/videodepth/<dataset>/<sequence>/frame_XXXX.npy (depth at the model's resolution), results/videodepth/<dataset>.json
"""
import argparse
import glob
import os
import time

import numpy as np

import common as C

DATASETS = ("sintel", "bonn", "kitti")


def cmd_infer(a):
    model = C.load_model(a.weights)
    files = C.pi3("utils/files.py")
    data = C.data_config("depth", a.data_root)
    for ds in a.datasets.split(","):
        seqs = files.get_all_sequences(data[ds])[: a.max_seqs or None]
        for i, seq in enumerate(seqs, 1):
            fl = files.list_imgs_a_sequence(data[ds], seq)
            od = f"{a.out}/videodepth/{ds}/{seq}"
            if len(glob.glob(f"{od}/frame_*.npy")) == len(fl) and os.path.exists(f"{od}/info.json"):
                continue
            fr = C.load_frames(fl)
            pred = C.predict(model, fr)
            os.makedirs(od, exist_ok=True)
            for k, d in enumerate(pred["depth"]):
                np.save(f"{od}/frame_{k:04d}.npy", d)
            C.write_json(f"{od}/info.json", C.info(pred, fr))
            print(f"[videodepth] {ds} [{i}/{len(seqs)}] {seq}: {len(fl)} frames, {pred['seconds']:.1f} s", flush=True)


def seq_average(per):
    """The pi^3 code's average over sequences: every metric weighted by the valid pixels of the sequence."""
    rs = list(per.values())
    out = {k: float(np.average([r[k] for r in rs], weights=[r["valid_pixels"] for r in rs])) for k in rs[0]
           if k not in ("valid_pixels", "fps", "total_time", "infer_time", "scale", "shift")}
    out["n_seq"] = len(rs)
    out["valid_pixels"] = int(sum(r["valid_pixels"] for r in rs))
    return out


def cmd_metrics(a):
    import cv2
    D = C.pi3("utils/depth.py")
    files = C.pi3("utils/files.py")
    data = C.data_config("depth", a.data_root)
    for ds in a.datasets.split(","):
        meta = D.EVAL_DEPTH_METADATA[ds]
        kw = dict(meta["depth_evaluation_kwargs"])                    # a copy: never change the module's dict
        if a.cpu:
            kw["use_gpu"] = False
        per, missing = {}, []
        t0 = time.time()
        for seq in files.get_all_sequences(data[ds]):
            gt_paths = files.list_depths_a_sequence(data[ds], seq)
            pf = sorted(glob.glob(f"{a.out}/videodepth/{ds}/{seq}/frame_*.npy"))
            if len(pf) != len(gt_paths):
                missing.append(seq)
                continue
            gt = np.stack([meta["depth_read_func"](p) for p in gt_paths], axis=0)
            H, W = gt.shape[1:]
            pred = np.stack([cv2.resize(np.load(f), (W, H), interpolation=cv2.INTER_CUBIC) for f in pf], axis=0)
            r = D.depth_evaluation(pred, gt, **kw, align_with_scale=True)[0]
            per[seq] = {k: float(v) for k, v in r.items()}
        if missing:
            print(f"[videodepth] {ds}: {len(missing)} sequences without output ({missing[:3]}); run infer first")
            continue
        avg = seq_average(per)
        C.write_json(f"{a.out}/videodepth/{ds}.json", dict(dataset=ds, avg=avg, per_seq=per))
        print(f"[videodepth] {ds}: AbsRel {avg['Abs Rel']:.4f} delta<1.25 {avg['δ < 1.25']:.4f} "
              f"({avg['n_seq']} sequences, {time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["infer", "metrics"])
    ap.add_argument("--data-root", required=True, help="folder with sintel/, bonn/, kitti/ (see eval/README.md)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--max-seqs", type=int, default=0, help="infer: only the first N sequences of each dataset (a quick check)")
    ap.add_argument("--cpu", action="store_true", help="metrics: on the CPU (the paper's numbers were computed on CUDA)")
    a = ap.parse_args()
    for k in ("out", "data_root", "weights"):
        if getattr(a, k, None):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    dict(infer=cmd_infer, metrics=cmd_metrics)[a.cmd](a)


if __name__ == "__main__":
    main()
