"""Camera trajectories on Sintel (14 sequences, all frames), TUM-dynamics (8 sequences x 90 frames) and ScanNet (100 clips x 90
frames), the sequences and ground-truth trajectories of the pi^3 evaluation code (prepared as MonST3R does), with its trajectory
metrics (relpose/evo_utils.py, after CUT3R): the predicted camera positions are aligned to the ground truth by a similarity
transform; ATE = RMSE of the aligned positions, RPE-t and RPE-r = RMSE of the relative translation and rotation (degrees) between
consecutive frames; each averaged over the sequences of a dataset. Sequences whose ground-truth trajectory is not finite are skipped,
as the pi^3 code does (6 ScanNet clips).

Every sequence is one video for VG3R (the whole sequence; windows of 37 frames when longer). On ScanNet the predicted depth is kept
for the temporal-consistency metric (run_tae.py).

    python eval/run_pose.py infer   --data-root data --weights weights --out results    (GPU)
    python eval/run_pose.py metrics --data-root data --out results                      (CPU; needs evo)
=> results/pose/<dataset>/<sequence>/pred_poses.npy (camera-to-world), results/pose/<dataset>.json
"""
import argparse
import os

import numpy as np

import common as C

DATASETS = ("sintel", "tum", "scannetv2")
STRIDE = 1                                       # pose_eval_stride of the pi^3 configuration


def cmd_infer(a):
    model = C.load_model(a.weights)
    files = C.pi3("utils/files.py")
    data = C.data_config("relpose-distance", a.data_root)
    for ds in a.datasets.split(","):
        seqs = files.get_all_sequences(data[ds])[: a.max_seqs or None]
        for i, seq in enumerate(seqs, 1):
            fl = files.list_imgs_a_sequence(data[ds], seq)[::STRIDE]
            od = f"{a.out}/pose/{ds}/{seq}"
            if os.path.exists(f"{od}/pred_poses.npy") and os.path.exists(f"{od}/info.json"):
                continue
            fr = C.load_frames(fl)
            pred = C.predict(model, fr)
            os.makedirs(od, exist_ok=True)
            np.save(f"{od}/pred_poses.npy", pred["c2w"])
            np.save(f"{od}/pred_intrinsics.npy", pred["K"])
            if ds == "scannetv2":
                np.save(f"{od}/pred_depth.npy", pred["depth"].astype(np.float32))
            C.write_json(f"{od}/info.json", C.info(pred, fr, dict(first=os.path.basename(fl[0]), last=os.path.basename(fl[-1]))))
            print(f"[pose] {ds} [{i}/{len(seqs)}] {seq}: {len(fl)} frames, {pred['seconds']:.1f} s", flush=True)


def cmd_metrics(a):
    import matplotlib
    matplotlib.use("Agg")
    E = C.pi3("relpose/evo_utils.py")
    files = C.pi3("utils/files.py")
    data = C.data_config("relpose-distance", a.data_root)
    for ds in a.datasets.split(","):
        info = data[ds]
        seqs = files.get_all_sequences(info)
        per, skipped, missing = {}, {}, []
        for seq in seqs:
            od = f"{a.out}/pose/{ds}/{seq}"
            if not os.path.exists(f"{od}/pred_poses.npy"):
                missing.append(seq)
                continue
            pr = np.load(f"{od}/pred_poses.npy")
            pred_traj = E.get_tum_poses(pr)
            E.save_tum_poses(pred_traj, f"{od}/pred_traj.txt")
            try:
                gt_traj = E.load_traj(gt_traj_file=info.anno.path.format(seq=seq), traj_format=info.anno.format, stride=STRIDE)
            except np.linalg.LinAlgError:
                skipped[seq] = "the ground-truth trajectory does not load"
                continue
            if not np.isfinite(gt_traj[0]).all():
                skipped[seq] = "non-finite ground-truth pose"
                continue
            assert len(gt_traj[0]) == len(pr), (seq, len(gt_traj[0]), len(pr))
            ate, rpe_t, rpe_r = E.eval_metrics(pred_traj, gt_traj, seq=seq, filename=f"{od}/eval_metric.txt")
            per[seq] = {"ATE": float(ate), "RPE trans": float(rpe_t), "RPE rot": float(rpe_r)}
        if missing:
            print(f"[pose] {ds}: {len(missing)} of {len(seqs)} sequences without prediction ({missing[:3]}); run infer first")
            continue
        ate, rpe_t, rpe_r = E.calculate_averages([(s, r["ATE"], r["RPE trans"], r["RPE rot"]) for s, r in per.items()])
        avg = {"ATE": float(ate), "RPE trans": float(rpe_t), "RPE rot": float(rpe_r), "n_seq": len(per)}
        C.write_json(f"{a.out}/pose/{ds}.json", dict(dataset=ds, avg=avg, per_seq=per, skipped=skipped))
        print(f"[pose] {ds}: ATE {ate:.4f} RPE-t {rpe_t:.4f} RPE-r {rpe_r:.4f} ({len(per)} sequences, {len(skipped)} skipped)",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["infer", "metrics"])
    ap.add_argument("--data-root", required=True, help="folder with sintel/, tum/, scannetv2/ (see eval/README.md)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--max-seqs", type=int, default=0, help="infer: only the first N sequences of each dataset (a quick check)")
    a = ap.parse_args()
    for k in ("out", "data_root", "weights"):
        if getattr(a, k, None):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    dict(infer=cmd_infer, metrics=cmd_metrics)[a.cmd](a)


if __name__ == "__main__":
    main()
