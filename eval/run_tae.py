"""Temporal consistency of the video depth: TAE, the metric of Video Depth Anything (third_party/vda/eval_tae.py), on the 100
ScanNet clips of the camera benchmark (90 frames at a stride of 3, so consecutive frames are 3 frames apart), from the depth that
run_pose.py infer keeps. Per clip:
  1) the predicted depth, resized (bilinear) to the ground truth and cropped like it ([8:-8, 11:-11], as Video Depth Anything does on
     ScanNet), as disparity clipped at 1e-3;
  2) one scale and shift per clip, fitted by least squares in disparity to the ground truth in (1e-3, 10 m); aligned disparity =>
     depth, clipped to [1e-3, 10 m];
  3) for every pair of consecutive frames, both ways: the depth of one frame back-projected with K (principal point shifted by the
     crop), moved by the ground-truth relative pose and projected into the other frame, AbsRel against its depth (tae_torch);
TAE = 100 x the mean over the 2 (N - 1) terms; the dataset TAE is the mean over clips. Clips with a non-finite ground-truth pose are
skipped (6). Lower is more consistent.

    python eval/run_tae.py --data-root data --out results       (CPU)
=> results/tae/scannetv2.json
"""
import argparse
import glob
import os

import numpy as np

import common as C

CROP = (8, -8, 11, -11)
MAX_DEPTH = 10.0


def tae_sequence(V, pred, gt, Ks, poses, max_depth):
    """pred [N, H, W] predicted depth on the ground-truth grid, gt [N, H, W] meters, Ks [N, 3, 3], poses [N, 4, 4] camera-to-world."""
    import torch
    infs = 1.0 / np.maximum(pred.astype(np.float64), 1e-8)            # the predicted depth as disparity
    valid_mask = np.logical_and(gt > 1e-3, gt < max_depth)
    gt_disp_masked = 1.0 / (gt[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
    infs = np.clip(infs, a_min=1e-3, a_max=None)
    pred_disp_masked = infs[valid_mask].reshape((-1, 1)).astype(np.float64)
    A = np.concatenate([pred_disp_masked, np.ones_like(pred_disp_masked)], axis=-1)
    scale, shift = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
    aligned = np.clip(scale * infs + shift, a_min=1e-3, a_max=None)
    pred_depth = np.clip(V.depth2disparity(aligned), a_min=1e-3, a_max=max_depth)
    err, n = 0.0, len(pred_depth)
    for i in range(n - 1):
        d1, d2 = torch.from_numpy(pred_depth[i]), torch.from_numpy(pred_depth[i + 1])
        T_2_1 = np.linalg.inv(poses[i + 1]) @ poses[i]
        T_1_2 = np.linalg.inv(T_2_1)
        m1, m2 = torch.ones_like(d1, dtype=torch.bool), torch.ones_like(d2, dtype=torch.bool)
        err += float(V.tae_torch(d1, d2, torch.from_numpy(T_2_1[:3, :3]), torch.from_numpy(T_2_1[:3, 3]), Ks[i], m2))
        err += float(V.tae_torch(d2, d1, torch.from_numpy(T_1_2[:3, :3]), torch.from_numpy(T_1_2[:3, 3]), Ks[i], m1))
    return 100.0 * err / (2 * (n - 1)), dict(scale=float(scale[0]), shift=float(shift[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder with scannetv2/ (see eval/README.md)")
    ap.add_argument("--out", required=True, help="the results folder of run_pose.py")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads of torch (the paper used 8)")
    a = ap.parse_args()
    for k in ("out", "data_root", "weights"):
        if getattr(a, k, None):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    import cv2
    import torch
    torch.set_num_threads(a.threads)
    V = C.by_path("_vda_eval_tae", C.VDA_TAE)
    root = os.path.join(a.data_root, "scannetv2")
    t, b, l, r = CROP
    per, skipped, missing = {}, {}, []
    for seq in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        f = f"{a.out}/pose/scannetv2/{seq}/pred_depth.npy"
        if not os.path.exists(f):
            missing.append(seq)
            continue
        poses = np.loadtxt(f"{root}/{seq}/pose_90.txt").reshape(-1, 4, 4)
        if not np.isfinite(poses).all():
            skipped[seq] = "non-finite ground-truth pose"
            continue
        gf = sorted(glob.glob(f"{root}/{seq}/depth_90/frame_*.png"))
        gt = np.stack([cv2.imread(g, -1).astype(np.float64) / 1000.0 for g in gf])
        H, W = gt.shape[1:]
        pr = np.load(f)
        assert len(pr) == len(gf) == len(poses), (seq, len(pr), len(gf), len(poses))
        pred = np.stack([cv2.resize(x.astype(np.float32), (W, H)) for x in pr])[:, t:b, l:r]
        K = np.loadtxt(f"{root}/{seq}/intrinsic/intrinsic_depth.txt")[:3, :3].copy()
        K[0, 2] -= l
        K[1, 2] -= t
        tae, fit = tae_sequence(V, pred, gt[:, t:b, l:r], np.tile(K, (len(pr), 1, 1)), poses, MAX_DEPTH)
        per[seq] = dict(tae=tae, align=fit)
    if missing:
        raise SystemExit(f"[tae] {len(missing)} clips without the depth of run_pose.py infer ({missing[:3]})")
    tae = float(np.mean([v["tae"] for v in per.values()]))
    C.write_json(f"{a.out}/tae/scannetv2.json", dict(dataset="scannetv2", tae=tae, n_seq=len(per), skipped=skipped, per_seq=per))
    print(f"[tae] scannetv2: TAE {tae:.4f} ({len(per)} clips, {len(skipped)} skipped)", flush=True)


if __name__ == "__main__":
    main()
