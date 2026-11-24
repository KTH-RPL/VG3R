"""Checks of the evaluation code (CPU, no dataset needed): the metric code on synthetic data and the reference numbers of the paper.
    pytest eval/tests
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

REF = json.load(open(os.path.join(os.path.dirname(HERE), "reference", "paper_vg3r.json"), encoding="utf-8"))
# Table 1 of the paper, VG3R, as printed (three decimals)
TABLE = {("videodepth", "sintel", "Abs Rel"): 0.254, ("videodepth", "sintel", "δ < 1.25"): 0.699,
         ("videodepth", "bonn", "Abs Rel"): 0.037, ("videodepth", "bonn", "δ < 1.25"): 0.979,
         ("videodepth", "kitti", "Abs Rel"): 0.058, ("videodepth", "kitti", "δ < 1.25"): 0.979,
         ("tae", "scannetv2", "TAE"): 0.734,
         ("pose", "sintel", "ATE"): 0.142, ("pose", "sintel", "RPE trans"): 0.034, ("pose", "sintel", "RPE rot"): 0.360,
         ("pose", "tum", "ATE"): 0.031, ("pose", "tum", "RPE trans"): 0.012, ("pose", "tum", "RPE rot"): 0.457,
         ("pose", "scannetv2", "ATE"): 0.035, ("pose", "scannetv2", "RPE trans"): 0.015, ("pose", "scannetv2", "RPE rot"): 0.536,
         ("recon", "7scenes", "Acc-mean"): 0.018, ("recon", "7scenes", "Comp-mean"): 0.017, ("recon", "7scenes", "NC-mean"): 0.609,
         ("recon", "nrgbd", "Acc-mean"): 0.032, ("recon", "nrgbd", "Comp-mean"): 0.015, ("recon", "nrgbd", "NC-mean"): 0.666}
N_SEQ = {("videodepth", "sintel"): 14, ("videodepth", "bonn"): 5, ("videodepth", "kitti"): 13, ("tae", "scannetv2"): 94,
         ("pose", "sintel"): 14, ("pose", "tum"): 8, ("pose", "scannetv2"): 94, ("recon", "7scenes"): 18, ("recon", "nrgbd"): 9}


def test_reference_is_table_1():
    for (b, d, k), v in TABLE.items():
        assert round(REF[b][d]["avg"][k], 3) == v, (b, d, k, REF[b][d]["avg"][k])


def test_reference_averages_of_sequences():
    for (b, d), n in N_SEQ.items():
        per = REF[b][d]["per_seq"]
        assert len(per) == n == REF[b][d]["avg"]["n_seq"], (b, d, len(per))
        keys = [k for (bb, dd, k) in TABLE if (bb, dd) == (b, d)]
        for k in keys:
            vals = [p[k] for p in per.values()]
            if b == "videodepth":                                   # weighted by the valid pixels of each sequence
                avg = np.average(vals, weights=[p["valid_pixels"] for p in per.values()])
            else:
                avg = np.mean(vals)
            assert abs(avg - REF[b][d]["avg"][k]) < 1e-9, (b, d, k)


def test_depth_scale_alignment():
    D = C.pi3("utils/depth.py")
    gt = np.random.default_rng(0).uniform(1.0, 10.0, (3, 20, 30)).astype(np.float32)
    r = D.depth_evaluation(0.5 * gt, gt, max_depth=70, use_gpu=False, align_with_scale=True)[0]
    assert float(r["Abs Rel"]) < 1e-5 and float(r["δ < 1.25"]) == 1.0


def test_umeyama():
    U = C.pi3("mv_recon/utils.py")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 500))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = q * np.sign(np.linalg.det(q))
    Y = 2.5 * R @ X + np.array([[0.3], [-1.0], [2.0]])
    c, R_, t = U.umeyama(X, Y)
    assert abs(c - 2.5) < 1e-9 and np.allclose(R_, R) and np.allclose(t.ravel(), [0.3, -1.0, 2.0])


def test_tae_static_scene():
    import torch
    V = C.by_path("_vda_eval_tae", C.VDA_TAE)
    d = torch.full((24, 32), 2.0, dtype=torch.float64)
    K = np.array([[30.0, 0, 16], [0, 30.0, 12], [0, 0, 1]])
    err = V.tae_torch(d, d.clone(), torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), K,
                      torch.ones_like(d, dtype=torch.bool))
    assert float(err) < 1e-12
