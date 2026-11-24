"""The VG3R row of Table 1 from the results of the evaluation scripts, and with --compare-paper the difference to the numbers of the
paper (reference/paper_vg3r.json), per dataset and per sequence.

    python eval/make_table.py --results results [--compare-paper]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# (benchmark, dataset, metric key, column label) in the order of Table 1
COLUMNS = [("videodepth", "sintel", "Abs Rel", "Sintel AbsRel"), ("videodepth", "sintel", "δ < 1.25", "Sintel δ<1.25"),
           ("videodepth", "bonn", "Abs Rel", "Bonn AbsRel"), ("videodepth", "bonn", "δ < 1.25", "Bonn δ<1.25"),
           ("videodepth", "kitti", "Abs Rel", "KITTI AbsRel"), ("videodepth", "kitti", "δ < 1.25", "KITTI δ<1.25"),
           ("tae", "scannetv2", "TAE", "ScanNet TAE"),
           ("pose", "sintel", "ATE", "Sintel ATE"), ("pose", "sintel", "RPE trans", "Sintel RPE-t"),
           ("pose", "sintel", "RPE rot", "Sintel RPE-r"),
           ("pose", "tum", "ATE", "TUM ATE"), ("pose", "tum", "RPE trans", "TUM RPE-t"), ("pose", "tum", "RPE rot", "TUM RPE-r"),
           ("pose", "scannetv2", "ATE", "ScanNet ATE"), ("pose", "scannetv2", "RPE trans", "ScanNet RPE-t"),
           ("pose", "scannetv2", "RPE rot", "ScanNet RPE-r"),
           ("recon", "7scenes", "Acc-mean", "7-Scenes Acc"), ("recon", "7scenes", "Comp-mean", "7-Scenes Comp"),
           ("recon", "7scenes", "NC-mean", "7-Scenes NC"),
           ("recon", "nrgbd", "Acc-mean", "NRGBD Acc"), ("recon", "nrgbd", "Comp-mean", "NRGBD Comp"),
           ("recon", "nrgbd", "NC-mean", "NRGBD NC")]


def load_results(d):
    """{benchmark: {dataset: {"avg": ..., "per_seq": ...}}} from the result files of the evaluation scripts."""
    out = {}
    for bench, ds, _, _ in COLUMNS:
        f = f"{d}/{bench}/{ds}.json"
        if os.path.exists(f) and ds not in out.get(bench, {}):
            r = json.load(open(f))
            if bench == "tae":
                r = dict(avg={"TAE": r["tae"], "n_seq": r["n_seq"]}, per_seq={s: {"TAE": v["tae"]} for s, v in r["per_seq"].items()})
            out.setdefault(bench, {})[ds] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--compare-paper", action="store_true")
    ap.add_argument("--reference", default=os.path.join(HERE, "reference", "paper_vg3r.json"))
    a = ap.parse_args()
    res = load_results(a.results)

    def val(r, bench, ds, key):
        v = r.get(bench, {}).get(ds)
        return None if v is None else v["avg"][key]

    print("| " + " | ".join(c[3] for c in COLUMNS) + " |")
    print("|" + "---:|" * len(COLUMNS))
    print("| " + " | ".join("-" if val(res, *c[:3]) is None else f"{val(res, *c[:3]):.3f}" for c in COLUMNS) + " |")
    if not a.compare_paper:
        return
    ref = json.load(open(a.reference))
    print(f"\n{'column':16s} {'paper':>10s} {'this run':>10s} {'difference':>11s}   largest per-sequence difference")
    for bench, ds, key, label in COLUMNS:
        p, q = val(ref, bench, ds, key), val(res, bench, ds, key)
        if q is None:
            print(f"{label:16s} {p:10.4f} {'-':>10s}")
            continue
        rp, rq = ref[bench][ds]["per_seq"], res[bench][ds]["per_seq"]
        common = sorted(set(rp) & set(rq))
        worst = max(((abs(rq[s][key] - rp[s][key]), s) for s in common), default=(float("nan"), "-"))
        extra = "" if set(rp) == set(rq) else f"; sequences differ: {len(rp)} in the paper, {len(rq)} here"
        print(f"{label:16s} {p:10.4f} {q:10.4f} {q - p:+11.2e}   {worst[0]:.2e} ({worst[1]}){extra}")


if __name__ == "__main__":
    main()
