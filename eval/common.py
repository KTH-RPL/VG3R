"""Shared helpers of the evaluation scripts.

The evaluation code of pi^3 (third_party/pi3_eval) is loaded by path, so that its top-level packages (`utils`, `datasets`) shadow
nothing, and its data configs are pointed to the data root given on the command line (the configs use `data/` as their root).
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PI3 = os.path.join(HERE, "third_party", "pi3_eval")
VDA_TAE = os.path.join(HERE, "third_party", "vda", "eval_tae.py")
CODE = os.path.dirname(HERE)                                   # the code folder, which holds the vg3r package


def by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pi3(rel):
    """A module of the evaluation code of pi^3, e.g. pi3("utils/depth.py")."""
    return by_path("_pi3_" + rel[:-3].replace("/", "_"), os.path.join(PI3, rel))


def data_config(name, data_root):
    """configs/data/<name>.yaml of the pi^3 evaluation code with its `data/` roots moved under data_root."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"data": OmegaConf.load(os.path.join(PI3, "configs", "data", f"{name}.yaml"))})
    for ds in cfg.data:
        rp = cfg.data[ds].get("root_path")
        if rp is not None:
            assert rp.startswith("data/"), rp
            cfg.data[ds].root_path = os.path.join(os.path.abspath(data_root), rp[len("data/"):])
    return cfg.data


def load_model(weights, device="cuda"):
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    from vg3r import VG3R
    return VG3R.from_pretrained(weights, device)


def predict(model, frames):
    from vg3r import predict as _predict
    return _predict(model, frames)


def load_frames(paths):
    from PIL import Image
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def write_csv(path, rows):
    """rows: list of dicts with the same keys."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def info(model_pred, frames, extra=None):
    """What every inference run records next to its outputs."""
    d = dict(frames=int(len(frames)), input_hw=list(frames.shape[1:3]), model_hw=list(model_pred["hw"]),
             seconds=float(model_pred["seconds"]), passes=int(model_pred["passes"]), windows=list(model_pred["windows"]))
    try:
        import torch
        if torch.cuda.is_available():
            d["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    except ImportError:
        pass
    d.update(extra or {})
    return d
