"""End-to-end test on the example videos: the predictions against reference outputs of the model of the paper.

    pytest tests/test_inference.py [--weights weights]       (needs the weights and a GPU)

tests/reference_outputs.json holds, per example, the per-frame median depth, the camera centers, the field of view and the shift of
the model of the paper on the frames shipped in examples/. On the configuration of the paper (NVIDIA GH200, PyTorch 2.11, CUDA 12.8)
the predictions match them exactly; other GPUs and versions change the bfloat16 kernels, so the test allows small differences.
"""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = json.load(open(f"{HERE}/reference_outputs.json"))


def summary(pred):
    """The compact form of a prediction stored in reference_outputs.json."""
    K = pred["K"][0]
    h, w = pred["hw"]
    return dict(hw=[int(h), int(w)], windows=[int(a) for a in pred["windows"]],
                depth_median=[float(np.median(d)) for d in pred["depth"]],
                centers=pred["c2w"][:, :3, 3].tolist(),
                fov_deg=[float(np.degrees(2 * np.arctan(h / 2 / K[1, 1]))), float(np.degrees(2 * np.arctan(w / 2 / K[0, 0])))],
                shift=float(pred["shift"]))


@pytest.fixture(scope="module")
def model(request):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    weights = request.config.getoption("--weights")
    if not os.path.exists(f"{weights}/vg3r.safetensors"):
        pytest.skip(f"no weights in {weights}")
    from vg3r import VG3R
    return VG3R.from_pretrained(weights, "cuda")


@pytest.mark.parametrize("name", sorted(REF["examples"]))
def test_example(model, name):
    from vg3r import predict
    from vg3r.io import read_frames
    ref = REF["examples"][name]
    frames = read_frames(f"{ROOT}/examples/{name}/frames")
    got = summary(predict(model, frames))
    assert got["hw"] == ref["hw"] and got["windows"] == ref["windows"]
    d, rd = np.array(got["depth_median"]), np.array(ref["depth_median"])
    c, rc = np.array(got["centers"]), np.array(ref["centers"])
    exact = np.array_equal(d, rd) and np.array_equal(c, rc)
    print(f"{name}: {'identical to the reference' if exact else 'close to the reference'}; max |depth median| difference "
          f"{np.abs(d - rd).max():.2e}, max camera center difference {np.abs(c - rc).max():.2e}")
    np.testing.assert_allclose(d, rd, rtol=2e-2)                       # depth: unit = the median depth of frame 0
    np.testing.assert_allclose(c, rc, atol=2e-2 * max(1.0, np.abs(rc).max()))
    np.testing.assert_allclose(got["fov_deg"], ref["fov_deg"], atol=0.5)
    np.testing.assert_allclose(got["shift"], ref["shift"], rtol=2e-2)
