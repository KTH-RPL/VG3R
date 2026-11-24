"""Unit tests of the geometry and output code (CPU only, no weights)."""
import numpy as np

from vg3r import inference as I
from vg3r.export import depth_edges, write_ply


def test_target_hw():
    for H, W in ((480, 640), (436, 1024), (375, 1242), (1080, 1920)):
        h, w = I.target_hw(H, W)
        assert h % 16 == 0 and w % 16 == 0
        assert abs(h * w / I.BUDGET - 1) < 0.1 and abs((w / h) / (W / H) - 1) < 0.06


def test_window_starts():
    assert I.window_starts(37) == [0]
    assert I.window_starts(50) == [0, 13]
    assert I.window_starts(90) == [0, 24, 48, 53]
    for N in range(37, 300):
        s = I.window_starts(N)
        assert s[0] == 0 and s[-1] + I.WINDOW == N and all(b - a <= I.STRIDE for a, b in zip(s, s[1:]))


def _rigid(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = q * np.sign(np.linalg.det(q)), rng.normal(size=3)
    return T


def test_chain_cameras_recovers_a_trajectory():
    """Windows that see the same trajectory, each in its own frame, chain back to it."""
    rng = np.random.default_rng(0)
    N, w = 90, I.WINDOW
    C = np.stack([np.eye(4)] + [_rigid(rng) for _ in range(N - 1)])          # camera-to-world, frame 0 = identity
    wins = []
    for a in I.window_starts(N):
        Ew = np.linalg.inv(C[a:a + w])                                       # world-to-camera of the window's frames
        wins.append((a, None, Ew @ C[a]))                                    # relative to the window's first frame
    np.testing.assert_allclose(I.chain_cameras(wins, w, [1.0] * len(wins)), C, atol=1e-9)


def test_chain_depth_keeps_consistent_windows():
    rng = np.random.default_rng(1)
    D1 = rng.uniform(0.5, 5.0, size=(90, 4, 5))
    wins = [(a, D1[a:a + I.WINDOW] * (1.0 + a), None) for a in I.window_starts(90)]
    D, scales = I.chain_depth(D1, wins, I.WINDOW)
    np.testing.assert_allclose(D, D1, rtol=1e-12)
    np.testing.assert_allclose(scales, [1.0 / (1.0 + a) for a, _, _ in wins], rtol=1e-12)


def test_unproject_projects_back():
    rng = np.random.default_rng(2)
    h, w = 6, 8
    K = np.array([[7.0, 0, w / 2], [0, 7.0, h / 2], [0, 0, 1]])
    C = _rigid(rng)
    D = rng.uniform(1, 3, size=(1, h, w)).astype(np.float32)
    X = I.unproject(D, K[None], C[None])[0].reshape(-1, 3).astype(np.float64)
    Xc = (X - C[:3, 3]) @ C[:3, :3]
    uv = Xc @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    np.testing.assert_allclose(uv, np.stack([u, v], -1).reshape(-1, 2), atol=1e-4)
    np.testing.assert_allclose(Xc[:, 2], D.reshape(-1), rtol=1e-6)


def test_camera_matrices_identity():
    import torch
    cams = torch.zeros(1, 2, 9)
    cams[..., 6] = 1.0                                                      # quaternion xyzw = identity
    cams[0, 1, :3] = torch.tensor([1.0, 2.0, 3.0])
    E = I.camera_matrices(cams)[0].numpy()
    np.testing.assert_allclose(E[0], np.eye(4), atol=1e-7)
    np.testing.assert_allclose(E[1][:3, 3], [1, 2, 3], atol=1e-7)


def test_depth_edges_and_ply(tmp_path):
    D = np.ones((1, 4, 6), np.float32)
    D[0, :, 3:] = 2.0
    E = depth_edges(D)
    assert E[0, :, 2].all() and E[0, :, 3].all() and not E[0, :, 1].any()
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    rgb = np.arange(12, dtype=np.uint8).reshape(4, 3)
    write_ply(tmp_path / "p.ply", xyz, rgb)
    raw = (tmp_path / "p.ply").read_bytes()
    head, body = raw.split(b"end_header\n")
    assert b"element vertex 4" in head and len(body) == 4 * 15
