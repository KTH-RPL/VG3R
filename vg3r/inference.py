"""Video in, depth, cameras and points out: preprocessing, the forward pass and the inference on videos longer than a training clip.

    pred = predict(model, frames)        frames: uint8 [N, H, W, 3] RGB in temporal order, any N >= 1

pred["depth"]   [N, h, w] float32   depth up to one scale per video (the median depth of frame 0 is 1)
pred["c2w"]     [N, 4, 4] float64   camera-to-world, OpenCV axes (x right, y down, z forward); world = the camera of frame 0
pred["K"]       [N, 3, 3] float64   pinhole intrinsics at (h, w), principal point at the image center
pred["points"]  [N, h, w, 3]        world points, the depth back-projected by K and moved by c2w
pred["hw"]      (h, w)              the model's input size: the frames resized to about 288 x 512 pixels, sides multiples of 16

Long videos (as in the paper): a video of up to 37 frames (the length of the training clips) takes one forward pass. A longer video is
processed both as a whole and in windows of 37 frames with a stride of 24, the last window ending with the video. The depth of each
window is scaled to that of the whole video by the median log ratio and blended linearly in log depth where windows overlap. The
cameras of each window are attached to those already placed by the rigid transform averaged over the shared frames.
"""
import math
import time

import numpy as np
import torch

from ._vggt.utils.rotation import quat_to_mat

BUDGET = 288 * 512           # pixels of the model input
PATCH = 16                   # VAE (8) x DiT patch (2)
WINDOW, STRIDE = 37, 24
TAU = 1e-3                   # lower bound of x + c, the inverse depth before scaling


def target_hw(h, w, budget_px=BUDGET, patch=PATCH):
    """Aspect-preserving size of about budget_px pixels, sides multiples of patch."""
    s = math.sqrt(budget_px / (h * w))
    return max(1, round(h * s / patch)) * patch, max(1, round(w * s / patch)) * patch


def resize(frames, hw):
    from PIL import Image
    H, W = hw
    return np.stack([np.asarray(Image.fromarray(f).resize((W, H), Image.BICUBIC)) for f in frames])


def unproject(depth, K, c2w):
    """[N, h, w] z-depth, [N, 3, 3] K, [N, 4, 4] c2w => [N, h, w, 3] world points (float64 maths, pixel centers)."""
    N, h, w = depth.shape
    v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    out = np.empty((N, h, w, 3), np.float32)
    for i in range(N):
        ray = pix @ np.linalg.inv(K[i]).T
        pc = ray * depth[i].reshape(-1, 1).astype(np.float64)
        out[i] = (pc @ c2w[i][:3, :3].T + c2w[i][:3, 3]).reshape(h, w, 3)
    return out


def camera_matrices(cams_last):
    """[B, L, 9] cameras (translation, quaternion xyzw, field of view) => [B, L, 4, 4] world-to-camera matrices."""
    B, L = cams_last.shape[:2]
    E = torch.eye(4, device=cams_last.device).repeat(B, L, 1, 1)
    E[..., :3, :3] = quat_to_mat(cams_last[..., 3:7])
    E[..., :3, 3] = cams_last[..., :3]
    return E


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def forward_pass(model, x):
    """One forward pass on resized frames x [n, h, w, 3] (padded to 4k + 1 frames by repeating the last one, cut back to n)
    => depth 1 / max(x + c, tau) [n, h, w] (not yet normalised), world-to-camera relative to frame 0 [n, 4, 4], the field of view
    of frame 0 (y, x), the shift c and the seconds of the pass."""
    n = len(x)
    L = 1 + 4 * math.ceil((n - 1) / 4)
    if L > n:
        x = np.concatenate([x, np.repeat(x[-1:], L - n, 0)], 0)
    dev = next(model.cam.parameters()).device
    rgb = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div(255.0)[None].to(dev)
    with torch.no_grad():
        _sync()
        t0 = time.time()
        latent, cams, c = model(rgb)
        xh = model.decode(latent)[:n]
        _sync()
        sec = time.time() - t0
        cl = cams[-1].float()                                          # the last refinement iteration
        E = camera_matrices(cl)[0].double().cpu().numpy()[:n]
        E = E @ np.linalg.inv(E[0])
        fov = cl[0, 0, 7:].double().cpu().numpy()
    return 1.0 / np.maximum(xh + float(c[0]), TAU), E, fov, float(c[0]), sec


def window_starts(N, w=WINDOW, stride=STRIDE):
    """A window every `stride` frames, the last one aligned to the end."""
    starts = list(range(0, N - w + 1, stride))
    if starts[-1] + w < N:
        starts.append(N - w)
    return starts


def _avg_rigid(Ts):
    """Average of rigid transforms [n, 4, 4]: rotation = SVD projection of the sum, translation = mean."""
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].sum(0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, Ts[:, :3, 3].mean(0)
    return T


def chain_cameras(wins, w, units):
    """Window passes [(start, depth, w2c)] => camera-to-world [N, 4, 4] (frame 0 = identity). The translations of window k are
    multiplied by units[k]; window k + 1 is attached to the frames already placed by the rigid transform averaged over the overlap."""
    N = wins[-1][0] + w
    C = [None] * N
    for (a, _, Ew), u in zip(wins, units):
        Ew = np.asarray(Ew, np.float64)
        Ew = Ew @ np.linalg.inv(Ew[0])
        cw = np.linalg.inv(Ew)
        cw[:, :3, 3] *= u
        if a == 0:
            T = np.eye(4)
        else:
            ov = [k for k in range(w) if C[a + k] is not None]
            T = _avg_rigid(np.stack([C[a + k] @ np.linalg.inv(cw[k]) for k in ov]))
        for k in range(w):
            if C[a + k] is None:
                C[a + k] = T @ cw[k]
    return np.stack(C)


def chain_depth(D1, wins, w):
    """Depth of the whole video [N, h, w] in the units of the one pass D1: each window scaled to D1 over its frames by the median
    log ratio, overlapping frames blended linearly in log depth from the earlier window to the later one. => depth, scales."""
    lD1 = np.log(np.maximum(D1, 1e-8))
    lg = np.zeros(D1.shape)
    placed = np.zeros(len(D1), bool)
    scales = []
    for a, Dw, _ in wins:
        lw = np.log(np.maximum(np.asarray(Dw, np.float64), 1e-8))
        ov = [k for k in range(w) if placed[a + k]]
        s = float(np.median(lD1[a:a + w] - lw))
        lw = lw + s
        scales.append(float(np.exp(s)))
        for j, k in enumerate(ov):
            t = (j + 1) / (len(ov) + 1)
            lg[a + k] = (1 - t) * lg[a + k] + t * lw[k]
        for k in range(w):
            if not placed[a + k]:
                lg[a + k], placed[a + k] = lw[k], True
    return np.exp(lg), scales


def predict(model, frames):
    """frames: uint8 [N, H, W, 3] RGB at their original resolution, in temporal order. See the module docstring."""
    frames = np.asarray(frames)
    assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3, (frames.dtype, frames.shape)
    torch.backends.cuda.matmul.allow_tf32 = False                     # float32 matmuls in full precision (the heads)
    N, H0, W0 = frames.shape[:3]
    h, w = target_hw(H0, W0)
    x = resize(frames, (h, w))
    log = []                                                           # (fov of frame 0, c, seconds) of every pass

    def run(xx):
        D_, E_, fov_, c_, s_ = forward_pass(model, xx)
        log.append((fov_, c_, s_))
        return D_, E_

    D1, E1 = run(x)
    d_med0 = float(np.median(D1[0]))
    D1 = D1 / d_med0
    if N <= WINDOW:
        D, C, starts, fovs = D1, np.linalg.inv(E1), [0], [f for f, _, _ in log]
    else:
        starts = window_starts(N)
        wins = [(a, *run(x[a:a + WINDOW])) for a in starts]
        D, sc = chain_depth(D1, wins, WINDOW)
        C = chain_cameras(wins, WINDOW, [s * float(np.median(Dw[0])) for s, (_, Dw, _) in zip(sc, wins)])
        fovs = [f for f, _, _ in log[1:]]
    fv = np.median(np.stack(fovs), 0)
    fy, fx = (h / 2) / math.tan(max(fv[0], 1e-3) / 2), (w / 2) / math.tan(max(fv[1], 1e-3) / 2)
    K = np.tile(np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]]), (N, 1, 1))
    D = D.astype(np.float32)
    return dict(depth=D, c2w=C, K=K, points=unproject(D, K, C), hw=(h, w), shift=log[0][1], windows=[int(a) for a in starts],
                seconds=float(sum(s for _, _, s in log)), passes=len(log))
