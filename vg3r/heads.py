"""The two heads of VG3R on the tokens of the video diffusion transformer (DiT).

ClipShiftHead   ClipShift: one shift c per clip, read from the final DiT tokens by attention pooling. The decoded affine-invariant
                inverse depth x of the depth branch becomes depth up to one scale per clip, D = 1 / max(x + c, tau).
LayerTapHead    LayerTap: the camera of every RGB frame (translation, rotation as a quaternion, field of view), relative to frame 0.
                It reads the final DiT tokens and the outputs of intermediate DiT blocks (blocks 10 and 20), pairs the tokens of
                every latent frame with those of latent frame 0, and gives each RGB frame its own query (the Wan VAE encodes frame
                0 alone and every following group of four frames into one latent frame). The per-frame queries then go through the
                trunk of VGGT's camera head across all frames, with four refinement iterations.

Both heads run in float32 with autocast off, on float32 copies of the DiT tokens, as the final model was trained.
"""
import math

import torch
import torch.nn as nn

from ._vggt.heads.camera_head import CameraHead
from ._vggt.layers.block import Block

DIM = 1536            # width of the Wan2.1-1.3B DiT
PATCH = (1, 2, 2)     # DiT patch size (t, h, w)


def frames_of_latent(T):
    """RGB frame indices of each latent frame of the causal Wan VAE: [[0], [1..4], [5..8], ...]."""
    return [[0]] + [list(range(4 * t - 3, 4 * t + 1)) for t in range(1, T)]


def n_frames(T):
    """Number of RGB frames of T latent frames."""
    return 1 + 4 * (T - 1)


def sincos(idx, dim):
    """[..., dim] sinusoidal embedding of integer positions."""
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=idx.device, dtype=torch.float32) / half)
    a = idx.float()[..., None] * f
    return torch.cat([a.sin(), a.cos()], dim=-1)


def sincos2d(h, w, dim, device):
    """[h*w, dim] 2-D sinusoidal embedding (half of the channels for rows, half for columns)."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.cat([sincos(yy.flatten(), dim // 2), sincos(xx.flatten(), dim // 2)], -1)


def _float32(fn):
    """Run a head in float32 with autocast off, on float32 copies of its inputs."""
    def wrapped(self, tok, *args, **kw):
        with torch.autocast(device_type=tok.device.type, enabled=False):
            args = tuple([t.float() for t in a] if isinstance(a, (list, tuple)) else a for a in args)
            if kw.get("taps") is not None:
                kw["taps"] = [t.float() for t in kw["taps"]]
            return fn(self, tok.float(), *args, **kw)
    return wrapped


class ClipShiftHead(nn.Module):
    def __init__(self, dim=DIM, hid=256, heads=12):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, 1))

    @_float32
    def forward(self, tok):                                    # tok [B, N, C] => c [B], c > 0
        kv = self.norm(tok)
        o, _ = self.attn(self.q.expand(tok.shape[0], -1, -1), kv, kv, need_weights=False)
        return nn.functional.softplus(self.mlp(o[:, 0]).squeeze(-1)) + 1e-4


class LayerTapHead(nn.Module):
    def __init__(self, dim_in=DIM, dim=768, heads=12, pair_depth=4, trunk_depth=4, iters=4, pool=1, n_taps=2):
        super().__init__()
        self.pool = pool
        self.inp = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim))
        # the intermediate DiT block outputs, each LayerNorm + Linear, summed with the projection of the final tokens
        self.inp_taps = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim)) for _ in range(n_taps)])
        self.ref_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.cur_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.q0 = nn.Parameter(torch.zeros(1, 1, dim))
        self.qs = nn.Parameter(torch.zeros(1, 4, dim))
        for t in (self.ref_emb, self.cur_emb, self.q0, self.qs):
            nn.init.normal_(t, std=0.02)
        self.pos = nn.Linear(256, dim)
        self.blocks = nn.ModuleList([Block(dim=dim, num_heads=heads, mlp_ratio=4, init_values=0.01) for _ in range(pair_depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.cam = CameraHead(dim_in=dim, trunk_depth=trunk_depth, num_heads=heads)
        self.iters = iters
        self.dim = dim

    @_float32
    def forward(self, tok, T, h, w, taps=()):
        """tok [B, T*h*w, C] final DiT tokens, taps: the intermediate block outputs of the same shape
        => list (one per refinement iteration) of [B, L, 9] cameras: translation, quaternion (xyzw), field of view (y, x)."""
        B, N, C = tok.shape
        D = self.dim
        assert len(taps) == len(self.inp_taps), (len(taps), len(self.inp_taps))
        x = self.inp(tok)
        for m, t in zip(self.inp_taps, taps):
            x = x + m(t)
        x = x.reshape(B * T, h, w, D).permute(0, 3, 1, 2)
        if self.pool > 1:
            x = nn.functional.avg_pool2d(x, self.pool)
        hp, wp = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2) + sincos2d(hp, wp, D, tok.device).to(x.dtype)[None]   # [B*T, P, D]
        x = x.reshape(B, T, hp * wp, D)
        ref = (x[:, :1] + self.ref_emb).expand(B, T, hp * wp, D)          # latent frame 0
        cur = x + self.cur_emb                                             # latent frame t
        fidx = torch.zeros(T, 4, dtype=torch.long, device=tok.device)
        for t, fr in enumerate(frames_of_latent(T)):
            for k, f in enumerate(fr):
                fidx[t, k] = f
        q = torch.cat([self.q0.expand(1, 4, D), self.qs.expand(T - 1, 4, D)], 0) if T > 1 else self.q0.expand(1, 4, D)
        q = (q + self.pos(sincos(fidx, 256)))[None].expand(B, T, 4, D)     # one query per RGB frame of latent t
        seq = torch.cat([q, ref, cur], 2)
        seq = seq.reshape(B * T, seq.shape[2], D)
        for blk in self.blocks:
            seq = blk(seq)
        o = self.out_norm(seq[:, :4]).reshape(B, T, 4, D)
        L = n_frames(T)
        sel = [o[:, 0, 0]] + [o[:, t, k] for t in range(1, T) for k in range(4)]
        ftok = torch.stack(sel, 1) + self.pos(sincos(torch.arange(L, device=tok.device), 256))[None]
        return self.cam.trunk_fn(self.cam.token_norm(ftok), self.iters)
