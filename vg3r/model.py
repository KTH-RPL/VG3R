"""VG3R: a video generation model turned into a feed-forward 3D reconstruction model.

The backbone is the Wan2.1-T2V-1.3B DiT with the LoRA of DVD v1.1 (a deterministic video depth regressor built on Wan2.1), fine-tuned
by us. It maps the latent of the input video, at the fixed timestep of DVD, to the latent of an affine-invariant inverse depth video,
which the Wan VAE decodes. Two heads read the DiT tokens: ClipShift (one shift per clip, which makes the inverse depth affine up to a
scale only) and LayerTap (the camera of every frame). See heads.py.

Weights (directory `weights/` by default):
    Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors, Wan2.1_VAE.pth   from the Wan2.1-T2V-1.3B release
    DVD/dvd_1.1.safetensors                                               from the DVD release (v1.1)
    vg3r.safetensors                                                      our LoRA and heads (huggingface.co/Jingong/VG3R)
The model is built exactly as it was trained: the DiT and the VAE in bfloat16, the 600 LoRA tensors unmerged in float32 (applied
under bfloat16 autocast), the heads in float32.
"""
import contextlib
import glob
import json
import os

import torch
import torch.nn as nn

from .heads import PATCH, ClipShiftHead, LayerTapHead

HERE = os.path.dirname(os.path.abspath(__file__))
N_DIT_TENSORS, N_LORA, N_LORA_PARAMS = 1425, 600, 699_924_480


class _Quiet:
    """WanTrainingModule prints through an accelerator; this one prints nothing."""

    def print(self, *a, **k):
        pass


@contextlib.contextmanager
def _local_tokenizer(wpl, wan_dir):
    """The pipeline would download the tokenizer of the text encoder, which VG3R does not use (the text context is zero)."""
    orig = wpl.WanVideoPipeline.from_pretrained
    wpl.WanVideoPipeline.from_pretrained = staticmethod(
        lambda *a, **k: orig(*a, **{"tokenizer_config": wpl.ModelConfig(path=wan_dir), **k}))
    try:
        yield
    finally:
        wpl.WanVideoPipeline.from_pretrained = orig


def build_backbone(wan_dir, dvd_file):
    """DVD's WanTrainingModule (Wan2.1-T2V-1.3B DiT and VAE, rank-512 LoRA on q, k, v, o, ffn.0, ffn.2) with the DiT of DVD v1.1,
    on the CPU. Attention runs through PyTorch's scaled_dot_product_attention, as for every result of the paper."""
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from ._dvd.diffsynth.models import wan_video_dit as wdit
    wdit.FLASH_ATTN_3_AVAILABLE = wdit.FLASH_ATTN_2_AVAILABLE = wdit.SAGE_ATTN_AVAILABLE = False
    from ._dvd.diffsynth.pipelines import wan_video_new_determine as wpl
    from ._dvd.wan_training_module import WanTrainingModule

    args = OmegaConf.load(f"{HERE}/_dvd/model_config.yaml")
    assert int(args.lora_rank) == 512 and float(args.denoise_step) == 0.5 and args.mode == "regression", args
    dit = sorted(glob.glob(f"{wan_dir}/diffusion_pytorch_model*.safetensors"))
    vae = f"{wan_dir}/Wan2.1_VAE.pth"
    if len(dit) != 1 or not os.path.exists(vae):
        raise FileNotFoundError(f"{wan_dir} must hold diffusion_pytorch_model.safetensors and Wan2.1_VAE.pth of Wan2.1-T2V-1.3B")
    if not os.path.exists(dvd_file):
        raise FileNotFoundError(f"{dvd_file}: the DVD v1.1 weights (dvd_1.1.safetensors)")
    with _local_tokenizer(wpl, wan_dir):
        wtm = WanTrainingModule(accelerator=_Quiet(), model_paths=json.dumps([dit[0], vae]), model_id_with_origin_paths=None,
                                trainable_models=None, use_gradient_checkpointing=False, lora_rank=args.lora_rank,
                                lora_base_model=args.lora_base_model, args=args)
    sd = load_file(dvd_file, device="cpu")
    sd = {k[len("pipe.dit."):]: v for k, v in sd.items() if k.startswith("pipe.dit.")}
    assert len(sd) == N_DIT_TENSORS, f"{dvd_file}: {len(sd)} DiT tensors, expected {N_DIT_TENSORS}"
    wtm.pipe.dit.load_state_dict(sd, strict=True)
    del sd
    lora = []
    for n, p in wtm.named_parameters():
        p.requires_grad = "dit" in n and "lora" in n
        if p.requires_grad:
            p.data = p.data.float()
            lora.append(p)
    assert len(lora) == N_LORA and sum(p.numel() for p in lora) == N_LORA_PARAMS, (len(lora), sum(p.numel() for p in lora))
    assert wtm.pipe.scheduler.timesteps.tolist() == [500.0] and wtm.pipe.scheduler.target == "x"
    assert wtm.pipe.dit.has_image_input is False and wtm.pipe.dit.patch_embedding.in_channels == 16
    return wtm


class VG3R(nn.Module):
    """forward(rgb) on a clip of 4k + 1 frames (rgb float32 [1, L, 3, H, W] in [0, 1], H and W multiples of 16)
    => the predicted depth latent [1, 16, T, H/8, W/8] (float32), the cameras of every refinement iteration (list of [1, L, 9]) and
    the shift c [1]. vg3r.inference.predict turns these into depth, cameras and points for a video of any length."""

    def __init__(self, backbone, taps=(9, 19), iters=4, pool=1):
        super().__init__()
        self.backbone = backbone
        dit = backbone.pipe.dit
        self.taps = tuple(int(i) for i in taps)
        assert all(0 <= i < len(dit.blocks) - 1 for i in self.taps), self.taps
        self.cam = LayerTapHead(iters=iters, n_taps=len(self.taps), pool=pool)
        self.shift = ClipShiftHead()
        self._tok, self._tap, self._capture = None, {}, False
        dit.head.register_forward_pre_hook(self._grab_final)           # input of the DiT output head = final tokens
        for i in self.taps:
            dit.blocks[i].register_forward_hook(self._grab_block(i))

    def _grab_final(self, mod, args):
        self._tok = args[0]                                            # [B, T*h*w, 1536]
        return None

    def _grab_block(self, i):
        def hook(mod, args, out):
            if self._capture:
                assert i not in self._tap, f"block {i} ran twice in one forward"
                self._tap[i] = out
        return hook

    @property
    def pipe(self):
        return self.backbone.pipe

    def forward(self, rgb):
        B, L, H, W = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[-2]), int(rgb.shape[-1])
        # The preprocessing units of DVD's pipeline: VAE encoding of the video (no autocast) and a zero text context. The unused
        # depth input of its training branch gets zeros. Then one DiT forward at t = 500 under bfloat16 autocast.
        data = dict(images=rgb, disparity=torch.zeros_like(rgb), height=H, width=W, num_frames=L, batch_size=B)
        self._tok, self._tap, self._capture = None, {}, True
        try:
            with torch.no_grad():
                inputs = self.backbone.forward_preprocess(data)
            pipe = self.pipe
            models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
            with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16):
                pred = pipe.training_predict(**models, **inputs)["pred"].float()
        finally:
            self._capture = False
        tok, taps = self._tok, [self._tap[i] for i in self.taps]
        T = int(pred.shape[2])
        assert tok is not None and tok.shape[1] == T * (pred.shape[3] // PATCH[1]) * (pred.shape[4] // PATCH[2]), tok
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16):
            cams = self.cam(tok, T, pred.shape[3] // PATCH[1], pred.shape[4] // PATCH[2], taps)
            c = self.shift(tok)
        self._tok, self._tap = None, {}
        return pred, [x.float() for x in cams], c.float()

    def decode(self, latent):
        """Depth latent => the decoded affine-invariant inverse depth x [L, H, W] in [0, 1] (numpy float32), the mean of the three
        channels of the VAE output."""
        pipe = self.pipe
        y = pipe.vae.decode(latent.to(pipe.torch_dtype), device=latent.device, tiled=False)        # [1, 3, L, H, W] in [-1, 1]
        return ((y[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0).mean(-1)

    @classmethod
    def from_pretrained(cls, weights="weights", device="cuda"):
        """Build the model from a weights directory (see the module docstring) and move it to `device`."""
        from safetensors import safe_open
        f = f"{weights}/vg3r.safetensors"
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f}: the VG3R weights, from huggingface.co/Jingong/VG3R (see README.md)")
        with safe_open(f, "pt") as fh:
            meta = json.loads(fh.metadata()["config"])
            lora = {k[len("lora."):]: fh.get_tensor(k) for k in fh.keys() if k.startswith("lora.")}
            heads = {k[len("heads."):]: fh.get_tensor(k) for k in fh.keys() if k.startswith("heads.")}
        backbone = build_backbone(f"{weights}/Wan2.1-T2V-1.3B", f"{weights}/DVD/dvd_1.1.safetensors")
        model = cls(backbone, taps=meta["taps"], iters=meta["iters"], pool=meta["pool"])
        named = dict(backbone.named_parameters())
        want = sorted(n for n, p in named.items() if p.requires_grad)
        assert sorted(lora) == want, f"LoRA tensors of {f} do not match the model ({len(lora)} vs {len(want)})"
        for n, v in lora.items():
            named[n].data.copy_(v.to(named[n].dtype))
        hk = sorted(k for k in model.state_dict() if k.startswith(("cam.", "shift.")))
        assert sorted(heads) == hk, f"head tensors of {f} do not match the model"
        model.load_state_dict(heads, strict=False)
        model = model.to(device)
        for p in list(model.cam.parameters()) + list(model.shift.parameters()):
            p.data = p.data.float()
        backbone.to(device)                                            # also sets the device of DVD's pipeline
        assert str(torch.device(backbone.pipe.device)) == str(torch.device(device)), backbone.pipe.device
        model.config = dict(meta)
        return model.eval()
