# Third-party notices

VG3R is built on the following work. Their licenses apply to the corresponding parts of this repository and of the weights it
loads; everything else is under the MIT License (`LICENSE`).

## DVD and DiffSynth-Studio (Apache License 2.0)

VG3R's backbone is the diffusion transformer of DVD, a deterministic video depth model built on Wan2.1, with its LoRA fine-tuned by us.

- `vg3r/_dvd/` is a copy of DVD (commit 62f52d16): its `diffsynth` package (DVD's modified copy of DiffSynth-Studio), its model
  configuration and the two modules that build the model, `diffusion_training_module.py` and `wan_training_module.py`. In the latter
  the absolute imports are made relative, and five comments are removed from three files of `diffsynth`; nothing else is changed
  (`vg3r/_dvd/NOTICE.md`).
- License: Apache License 2.0, in `vg3r/_dvd/LICENSE`. DiffSynth-Studio is also released under the Apache License 2.0.

## Wan2.1 (Apache License 2.0)

The network architecture of the DiT and the VAE in `vg3r/_dvd/diffsynth/models/` follows Wan2.1, and VG3R loads the
Wan2.1-T2V-1.3B weights (released under the Apache License 2.0) for its VAE and its DiT.

## VGGT (VGGT License)

LayerTap ends in the trunk of VGGT's camera head.

- `vg3r/_vggt/` is a copy of VGGT (commit 44b3afbd): `heads/camera_head.py`, `heads/head_act.py`, the files of `layers/` they import,
  and `utils/rotation.py`, with the imports renamed (`vg3r/_vggt/NOTICE.md`).
- Copyright (c) Meta Platforms, Inc. and affiliates. These files are distributed under the VGGT License, a copy of which is in
  `vg3r/_vggt/LICENSE.txt`; anyone who receives them receives them under that license.

## Evaluation code of π³ (license for non-commercial purposes)

The evaluation scripts in `eval/` run parts of the evaluation code of π³.

- `eval/third_party/pi3_eval/` holds files of that code (commit 97ce97b8), copied unchanged (`eval/third_party/pi3_eval/NOTICE.md`).
- License: "Pi3 for non-commercial purposes", in `eval/third_party/pi3_eval/LICENSE`, whose copyright notice and conditions are kept.

## Video Depth Anything (Apache License 2.0)

- `eval/third_party/vda/eval_tae.py` is the TAE code of Video Depth Anything (commit 4f5ae231), unchanged
  (`eval/third_party/vda/NOTICE.md`).
- License: Apache License 2.0, in `eval/third_party/vda/LICENSE`.

## Weights

VG3R loads three sets of weights; none of them is in this folder (see README.md):

| Weights | Source | License |
|---|---|---|
| Wan2.1-T2V-1.3B (VAE and DiT) | Wan-AI | Apache License 2.0 |
| DVD v1.1 (`dvd_1.1.safetensors`) | DVD | CC BY-NC 4.0 |
| VG3R (`vg3r.safetensors`: our LoRA, fine-tuned from that of DVD v1.1, and our heads) | huggingface.co/Jingong/VG3R | CC BY-NC 4.0, for non-commercial research |

The VG3R weights are trained on the datasets listed in the paper, several of which allow only non-commercial research use.

## Example data

The frames in `examples/` come from public datasets and keep their licenses (each folder has a `SOURCE.md` and the text of its
license, `LICENSE.txt`):

| Folder | Source | License |
|---|---|---|
| `examples/tum_walking_xyz` | TUM RGB-D, freiburg3_walking_xyz | CC BY 4.0 |
| `examples/kitti_0926_0002` | KITTI raw, 2011_09_26_drive_0002 | CC BY-NC-SA 3.0 |
