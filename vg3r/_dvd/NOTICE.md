# DVD (Apache-2.0)

Copied from DVD, commit 62f52d16faa2cac10e31eb7815ece6e61d5449ec: `diffsynth/` (DVD's modified copy of DiffSynth-Studio, also Apache-2.0), `ckpt/model_config.yaml` (as `model_config.yaml`) and `examples/wanvideo/model_training/DiffusionTrainingModule.py` and `WanTrainingModule.py` (as `diffusion_training_module.py` and `wan_training_module.py`).

Changes: in `wan_training_module.py` the absolute imports of `diffsynth` and of the DiffusionTrainingModule are made relative, and five comments are removed, one from `diffsynth/data/video.py`, three from `diffsynth/pipelines/wan_video_new_determine.py` and one from `diffsynth/util/metric.py`. No other change. The license is in `LICENSE`.
