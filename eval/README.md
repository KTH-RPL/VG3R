# Evaluation: the VG3R row of Table 1

This directory reproduces the numbers of VG3R in the main comparison table of the paper (video depth, temporal consistency, camera
trajectories and reconstruction from video) from the original datasets:

```
eval/
  run_videodepth.py         video depth on Sintel, Bonn and KITTI (AbsRel, delta < 1.25)
  run_pose.py               camera trajectories on Sintel, TUM-dynamics and ScanNet (ATE, RPE-t, RPE-r)
  run_tae.py                temporal consistency of the depth on ScanNet (TAE)
  run_recon.py              reconstruction from video on 7-Scenes and Neural RGB-D (Acc, Comp, NC)
  make_table.py             the table row from the results, and the difference to the paper (--compare-paper)
  common.py                 shared helpers
  reference/paper_vg3r.json the paper's numbers of VG3R, per dataset and per sequence
  third_party/pi3_eval/     the parts of the evaluation code of pi^3 that the scripts run (unchanged; its license)
  third_party/vda/          the TAE code of Video Depth Anything (unchanged; Apache-2.0)
  tests/                    checks of the metric code and of the reference file
```

No dataset image, depth map or pose is shipped. Every benchmark uses the sequences, ground truth and metric code of the evaluation
code of pi^3, which the scripts load from `third_party/pi3_eval` (see its `NOTICE.md`). VG3R takes every sequence as one video,
through `vg3r.predict` (one forward pass for up to 37 frames, windows of 37 frames with a stride of 24 beyond).

## The benchmarks

| benchmark | dataset | sequences | frames | metrics | alignment |
|---|---|---|---|---|---|
| video depth | Sintel (final pass) | 14 | 20 to 50, all | AbsRel, delta < 1.25 | one scale per sequence |
| | Bonn | 5 | 110 | | |
| | KITTI | 13 | 67 to 110 | | |
| temporal consistency | ScanNet | 100 (94 scored) | 90, stride 3 | TAE | one scale and shift per clip, in disparity |
| cameras | Sintel | 14 | all | ATE, RPE-t, RPE-r | similarity transform per sequence |
| | TUM-dynamics | 8 | 90, stride 3 | | |
| | ScanNet | 100 (94 scored) | 90, stride 3 | | |
| reconstruction | 7-Scenes | 18 (test split) | first 90, stride 3 | Acc, Comp, NC | similarity transform + ICP per sequence |
| | Neural RGB-D | 9 | first 90, stride 3 | | |

* **Video depth** (`run_videodepth.py`). VG3R predicts depth up to one scale per video. The predicted depth is resized to the ground
  truth (bicubic) and scaled by the pi^3 code's own fit (`depth_evaluation(align_with_scale=True)`: 10 Weiszfeld iterations of the
  L1 error) on the valid ground truth (depth > 0, and below 70 m on Sintel and Bonn; Sintel predictions are clipped at 70 m).
  Metrics of each dataset are averaged over its sequences, weighted by their valid pixels.
* **Temporal consistency** (`run_tae.py`). TAE of Video Depth Anything on the ScanNet clips of the camera benchmark, from the depth
  that `run_pose.py infer` keeps: the depth is resized (bilinear) to the ground truth and cropped like it ([8:-8, 11:-11]), one scale
  and shift per clip is fitted in disparity to the ground truth in (1e-3, 10 m), and every pair of consecutive frames is scored in
  both directions (reprojection with the ground-truth intrinsics and relative pose, AbsRel). TAE = 100 x the mean over the pairs;
  the dataset value is the mean over clips. Lower is better.
* **Cameras** (`run_pose.py`). The trajectory metrics of the pi^3 code (after CUT3R, with evo): the predicted camera positions are
  aligned to the ground truth by a similarity transform; ATE is the RMSE of the aligned positions, RPE-t and RPE-r the RMSE of the
  relative translation and rotation (degrees) between consecutive frames; each is averaged over the sequences of a dataset.
* **Reconstruction** (`run_recon.py`). The scenes, loaders and ground-truth point maps of the pi^3 code for 7-Scenes and Neural RGB-D
  (518 pixels wide), with consecutive frames: the first 90 frames at a stride of 3 of every sequence (the sampling of the TUM-dynamics
  and ScanNet camera benchmarks) instead of keyframes 40 to 500 frames apart. VG3R's points are resized to the ground-truth grid
  (bilinear, antialiased) and scored on every second pixel row and column: Umeyama on the valid pixels, ICP (point to point,
  threshold 0.1 m), normals, then accuracy and completeness (mean nearest-neighbor distance, m) and normal consistency (the mean of
  its two directions), averaged over the sequences.
* **Skipped sequences.** 6 of the 100 ScanNet clips have non-finite ground-truth poses; the pi^3 code skips them in the camera metrics,
  and TAE skips them as well.

## Setup

```bash
pip install -e .                       # the vg3r package (from the code folder)
pip install -r eval/requirements.txt   # metrics: open3d, evo, rootutils, opencv, ...
```

The weights go to `weights/` as described in the main README. Inference needs a GPU (about 20 GB of memory for a 110-frame video);
the video-depth metrics run on CUDA as in the pi^3 code; the trajectory, TAE and reconstruction metrics run on the CPU. evo writes
its settings to `~/.evo` the first time it is imported.

## 1. Datasets

Prepare the datasets as the evaluation code of pi^3 does, and put them under one folder (`--data-root`, `data/` below):

| folder | dataset | preparation | layout read by the scripts |
|---|---|---|---|
| `sintel/` | MPI Sintel, training set | MonST3R (`data/evaluation_script.md` of its repository) | `training/final/<seq>/frame_XXXX.png`, `training/depth/<seq>/frame_XXXX.dpt`, `training/camdata_left/<seq>/frame_XXXX.cam` |
| `bonn/` | Bonn RGB-D Dynamic | MonST3R (110 frames per sequence) | `rgbd_bonn_dataset/<seq>/rgb_110/*.png`, `.../depth_110/*.png` |
| `kitti/` | KITTI depth (validation selection) | MonST3R (up to 110 frames per sequence) | `depth_selection/val_selection_cropped/image_gathered/<seq>/*.png`, `.../groundtruth_depth_gathered/<seq>/*.png` |
| `tum/` | TUM RGB-D, the 8 freiburg3 sitting / walking sequences | MonST3R (90 frames at a stride of 3) | `<seq>/rgb_90/*.png`, `<seq>/groundtruth_90.txt` |
| `scannetv2/` | ScanNet v2, 100 test scenes | MonST3R (90 frames at a stride of 3) | `<scene>/color_90/*.jpg`, `<scene>/depth_90/*.png`, `<scene>/pose_90.txt`, `<scene>/intrinsic/intrinsic_depth.txt` |
| `7scenes/` | 7-Scenes | Spann3R (`docs/data_preprocess.md` of its repository) | `<scene>/TestSplit.txt`, `<scene>/seq-XX/frame-XXXXXX.{color.png,depth.proj.png,pose.txt}` |
| `nrgbd/` | Neural RGB-D | Spann3R | `<scene>/images/img*.png`, `<scene>/depth/depth*.png`, `<scene>/poses.txt` |

The folders `tum/`, `scannetv2/`, `7scenes/` and `nrgbd/` must hold only the sequence or scene folders: the loaders take every
subfolder as one. Obtain each dataset under its own terms of use.

## 2. Run

```bash
python eval/run_videodepth.py infer   --data-root data --weights weights --out results
python eval/run_videodepth.py metrics --data-root data --out results
python eval/run_pose.py infer         --data-root data --weights weights --out results
python eval/run_pose.py metrics       --data-root data --out results
python eval/run_tae.py                --data-root data --out results       # after run_pose.py infer
python eval/run_recon.py infer        --data-root data --weights weights --out results
python eval/run_recon.py metrics      --out results
python eval/make_table.py --results results --compare-paper
```

`infer` skips sequences whose outputs exist, so an interrupted run resumes; `--datasets` and `--max-seqs` restrict a run. Results
are written to `results/<benchmark>/<dataset>.json` (dataset averages and per-sequence values), next to the predictions.

## 3. Results

RESULTS_PLACEHOLDER

## Licenses

The scripts are under the license of the code folder. `third_party/pi3_eval` is under the license of the pi^3 evaluation code
("Pi3 for non-commercial purposes", `third_party/pi3_eval/LICENSE`) and `third_party/vda` under Apache-2.0
(`third_party/vda/LICENSE`).
