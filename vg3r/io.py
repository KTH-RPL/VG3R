"""Reading videos and frame folders."""
import glob
import os

import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_frames(folder):
    """Image files of a folder, sorted by name."""
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(IMAGE_EXT))
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    return files


def read_frames(path, stride=1, max_frames=None, start=0):
    """A video file or a folder of images => uint8 [N, H, W, 3] RGB, every `stride`-th frame from `start`, at most `max_frames`."""
    from PIL import Image
    if os.path.isdir(path):
        files = list_frames(path)[start::stride][:max_frames]
        frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    else:
        import imageio.v3 as iio
        frames = []
        for i, f in enumerate(iio.imiter(path)):
            if i < start or (i - start) % stride:
                continue
            frames.append(np.asarray(f)[..., :3])
            if max_frames and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"no frames read from {path}")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"frames of {path} differ in size: {sorted(shapes)[:3]}")
    return np.stack(frames).astype(np.uint8)
