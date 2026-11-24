"""VG3R: video generation model for feed-forward 3D reconstruction."""
from .inference import predict
from .model import VG3R

__version__ = "1.0.0"
__all__ = ["VG3R", "predict"]
