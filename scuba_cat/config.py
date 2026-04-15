from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5

    window_seconds: float = 2.0
    pose_threshold: float = 0.25
    state_min_frames: int = 3
    state_timeout_s: float = 1.5
    cooldown_s: float = 3.0

    overlay_duration_s: float = 4.0
    gif_path: Path = ASSETS / "scuba_cat.gif"
    reference_path: Path = ASSETS / "reference_dance.json"


CFG = Config()
