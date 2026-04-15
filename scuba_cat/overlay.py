import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

class GifOverlay:
    def __init__(
        self,
        gif_path: Path,
        duration_s: float = 4.0,
        scale: float = 0.4,
        remove_chroma: bool = True,
    ):
        self.frames = self._load_gif(gif_path, remove_chroma=remove_chroma)
        self.duration = duration_s
        self.scale = scale
        self._start = None

    @classmethod
    def _load_gif(cls, path: Path, remove_chroma: bool):
        img = Image.open(path)
        raw_frames = []
        for frame in ImageSequence.Iterator(img):
            rgba = frame.convert("RGBA")
            arr = np.array(rgba, dtype=np.uint8)
            bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
            alpha = arr[..., 3:4]
            raw_frames.append(np.concatenate([bgr, alpha], axis=-1))
        if not raw_frames:
            raise ValueError(f"No frames decoded from {path}")

        if not remove_chroma:
            return raw_frames

        bg_hsv = cls._detect_background_hsv(raw_frames[0])
        return [cls._apply_chroma_key(f, bg_hsv) for f in raw_frames]

    @staticmethod
    def _detect_background_hsv(bgra: np.ndarray) -> np.ndarray:
        h, w = bgra.shape[:2]
        ph = max(h // 20, 1)
        pw = max(w // 20, 1)
        patches = [
            bgra[:ph, :pw, :3],
            bgra[:ph, -pw:, :3],
            bgra[-ph:, :pw, :3],
            bgra[-ph:, -pw:, :3],
        ]
        pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
        median_bgr = np.median(pixels, axis=0).astype(np.uint8).reshape(1, 1, 3)
        return cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)[0, 0]

    @staticmethod
    def _apply_chroma_key(
        bgra: np.ndarray,
        bg_hsv: np.ndarray,
        hue_tol: int = 25,
        sat_min: int = 40,
        val_min: int = 40,
    ) -> np.ndarray:
        bgr = bgra[..., :3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0].astype(np.int16)
        s = hsv[..., 1]
        v = hsv[..., 2]

        hue_diff = np.abs(h - int(bg_hsv[0]))
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        mask = (hue_diff <= hue_tol) & (s >= sat_min) & (v >= val_min)

        mask_u8 = (mask.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.GaussianBlur(mask_u8, (5, 5), 0)

        alpha = bgra[..., 3].astype(np.int16)
        alpha -= mask_u8.astype(np.int16)
        np.clip(alpha, 0, 255, out=alpha)

        out = bgra.copy()
        out[..., 3] = alpha.astype(np.uint8)
        return out

    def trigger(self):
        self._start = time.time()

    @property
    def active(self) -> bool:
        return self._start is not None and (time.time() - self._start) < self.duration

    def compose(self, frame_bgr):
        if not self.active:
            return frame_bgr
        elapsed = time.time() - self._start
        gif_fps = max(len(self.frames) / max(self.duration, 0.1), 1.0)
        idx = int(elapsed * gif_fps) % len(self.frames)
        sprite = self.frames[idx]

        h, w = frame_bgr.shape[:2]
        sh, sw = sprite.shape[:2]
        target_w = max(int(w * self.scale), 1)
        target_h = max(int(sh * target_w / sw), 1)
        sprite = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)

        x = (w - target_w) // 2
        y = max(h - target_h - 20, 0)
        x2 = min(x + target_w, w)
        y2 = min(y + target_h, h)
        sprite = sprite[: y2 - y, : x2 - x]
        roi = frame_bgr[y:y2, x:x2]

        bgr = sprite[..., :3].astype(np.float32)
        alpha = sprite[..., 3:4].astype(np.float32) / 255.0
        blended = roi.astype(np.float32) * (1.0 - alpha) + bgr * alpha
        frame_bgr[y:y2, x:x2] = blended.astype(np.uint8)
        return frame_bgr
