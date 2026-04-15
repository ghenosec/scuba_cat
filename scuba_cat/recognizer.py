import time
from collections import deque

import numpy as np


FINGER_TIP_IDS = (4, 8, 12, 16, 20)
FINGER_PIP_IDS = (3, 6, 10, 14, 18)
FINGER_MCP_IDS = (2, 5, 9, 13, 17)


def is_fist(hand: np.ndarray, tolerance: float = 1.1) -> bool:
    """Return True if the 4 non-thumb fingers look curled."""
    wrist = hand[0]
    folded = 0
    for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
        tip_d = float(np.linalg.norm(hand[tip] - wrist))
        pip_d = float(np.linalg.norm(hand[pip] - wrist))
        if tip_d < pip_d * tolerance:
            folded += 1
    return folded >= 3


def min_hand_to_point(hand: np.ndarray, point_xy) -> float:
    probe_idx = [0, 4, 8, 12]
    pts = hand[probe_idx, :2]
    dx = pts[:, 0] - point_xy[0]
    dy = pts[:, 1] - point_xy[1]
    return float(np.sqrt(dx * dx + dy * dy).min())


def hand_center(hand: np.ndarray) -> tuple[float, float]:
    return float(np.mean(hand[:, 0])), float(np.mean(hand[:, 1]))


def count_cycles(values: np.ndarray, min_amplitude: float) -> int:
    if len(values) < 4:
        return 0
    smooth = np.convolve(values, np.ones(3) / 3.0, mode="same")
    diffs = np.diff(smooth)
    direction = 0
    extrema = 0
    ref = smooth[0]
    for i, d in enumerate(diffs):
        if abs(d) < 1e-5:
            continue
        sign = 1 if d > 0 else -1
        if direction == 0:
            direction = sign
            ref = smooth[i]
            continue
        if sign != direction:
            cur = smooth[i + 1] if i + 1 < len(smooth) else smooth[i]
            if abs(cur - ref) >= min_amplitude:
                extrema += 1
                ref = cur
                direction = sign
    return extrema // 2


class ScubaRecognizer:
    """Detects: one hand touching the nose + other hand in a fist + shaking.

    The shaking is measured as oscillations of the fist's center of mass in X or Y
    within a short temporal window.
    """

    def __init__(
        self,
        window_s: float = 1.5,
        required_shakes: int = 3,
        shake_amplitude: float = 0.025,
        nose_dist_threshold: float = 0.12,
        min_samples: int = 10,
        cooldown_s: float = 4.0,
    ):
        self.window_s = window_s
        self.required_shakes = required_shakes
        self.shake_amplitude = shake_amplitude
        self.nose_dist_threshold = nose_dist_threshold
        self.min_samples = min_samples
        self.cooldown_s = cooldown_s
        self._history: deque = deque()
        self._last_trigger_t: float = float("-inf")
        self.progress: float = 0.0
        self.status: str = "aguardando"

    def reset(self):
        self._history.clear()
        self.progress = 0.0
        self.status = "aguardando"

    def _fail(self, msg: str):
        self._history.clear()
        self.progress = 0.0
        self.status = msg
        return False

    def update(self, hands_np, nose_xy, timestamp: float | None = None) -> bool:
        t = timestamp if timestamp is not None else time.time()

        if t - self._last_trigger_t < self.cooldown_s:
            self.status = "cooldown"
            return False

        if nose_xy is None:
            return self._fail("rosto nao detectado")

        if len(hands_np) < 2:
            return self._fail("mostre as duas maos")

        hands = hands_np[:2]
        dists = [min_hand_to_point(h, nose_xy) for h in hands]
        closer = int(np.argmin(dists))
        if dists[closer] > self.nose_dist_threshold:
            return self._fail("uma mao no nariz")

        shake_hand = hands[1 - closer]
        if not is_fist(shake_hand):
            return self._fail("feche a outra mao")

        cx, cy = hand_center(shake_hand)
        self._history.append((t, cx, cy))
        while self._history and (t - self._history[0][0]) > self.window_s:
            self._history.popleft()

        if len(self._history) < self.min_samples:
            self.status = "chacoalhe a mao fechada..."
            self.progress = 0.0
            return False

        xs = np.array([h[1] for h in self._history], dtype=np.float32)
        ys = np.array([h[2] for h in self._history], dtype=np.float32)
        cycles = max(
            count_cycles(xs, self.shake_amplitude),
            count_cycles(ys, self.shake_amplitude),
        )
        self.progress = min(cycles / self.required_shakes, 1.0)
        self.status = f"chacoalhando {cycles}/{self.required_shakes}"

        if cycles >= self.required_shakes:
            self._last_trigger_t = t
            self._history.clear()
            self.progress = 0.0
            self.status = "SCUBA CAT!"
            return True
        return False
