import sys

import cv2

from scuba_cat.capture import WebcamCapture
from scuba_cat.config import CFG
from scuba_cat.face_detector import FaceDetector
from scuba_cat.hand_tracker import HandTracker
from scuba_cat.overlay import GifOverlay
from scuba_cat.recognizer import ScubaRecognizer


def draw_progress_bar(frame, progress: float, armed: bool):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.5)
    bar_h = 14
    x = (w - bar_w) // 2
    y = h - 30
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * max(0.0, min(progress, 1.0)))
    color = (0, 200, 0) if armed else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 1)


def main():
    if sys.version_info >= (3, 13):
        print(
            "ERRO: MediaPipe nao suporta Python >= 3.13. Use Python 3.11 ou 3.12.",
            file=sys.stderr,
        )
        sys.exit(1)

    tracker = HandTracker(
        max_hands=CFG.max_num_hands,
        det_conf=CFG.min_detection_confidence,
        trk_conf=CFG.min_tracking_confidence,
    )
    face = FaceDetector(min_confidence=0.5)
    recognizer = ScubaRecognizer(
        window_s=1.5,
        required_shakes=3,
        shake_amplitude=0.025,
        nose_dist_threshold=0.15,
        cooldown_s=CFG.cooldown_s,
    )

    overlay = None
    if CFG.gif_path.exists():
        try:
            overlay = GifOverlay(
                CFG.gif_path,
                duration_s=CFG.overlay_duration_s,
                remove_chroma=True,
            )
            print(f"GIF carregado: {CFG.gif_path}")
        except Exception as e:
            print(f"AVISO: nao foi possivel carregar o GIF ({e}). Rodando sem overlay.")
    else:
        print(f"AVISO: GIF nao encontrado em {CFG.gif_path}.")

    with WebcamCapture(CFG.camera_index, CFG.frame_width, CFG.frame_height) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break

            results, hands_np = tracker.process(frame)
            nose_xy = face.nose_tip(frame)
            tracker.draw(frame, results)

            h_px, w_px = frame.shape[:2]
            if nose_xy is not None:
                nx = int(nose_xy[0] * w_px)
                ny = int(nose_xy[1] * h_px)
                cv2.circle(frame, (nx, ny), 6, (0, 255, 255), -1)
                cv2.circle(frame, (nx, ny), 30, (0, 255, 255), 1)

            triggered = recognizer.update(hands_np, nose_xy)
            if triggered and overlay is not None:
                overlay.trigger()

            if overlay is not None:
                frame = overlay.compose(frame)

            draw_progress_bar(
                frame,
                recognizer.progress,
                armed=overlay is not None and overlay.active,
            )

            status = recognizer.status
            if overlay and overlay.active:
                status = "SCUBA CAT!"
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                frame,
                "uma mao no nariz + outra fechada chacoalhando",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )
            cv2.putText(
                frame, "q: sair  |  d: forcar  |  r: reset",
                (10, h_px - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            cv2.imshow("Scuba Cat", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d") and overlay is not None:
                overlay.trigger()
            if key == ord("r"):
                recognizer.reset()

    tracker.close()
    face.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
