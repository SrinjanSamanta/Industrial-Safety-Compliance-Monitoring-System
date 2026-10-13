"""
SafeGuardAI -
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO

from database import log_violation

VIDEO_SOURCE = "sample4.mp4"  # None for webcam
PPE_MODEL_PATH = "ppe.pt"    # classes: 0=head, 1=helmet

ZONE_REL = np.array([
    [0.19, 0.18],
    [0.20, 0.34],
    [0.33, 0.33],
    [0.33, 0.18],   
], dtype=np.float32)

FALL_RATIO_THRESH = 1.2
FALL_CONFIRM_SECONDS = 5
FALL_GRACE_SECONDS = 1
LOG_COOLDOWN_SECONDS = 5
PPE_CONF = 0.4

RED, BLUE, MAGENTA, WHITE = (0, 0, 255), (255, 0, 0), (255, 0, 255), (255, 255, 255)
ORANGE = (0, 165, 255)


def draw_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y, th + 8)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw, y), color, -1)
    cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


def safe_log(event_type, details):
    try:
        log_violation(event_type, details)
    except Exception as e:
        print(f"[DB ERROR] {e}")


def log_with_cooldown(last_logged, event_type, details, now):
    if now - last_logged.get(event_type, 0) >= LOG_COOLDOWN_SECONDS:
        safe_log(event_type, details)
        last_logged[event_type] = now


def centers_of(result, names, wanted):
    """Centers of PPE boxes whose class name equals `wanted`."""
    out = []
    for b in result.boxes:
        if names[int(b.cls[0])].lower() == wanted:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            out.append(((x1 + x2) // 2, (y1 + y2) // 2))
    return out


def in_head_area(pt, x1, y1, x2, y2):
    """True if point lies in the top ~40% of the person box (plus a margin above)."""
    hx, hy = pt
    return x1 <= hx <= x2 and (y1 - 20) <= hy <= (y1 + 0.4 * (y2 - y1))


def main():
    model = YOLO("yolov8n.pt")
    ppe_model = YOLO(PPE_MODEL_PATH)

    source = VIDEO_SOURCE if VIDEO_SOURCE else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 1

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zone = (ZONE_REL * [fw, fh]).astype(np.int32)

    fall_start = None
    last_fall_seen = 0
    alert_active = False
    last_logged = {}
    frame_i = 0
    helmets, heads = [], []
    print("Press 'q' to quit.")
    cv2.namedWindow("SafeGuardAI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SafeGuardAI", 1000, 560)  # adjust to your screen

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Inference on the clean frame
        frame_i += 1
        results = model(frame, classes=[0], conf=0.4, imgsz=480, verbose=False)
        if frame_i % 3 == 0:
            ppe_res = ppe_model(frame, conf=PPE_CONF, imgsz=480, verbose=False)[0]
            helmets = centers_of(ppe_res, ppe_model.names, "helmet")
            heads = centers_of(ppe_res, ppe_model.names, "head")

        # Zone overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], RED)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [zone], True, RED, 2)
        cv2.putText(frame, "RESTRICTED ZONE",
                    (zone[0][0] + 10, zone[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        detections = []
        any_fall = False
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            bw, bh = x2 - x1, y2 - y1
            ratio = bw / bh if bh > 0 else 0
            possible_fall = ratio > FALL_RATIO_THRESH
            any_fall |= possible_fall
            detections.append((x1, y1, x2, y2, conf, ratio, possible_fall))

        # Time-based fall confirmation
        now = time.time()
        if any_fall:
            if fall_start is None:
                fall_start = now
            last_fall_seen = now
        elif fall_start is not None and now - last_fall_seen > FALL_GRACE_SECONDS:
            fall_start = None
            alert_active = False

        fall_confirmed = fall_start is not None and (now - fall_start) >= FALL_CONFIRM_SECONDS

        if fall_confirmed and not alert_active:
            print(f"[ALERT] FALL / MAN-DOWN detected (down for {now - fall_start:.1f}s)")
            alert_active = True
            safe_log("FALL_EMERGENCY", f"Person down for {now - fall_start:.1f}s")

        for x1, y1, x2, y2, conf, ratio, possible_fall in detections:
            foot = (int((x1 + x2) / 2), y2)
            inside = cv2.pointPolygonTest(zone, foot, False) >= 0

            # Hard-hat check: bare head seen and no helmet on this worker
            has_helmet = any(in_head_area(p, x1, y1, x2, y2) for p in helmets)
            bare_head = any(in_head_area(p, x1, y1, x2, y2) for p in heads)
            no_hat = bare_head and not has_helmet and not possible_fall

            if no_hat:
                log_with_cooldown(last_logged, "NO_HARDHAT",
                                  f"Worker without hard hat conf={conf:.2f}", now)

            if possible_fall and fall_confirmed:
                color, thickness = MAGENTA, 4
                label = f"FALL EMERGENCY {conf:.2f}"
            elif possible_fall and fall_start is not None:
                color, thickness = MAGENTA, 2
                label = f"Possible fall {now - fall_start:.1f}s/{FALL_CONFIRM_SECONDS}s"
            elif inside:
                color, thickness = RED, 2
                label = f"HAZARD BREACH {conf:.2f}"
                log_with_cooldown(last_logged, "HAZARD_BREACH",
                                  f"Foot at ({foot[0]},{foot[1]}) conf={conf:.2f}", now)
            elif no_hat:
                color, thickness = ORANGE, 3
                label = f"NO HARD HAT {conf:.2f}"
            else:
                color, thickness = BLUE, 2
                label = f"Worker Safe {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(frame, foot, 5, color, -1)
            draw_label(frame, label, x1, y1, color)

        # Draw hard hats (green) for visual debugging
        for hx, hy in helmets:
            cv2.circle(frame, (hx, hy), 6, (0, 200, 0), -1)

        cv2.imshow("SafeGuardAI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()