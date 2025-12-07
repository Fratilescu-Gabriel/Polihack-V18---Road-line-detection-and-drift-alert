from pathlib import Path
import sys
import time

import cv2
import mss
import numpy as np


ENABLE_BUZZ = True


BUZZ_FREQ_HZ = 1200
BUZZ_DURATION_MS = 150

TIME_SEC = 1.25

SCREEN_W = 2560
SCREEN_H = 1600

CAPTURE_TOP_OFFSET = 200


try:
    import winsound
    HAVE_WINSOUND = True
except ImportError:
    HAVE_WINSOUND = False


def buzz():
    """short beep"""
    if not ENABLE_BUZZ:
        return
    if not HAVE_WINSOUND:
        return
    winsound.Beep(BUZZ_FREQ_HZ, BUZZ_DURATION_MS)


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from lane_multi_utils import (
    detect_lanes_multi,
    draw_boundaries_and_lanes,
)
from ego_lane_tracker import EgoLaneTracker  # noqa: E402


def resize_keep_aspect(img, max_w=960, max_h=540):
    """keep aspect"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    project_root = SCRIPT_DIR.parents[0]
    print(f"[INFO] Project root: {project_root}")

    tracker = EgoLaneTracker(commit_frames=6)

    
    last_valid_det = None
    last_valid_time = None

    
    CAPTURE_H = SCREEN_H - CAPTURE_TOP_OFFSET
    MONITOR = {
        "top": CAPTURE_TOP_OFFSET,
        "left": 0,
        "width": SCREEN_W,
        "height": CAPTURE_H,
    }
    print(f"[INFO] Capture: {MONITOR['width']}x{MONITOR['height']} at y={MONITOR['top']}")

    
    roi_win = "Lane Overlay"
    cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(roi_win, 960, 540)
    cv2.moveWindow(roi_win, 0, 0)
    try:
        cv2.setWindowProperty(roi_win, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    frame_idx = 0
    prev_display_state = "STRAIGHT"

    with mss.mss() as sct:
        while True:
            
            sct_img = sct.grab(MONITOR)
            frame = np.array(sct_img)[:, :, :3]  # BGRA -> BGR

            h, w, _ = frame.shape
            now = time.time()

            
            det = detect_lanes_multi(frame)
            valid_now = det["valid"] and bool(det["lanes"])

            
            if valid_now:
                last_valid_det = det
                last_valid_time = now

            
            roi = det["roi"]
            if last_valid_det is not None:
                det_draw = last_valid_det
            else:
                det_draw = det

            roi_dbg = draw_boundaries_and_lanes(roi, det_draw, tracked_lane_idx=None)

            
            if last_valid_det is not None and last_valid_det["valid"] and last_valid_det["lanes"]:
                tracker_out = tracker.update(last_valid_det)
                offset_norm = tracker_out["offset_norm"]
            else:
                offset_norm = 0.0

           
            if last_valid_time is None:
                display_state = "STRAIGHT"
            else:
                dt = now - last_valid_time
                if dt > TIME_SEC:
                    display_state = "DRIFTING"
                else:
                    display_state = "STRAIGHT"

            
            if display_state == "DRIFTING" and prev_display_state != "DRIFTING":
                buzz()
            prev_display_state = display_state

            
            if display_state == "DRIFTING":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            
            rh, rw, _ = roi_dbg.shape
            text1 = f"STATE: {display_state}"
            text2 = f"off_norm={offset_norm:.2f}"
            text3 = f"frame_idx={frame_idx}"

            x = 15
            y1 = rh - 70
            y2 = rh - 35
            y3 = rh - 5

            cv2.putText(roi_dbg, text1, (x, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
            cv2.putText(roi_dbg, text2, (x, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(roi_dbg, text3, (x, y3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 0), 2, cv2.LINE_AA)

            roi_disp = resize_keep_aspect(roi_dbg, max_w=960, max_h=540)
            cv2.imshow(roi_win, roi_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
