from pathlib import Path
import sys
import time
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from lane_multi_utils import (
    detect_lanes_multi,
    draw_boundaries_and_lanes,
)
from ego_lane_tracker import EgoLaneTracker  # noqa: E402

# how long we tolerate no new lane detections before flagging deviation (seconds)
NO_LANE_TIME_SEC = 2.0


def resize_keep_aspect(img, max_w=1280, max_h=720):
    """keep aspect"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def replay_run(run_dir: Path):
    """replay with new tracker"""
    print(f"[INFO] Replaying run: {run_dir}")

    image_files = sorted(run_dir.glob("*.jpg"))
    if not image_files:
        print(f"[WARN] No .jpg files in {run_dir}")
        return

    tracker = EgoLaneTracker(commit_frames=6)

    # last valid geometry + gap timer
    last_valid_det = None
    no_lane_start_time = None  # when we first started reusing last_valid_det

    main_win = "Full frame"
    roi_win = "ROI / lanes"
    cv2.namedWindow(main_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(roi_win, cv2.WINDOW_AUTOSIZE)

    SKIP_FRAMES = 300

    i = 0
    n = len(image_files)

    while i < n:
        img_path = image_files[i]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read {img_path}")
            i += 1
            continue

        h, w, _ = frame.shape

        det = detect_lanes_multi(frame)
        now = time.time()

        # --- gap timer logic: are we in "no new lane, using last" mode? ---
        if (not det["valid"]) and (last_valid_det is not None):
            # we have a previous lane, but current frame has no valid detection
            if no_lane_start_time is None:
                no_lane_start_time = now
        else:
            # either we have a fresh valid lane or no history at all -> reset timer
            no_lane_start_time = None

        # choose geom for tracker:
        #  - valid now      -> current det
        #  - invalid        -> last_valid_det (freeze lanes)
        if det["valid"]:
            det_for_tracker = dict(det)  # shallow copy ok
            det_for_tracker["valid"] = True
            last_valid_det = det_for_tracker
        else:
            if last_valid_det is not None:
                det_for_tracker = last_valid_det
            else:
                # no history yet
                det_for_tracker = {
                    "lanes": [],
                    "geom_lane_idx": None,
                    "valid": False,
                }

        # tracker update (for offset), but we ignore its state for display
        if det_for_tracker.get("valid", False) and det_for_tracker.get("lanes"):
            tracker_out = tracker.update(det_for_tracker)
        else:
            tracker_out = {
                "tracked_idx": None,
                "state": "NO_LANE",
                "offset_norm": 0.0,
            }

        tracked_idx = tracker_out["tracked_idx"]
        offset_norm = tracker_out["offset_norm"]

        # --- high-level display state based on gap timer ---
        if no_lane_start_time is not None and (now - no_lane_start_time) >= NO_LANE_TIME_SEC:
            display_state = "DEVIATING"
        else:
            display_state = "STRAIGHT"
        # ---------------------------------------------------

        # visual ROI with lanes
        roi = det["roi"]  # always current frame ROI
        # for drawing, if no current lanes but we have last, reuse last lanes
        if not det["valid"] and last_valid_det is not None:
            det_draw = dict(last_valid_det)
        else:
            det_draw = det

        roi_dbg = draw_boundaries_and_lanes(roi, det_draw, tracked_lane_idx=tracked_idx)

        # draw roi rect on full frame (cast everything to int)
        top = int(det["top"])
        bottom = int(det["bottom"])
        left = int(det["left"])
        right = int(det["right"])

        full_img = frame.copy()
        cv2.rectangle(full_img, (left, top), (right, bottom), (0, 0, 255), 2)

        # status text color
        if display_state == "DEVIATING":
            color = (0, 0, 255)   # red
        else:
            color = (0, 255, 0)   # green

        text1 = f"STATE: {display_state}"
        text2 = f"offset_norm={offset_norm:.2f}"
        text3 = f"tracked_lane={tracked_idx}"
        text4 = f"frame_idx={i}"

        y1 = h - 80
        y2 = h - 55
        y3 = h - 30
        y4 = 30
        x = 10

        cv2.putText(full_img, text1, (x, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(full_img, text2, (x, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(full_img, text3, (x, y3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(full_img, text4, (w - 220, y4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

        full_disp = resize_keep_aspect(full_img, max_w=1280, max_h=720)
        roi_disp = resize_keep_aspect(roi_dbg, max_w=800, max_h=450)

        cv2.imshow(main_win, full_disp)
        cv2.imshow(roi_win, roi_disp)

        key = cv2.waitKey(33) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            # pause
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    return
                if k2 == ord(" "):
                    break
        elif key == ord("n"):
            # skip ~10s
            i = min(i + SKIP_FRAMES, n - 1)
            tracker = EgoLaneTracker(commit_frames=6)
            last_valid_det = None   # no stale lanes after big jump
            no_lane_start_time = None
            continue

        i += 1

    cv2.destroyAllWindows()


def main():
    project_root = SCRIPT_DIR.parents[0]
    run_dir = project_root / "data" / "processed" / "gameplay_overtake"

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Run dir    : {run_dir}")

    if not run_dir.exists():
        print("[ERROR] Run dir missing")
        return

    replay_run(run_dir)


if __name__ == "__main__":
    main()
