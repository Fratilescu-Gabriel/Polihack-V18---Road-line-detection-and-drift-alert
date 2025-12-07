from pathlib import Path
import sys
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from lane_detection_utils import (
    detect_lanes_raw,
    ROI_TOP_RATIO,
    ROI_BOTTOM_RATIO,
    ROI_LEFT_RATIO,
    ROI_RIGHT_RATIO,
)
from drift_detector import DriftDetector  # noqa: E402


def resize_keep_aspect(img, max_w=1280, max_h=720):
    """keep aspect"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_lanes_on_roi(roi, lane):
    """draw lanes"""
    out = roi.copy()
    if lane is None:
        return out
    
    left_params = lane.get("left_params")
    right_params = lane.get("right_params")
    lane_center = lane.get("lane_center")
    vehicle_center = lane.get("vehicle_center")

    if left_params is None or right_params is None or lane_center is None:
        return out

    h_roi, w_roi, _ = out.shape

    def x_at_y(m, b, y):
        return (y - b) / m

    mL, bL = left_params
    mR, bR = right_params

    y1 = h_roi
    y2 = int(h_roi * 0.4)

    x1L = int(x_at_y(mL, bL, y1))
    x2L = int(x_at_y(mL, bL, y2))
    x1R = int(x_at_y(mR, bR, y1))
    x2R = int(x_at_y(mR, bR, y2))

    # left red / right green
    cv2.line(out, (x1L, y1), (x2L, y2), (0, 0, 255), 3)
    cv2.line(out, (x1R, y1), (x2R, y2), (0, 255, 0), 3)

    # centers
    lane_cx = int(lane_center)
    veh_cx = int(vehicle_center if vehicle_center is not None else w_roi // 2)

    cv2.line(out, (lane_cx, 0), (lane_cx, h_roi - 1), (255, 0, 0), 2)        # lane blue
    cv2.line(out, (veh_cx, 0), (veh_cx, h_roi - 1), (0, 255, 255), 2)        # veh yellow

    return out


def replay_run(run_dir: Path):
    """replay one run"""
    print(f"[INFO] Replaying run: {run_dir.name}")

    image_files = sorted(run_dir.glob("*.jpg"))
    if not image_files:
        print(f"[WARN] No .jpg files in {run_dir}")
        return

    detector = DriftDetector(history_len=15)

    main_win = f"Run: {run_dir.name} (full)"
    roi_win = f"Run: {run_dir.name} (roi)"
    cv2.namedWindow(main_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(roi_win, cv2.WINDOW_AUTOSIZE)

    # approx skip (~10s – tune for your fps)
    SKIP_FRAMES = 300

    last_lane = None  # last valid lane

    lane_id = 0               # relative lane index
    lane_change_pending = False
    lane_change_dir = 0       # -1 left, +1 right
    prev_state = "STRAIGHT"

    i = 0
    n = len(image_files)

    while i < n:
        if i < 1200:
            i += 1
            continue

        print(i)
        img_path = image_files[i]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read {img_path}")
            i += 1
            continue

        h, w, _ = frame.shape
        top = int(ROI_TOP_RATIO * h)
        bottom = int(ROI_BOTTOM_RATIO * h)
        left = int(ROI_LEFT_RATIO * w)
        right = int(ROI_RIGHT_RATIO * w)

        full_img = frame.copy()
        cv2.rectangle(full_img, (left, top), (right, bottom), (0, 0, 255), 2)

        # lane detect (current frame)
        d = detect_lanes_raw(frame)
        offset = d["offset"]
        geom_valid = d["valid"]

        # drift state uses raw offset + geom_valid
        state = detector.update(offset, valid=geom_valid)

        # lane-change -> lane_id
        if not lane_change_pending:
            if state == "LANE_CHANGE_LEFT":
                lane_change_pending = True
                lane_change_dir = -1
            elif state == "LANE_CHANGE_RIGHT":
                lane_change_pending = True
                lane_change_dir = +1
        else:
            # wait until lane-change finishes
            if state == "STRAIGHT":
                lane_id += lane_change_dir
                lane_change_pending = False
                lane_change_dir = 0

        prev_state = state

        # current ROI crop
        roi = frame[top:bottom, left:right]

        # lane struct for current frame
        curr_lane = None
        if geom_valid and d["left_params"] is not None and d["right_params"] is not None and d["lane_center"] is not None:
            curr_lane = {
                "left_params": d["left_params"],
                "right_params": d["right_params"],
                "lane_center": d["lane_center"],
                "vehicle_center": d["vehicle_center"],
            }
            last_lane = curr_lane  # update last_lane

        # draw lanes
        if curr_lane is not None:
            roi_used = draw_lanes_on_roi(roi, curr_lane)
        elif last_lane is not None:
            roi_used = draw_lanes_on_roi(roi, last_lane)
        else:
            roi_used = roi

        # state color
        color = (0, 255, 0)
        if "DRIFT" in state:
            color = (0, 255, 255)
        if "LANE_CHANGE" in state:
            color = (0, 0, 255)

        text1 = f"STATE: {state}"
        text2 = f"off={offset:.2f}  valid={geom_valid}"
        text3 = f"lane_id={lane_id}"

        y1 = h - 60
        y2 = h - 35
        y3 = h - 10
        x = 10

        cv2.putText(full_img, text1, (x, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(full_img, text2, (x, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(full_img, text3, (x, y3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1, cv2.LINE_AA)

        full_disp = resize_keep_aspect(full_img, max_w=1280, max_h=720)
        roi_disp = resize_keep_aspect(roi_used, max_w=640, max_h=480)

        cv2.imshow(main_win, full_disp)
        cv2.imshow(roi_win, roi_disp)

        key = cv2.waitKey(33) & 0xFF

        if key in (ord("q"), 27):       # quit run
            break

        elif key == ord(" "):           # pause
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    return
                if k2 == ord(" "):
                    break

        elif key == ord("n"):           # skip ~10s
            i = min(i + SKIP_FRAMES, n - 1)
            detector = DriftDetector(history_len=15)
            last_lane = None
            lane_id = 0
            lane_change_pending = False
            lane_change_dir = 0
            prev_state = "STRAIGHT"
            continue

        i += 1

    cv2.destroyWindow(main_win)
    cv2.destroyWindow(roi_win)


def main():
    project_root = SCRIPT_DIR.parents[0]
    runs_root = project_root / "data" / "processed" / "gameplay_overtake"

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Runs root  : {runs_root}")

    if not runs_root.exists():
        print("[ERROR] Runs root does not exist. Did you run extract_user_video_frames.py?")
        return

    replay_run(runs_root)


if __name__ == "__main__":
    main()
