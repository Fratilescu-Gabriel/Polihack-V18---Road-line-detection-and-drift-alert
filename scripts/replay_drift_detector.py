
from pathlib import Path
import sys
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from lane_detection_utils import (
    compute_lane_offset,
    ROI_TOP_RATIO,
    ROI_BOTTOM_RATIO,
    ROI_LEFT_RATIO,
    ROI_RIGHT_RATIO,
)
from drift_detector import DriftDetector 


def main():
    project_root = SCRIPT_DIR.parents[0]
    frames_dir = project_root / "data" / "processed" / "sample_frames"

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Frames dir  : {frames_dir}")

    if not frames_dir.exists():
        print("[ERROR] Frames dir does not exist. Run your sampling script or adjust frames_dir.")
        return

    image_files = list(frames_dir.glob("*.jpg"))
    if not image_files:
        print("[ERROR] No .jpg files in frames_dir.")
        return

    image_files = sorted(image_files, key=lambda p: p.name)
    print(f"[INFO] Using {len(image_files)} frames.")

    detector = DriftDetector(history_len=15)

    # Create resizable windows
    main_win = "Full frame + ROI + state"
    roi_win = "Lane detection in ROI"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        h, w, _ = frame.shape
        top = int(ROI_TOP_RATIO * h)
        bottom = int(ROI_BOTTOM_RATIO * h)
        left = int(ROI_LEFT_RATIO * w)
        right = int(ROI_RIGHT_RATIO * w)

        # full frame with ROI rectangle
        full_img = frame.copy()
        cv2.rectangle(full_img, (left, top), (right, bottom), (0, 0, 255), 2)

        offset, valid, debug_img = compute_lane_offset(frame, debug=True)
        state = detector.update(offset, valid=valid)

        # choose text color by state
        color = (0, 255, 0)   # STRAIGHT: green
        if "DRIFT" in state:
            color = (0, 255, 255)  # DRIFT: yellow
        if "LANE_CHANGE" in state:
            color = (0, 0, 255)    # LANE_CHANGE: red

        #TEXT ON IMAGE
        text1 = f"STATE: {state}"
        text2 = f"offset={offset:.2f}  valid={valid}"


        y1 = h - 40
        y2 = h - 10
        x = 10

        cv2.putText(
            full_img,
            text1,
            (x, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            full_img,
            text2,
            (x, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ROI debug window
        if debug_img is None:
            debug_img = frame.copy()

        cv2.imshow(main_win, full_img)
        cv2.imshow(roi_win, debug_img)

        # ~30 FPS; space to pause; q/ESC to quit
        key = cv2.waitKey(33) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    return
                if k2 == ord(" "):
                    break

    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
