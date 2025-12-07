from pathlib import Path
import random
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


def main():
    project_root = SCRIPT_DIR.parents[0]
    sample_dir = project_root / "data" / "processed" / "gameplay_straight"

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Sample frames dir: {sample_dir}")

    if not sample_dir.exists():
        print("[ERROR] Sample frames directory does not exist.")
        return

    image_files = list(sample_dir.glob("*.jpg"))
    if not image_files:
        print("[ERROR] No .jpg in sample_frames.")
        return

    print(f"[INFO] Found {len(image_files)} sample images.")
    random.shuffle(image_files)

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        h, w, _ = frame.shape
        
        #ROI BOX
        top = int(ROI_TOP_RATIO * h)
        bottom = int(ROI_BOTTOM_RATIO * h)
        left = int(ROI_LEFT_RATIO * w)
        right = int(ROI_RIGHT_RATIO * w)

        frame_with_roi = frame.copy()
        cv2.rectangle(frame_with_roi, (left, top), (right, bottom), (0, 0, 255), 2)

        offset, valid, debug_img = compute_lane_offset(frame, debug=True)
        if debug_img is None:
            debug_img = frame.copy()

        text = f"{img_path.name}  offset={offset:.2f}  valid={valid}"
        cv2.putText(
            debug_img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Full frame + ROI", frame_with_roi)
        cv2.imshow("Lane detection in ROI", debug_img)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
