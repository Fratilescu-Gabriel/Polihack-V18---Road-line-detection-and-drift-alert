# ETS2 Lane Drift Demo – Code Overview

This project is a small vision-only lane detection and drift alert demo for Euro Truck Simulator 2.  
It reads frames from the screen (like a dashcam), detects lane lines, keeps them stable over time, and flags when the lane hasn’t been reliably seen for a while.

THE DATA FOLDER WAS EXCLUDED DUE TO BIG DISK SIZE OF DATASET

---

## Main Components

### `lane_multi_utils.py`
- Crops a **region of interest (ROI)** from the frame where the road is expected.
- Converts ROI to grayscale, blurs it, and runs **Canny** edge detection.
- Runs **HoughLinesP** on edges to get candidate line segments.
- Filters segments by:
  - slope (ignores almost-horizontal lines),
  - length,
  - position (bottom part of the ROI),
  - brightness (lane paint vs asphalt).
- Splits segments into **left** and **right** groups (by slope sign).
- Fits **one line per side** using least-squares (`np.polyfit`) from all candidate points.
- Applies **temporal smoothing** to the left/right line parameters so they don’t jitter.
- Computes lane width and center, checks basic sanity, and sets a `valid` flag.
- `draw_boundaries_and_lanes(...)` draws the detected lane lines and ego center on the ROI.

---

### `ego_lane_tracker.py`
- Maintains a short **history** of lane offsets over recent frames.
- Smooths the lateral offset from lane center.
- Can be extended to implement more advanced state machines (lane change, drift states, etc.).

---

### `scripts/live_lane_overlay.py`
- Captures the **screen** every frame (full display minus a top strip where the overlay lives).
- Calls `detect_lanes_multi(frame)` to get lane detection results.
- Stores the **last valid detection** (`last_valid_det`) and reuses its geometry when the current frame is invalid, so lane lines stay visible and stable.
- Tracks `last_valid_time` (timestamp of last valid lane).
- Compares `now - last_valid_time` against `TIME_SEC`:
  - If the gap is small → state `"STRAIGHT"`.
  - If the gap is larger than `TIME_SEC` → state `"DRIFTING"`.
- Plays a short **beep** (if `ENABLE_BUZZ` is `True`) when the state first changes from `"STRAIGHT"` to `"DRIFTING"`.
- Renders a small **overlay window**:
  - Shows the ROI with lane lines and ego center.
  - Overlays status text: `STATE`, normalized offset, frame index.

---

### `scripts/replay_lane_multi.py`
- Loads a sequence of saved frames from disk.
- Runs the same `detect_lanes_multi` logic on each frame.
- Displays full frame + ROI with lane drawing for offline inspection and tuning.


### `OTHER FILES`
- Sampling and preprocessing of gameplay video screen capture
- Assisted testing and tuning on the video preprocessed data


### `Arduino prototype implementation`
- The arduino sketch for the physical car prototype