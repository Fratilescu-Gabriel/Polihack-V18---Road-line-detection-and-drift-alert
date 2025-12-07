import cv2
import numpy as np


ROI_TOP_RATIO    = 0.4
ROI_BOTTOM_RATIO = 0.80
ROI_LEFT_RATIO   = 0.15
ROI_RIGHT_RATIO  = 0.95


CANNY_LOW  = 60
CANNY_HIGH = 180
BLUR_KSIZE = (5, 5)


HOUGH_THRESHOLD    = 30
HOUGH_MIN_LINE_LEN = 40
HOUGH_MAX_LINE_GAP = 200

MIN_ABS_SLOPE      = 0.30   
MIN_VERT_LEN_FRAC  = 0.05    
MIN_VERT_LEN_PIX   = 10       
BRIGHT_MIN         = 60      


MIN_Y_FRAC         = 0.55     

LANE_MIN_OFFSET_FRAC = 0.05   
LANE_MAX_OFFSET_FRAC = 0.2


MIN_LANE_WIDTH_FRAC = 0.25
MAX_LANE_WIDTH_FRAC = 0.75

SMOOTH_ALPHA = 0.75 

_prev_left_params = None  
_prev_right_params = None  


def _line_mean_intensity(gray, x1, y1, x2, y2, samples=10):
    """mean gray on line"""
    h, w = gray.shape
    vals = []
    for t in np.linspace(0.0, 1.0, samples):
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))
        if 0 <= x < w and 0 <= y < h:
            vals.append(gray[y, x])
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _fit_side(lines_list, w_roi, side):
    """fit single line y = m x + b for one side"""
    if not lines_list:
        return None

    cx = w_roi / 2.0
    min_off = LANE_MIN_OFFSET_FRAC * w_roi
    max_off = LANE_MAX_OFFSET_FRAC * w_roi

    pts_x = []
    pts_y = []

    for (x1, y1, x2, y2) in lines_list:
        if y1 > y2:
            xb, yb = x1, y1
        else:
            xb, yb = x2, y2

        if side == "left":
            if xb >= cx:
                continue
            dx = cx - xb
        else:
            if xb <= cx:
                continue
            dx = xb - cx

        if dx < min_off or dx > max_off:
            continue


        pts_x.extend([x1, x2])
        pts_y.extend([y1, y2])

    if len(pts_x) < 4:
        return None

    X = np.array(pts_x, dtype=np.float32)
    Y = np.array(pts_y, dtype=np.float32)


    m, b = np.polyfit(X, Y, 1)
    return float(m), float(b)


def detect_lanes_multi(frame):
    """ego lane with simple robust fit + smoothing"""
    global _prev_left_params, _prev_right_params

    h, w, _ = frame.shape

    top = int(ROI_TOP_RATIO * h)
    bottom = int(ROI_BOTTOM_RATIO * h)
    left = int(ROI_LEFT_RATIO * w)
    right = int(ROI_RIGHT_RATIO * w)
    if bottom <= top:
        bottom = h
    if right <= left:
        right = w

    roi = frame[top:bottom, left:right]
    h_roi, w_roi, _ = roi.shape
    ego_center_x = w_roi / 2.0


    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    left_raw = []
    right_raw = []

    if lines is not None:
        min_len = max(MIN_VERT_LEN_PIX, int(MIN_VERT_LEN_FRAC * h_roi))
        min_y = int(MIN_Y_FRAC * h_roi)

        for (x1, y1, x2, y2) in lines[:, 0, :]:
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            vert_len = abs(y2 - y1)
            yb = max(y1, y2)

            if abs(slope) < MIN_ABS_SLOPE:
                continue
            if vert_len < min_len:
                continue
            if yb < min_y:
                continue

            mean_int = _line_mean_intensity(gray, x1, y1, x2, y2)
            if mean_int < BRIGHT_MIN:
                continue

            if slope < 0:
                left_raw.append((x1, y1, x2, y2))
            else:
                right_raw.append((x1, y1, x2, y2))


    left_params = _fit_side(left_raw, w_roi, side="left")
    right_params = _fit_side(right_raw, w_roi, side="right")

    valid = left_params is not None and right_params is not None

    if valid:
        if _prev_left_params is None:
            _prev_left_params = left_params
            _prev_right_params = right_params
        else:
            mL_prev, bL_prev = _prev_left_params
            mR_prev, bR_prev = _prev_right_params
            mL_curr, bL_curr = left_params
            mR_curr, bR_curr = right_params

            mL_sm = SMOOTH_ALPHA * mL_prev + (1.0 - SMOOTH_ALPHA) * mL_curr
            bL_sm = SMOOTH_ALPHA * bL_prev + (1.0 - SMOOTH_ALPHA) * bL_curr
            mR_sm = SMOOTH_ALPHA * mR_prev + (1.0 - SMOOTH_ALPHA) * mR_curr
            bR_sm = SMOOTH_ALPHA * bR_prev + (1.0 - SMOOTH_ALPHA) * bR_curr

            left_params = (mL_sm, bL_sm)
            right_params = (mR_sm, bR_sm)

            _prev_left_params = left_params
            _prev_right_params = right_params
    else:

        pass

    boundaries = []
    lanes = []

    if valid:
        mL, bL = left_params
        mR, bR = right_params

        y_bottom = h_roi - 1

        def x_at_y(m, b, y):
            return (y - b) / m

        xL = x_at_y(mL, bL, y_bottom)
        xR = x_at_y(mR, bR, y_bottom)
        lane_center = 0.5 * (xL + xR)
        width = xR - xL

        if width <= 0:
            valid = False
        else:
            if width < MIN_LANE_WIDTH_FRAC * w_roi or width > MAX_LANE_WIDTH_FRAC * w_roi:
                valid = False

        if valid:
            boundaries = [
                {"x_bottom": float(xL), "m": float(mL), "b": float(bL)},
                {"x_bottom": float(xR), "m": float(mR), "b": float(bR)},
            ]
            boundaries.sort(key=lambda b: b["x_bottom"])
            lanes = [{
                "left_idx": 0,
                "right_idx": 1,
                "center_x": (boundaries[0]["x_bottom"] + boundaries[1]["x_bottom"]) / 2.0,
                "width_px": abs(boundaries[1]["x_bottom"] - boundaries[0]["x_bottom"]),
            }]
    else:
        lane_center = None

    geom_lane_idx = 0 if (valid and lanes) else None

    return {
        "roi": roi,
        "edges": edges,
        "boundaries": boundaries,
        "lanes": lanes,
        "geom_lane_idx": geom_lane_idx,
        "ego_center_x": ego_center_x,
        "valid": valid,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "h_roi": h_roi,
        "w_roi": w_roi,
    }


def draw_boundaries_and_lanes(roi, det, tracked_lane_idx=None):
    """draw lanes"""
    out = roi.copy()
    h_roi, w_roi, _ = out.shape
    boundaries = det["boundaries"]
    lanes = det["lanes"]

    for b in boundaries:
        m = b["m"]
        bb = b["b"]

        def x_at_y(y):
            return int((y - bb) / m)

        y1 = h_roi
        y2 = int(h_roi * 0.4)
        x1 = x_at_y(y1)
        x2 = x_at_y(y2)
        cv2.line(out, (x1, y1), (x2, y2), (100, 100, 100), 2)

    for idx, lane in enumerate(lanes):
        left = boundaries[lane["left_idx"]]
        right = boundaries[lane["right_idx"]]

        def x_at_y(m, b, y):
            return int((y - b) / m)

        y1 = h_roi
        y2 = int(h_roi * 0.4)

        x1L = x_at_y(left["m"], left["b"], y1)
        x2L = x_at_y(left["m"], left["b"], y2)
        x1R = x_at_y(right["m"], right["b"], y1)
        x2R = x_at_y(right["m"], right["b"], y2)

        color = (0, 255, 0)
        thick = 3
        if tracked_lane_idx is not None and idx == tracked_lane_idx:
            color = (0, 0, 255)
            thick = 4

        cv2.line(out, (x1L, y1), (x2L, y2), color, thick)
        cv2.line(out, (x1R, y1), (x2R, y2), color, thick)

    ego_cx = int(det["ego_center_x"])
    cv2.line(out, (ego_cx, 0), (ego_cx, h_roi - 1), (0, 255, 255), 2)

    return out
