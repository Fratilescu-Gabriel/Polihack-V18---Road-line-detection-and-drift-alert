import cv2
import numpy as np
# ---------- ROI (relative) ----------
ROI_TOP_RATIO    = 0.15   # cut top
ROI_BOTTOM_RATIO = 0.80   # cut bottom
ROI_LEFT_RATIO   = 0.15   # cut left
ROI_RIGHT_RATIO  = 0.95   # cut right

# ---------- Canny ----------
CANNY_LOW  = 60
CANNY_HIGH = 180
BLUR_KSIZE = (5, 5)

# ---------- Hough ----------
HOUGH_THRESHOLD    = 35
HOUGH_MIN_LINE_LEN = 25
HOUGH_MAX_LINE_GAP = 250

# ---------- Line filters ----------
MIN_ABS_SLOPE      = 0.25   # min |slope|
MIN_VERT_LEN_FRAC  = 0.15   # min vertical span (h_frac)

# ---------- Lane band ----------
LANE_MIN_OFFSET_FRAC = 0.0  # min dist from center (w_frac)
LANE_MAX_OFFSET_FRAC = 0.75  # max dist from center
MAX_CLUSTER_DEV_FRAC = 0.5  # max dev from median bottom x

# ---------- Mask ----------
USE_TRAPEZOID_MASK = True
TRAP_BOTTOM_Y_FRAC = 1.0
TRAP_TOP_Y_FRAC    = 0.5
TRAP_LEFT_BOTTOM_FRAC  = 0.05
TRAP_RIGHT_BOTTOM_FRAC = 0.95
TRAP_LEFT_TOP_FRAC     = 0.35
TRAP_RIGHT_TOP_FRAC    = 0.65

# ---------- Brightness heuristic ----------
LINE_BRIGHT_MIN = 65  # tune brightness


def _build_trapezoid_mask(edges):
    """build mask"""
    h_roi, w_roi = edges.shape
    yb = int(TRAP_BOTTOM_Y_FRAC * h_roi)
    yt = int(TRAP_TOP_Y_FRAC * h_roi)
    xlb = int(TRAP_LEFT_BOTTOM_FRAC * w_roi)
    xrb = int(TRAP_RIGHT_BOTTOM_FRAC * w_roi)
    xlt = int(TRAP_LEFT_TOP_FRAC * w_roi)
    xrt = int(TRAP_RIGHT_TOP_FRAC * w_roi)

    mask = np.zeros_like(edges)
    pts = np.array([[(xlb, yb), (xlt, yt), (xrt, yt), (xrb, yb)]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask


def _line_mean_intensity(gray, x1, y1, x2, y2, samples=10):
    """mean gray"""
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


def _average_line_robust(lines_list, w_roi, side):
    """robust lane avg"""
    if not lines_list:
        return None

    center_x = w_roi / 2.0
    min_off = LANE_MIN_OFFSET_FRAC * w_roi
    max_off = LANE_MAX_OFFSET_FRAC * w_roi

    candidates = []
    for (x1, y1, x2, y2) in lines_list:
        if x2 == x1:
            continue

        # bottom endpoint
        if y1 > y2:
            xb, yb = x1, y1
        else:
            xb, yb = x2, y2

        # side filter
        if side == "left":
            if xb >= center_x:
                continue
            dx = center_x - xb
        else:
            if xb <= center_x:
                continue
            dx = xb - center_x

        # lane band filter
        if dx < min_off or dx > max_off:
            continue

        # vertical length
        L = abs(y2 - y1)
        if L <= 0:
            continue

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        candidates.append((L, xb, m, b))

    if not candidates:
        return None

    # keep top-K by length
    candidates.sort(key=lambda t: t[0], reverse=True)
    K = min(6, len(candidates))
    top = candidates[:K]

    # cluster by bottom x
    xbs = np.array([c[1] for c in top], dtype=float)
    med_xb = np.median(xbs)
    max_dev = MAX_CLUSTER_DEV_FRAC * w_roi
    filtered = [c for c in top if abs(c[1] - med_xb) <= max_dev]
    if not filtered:
        filtered = top

    Ls = np.array([c[0] for c in filtered], dtype=float)
    ms = np.array([c[2] for c in filtered], dtype=float)
    bs = np.array([c[3] for c in filtered], dtype=float)
    wsum = np.sum(Ls)

    m_avg = float(np.sum(ms * Ls) / wsum)
    b_avg = float(np.sum(bs * Ls) / wsum)
    return m_avg, b_avg


def detect_lanes_raw(frame):
    """low-level lane detect"""
    h, w, _ = frame.shape

    # ROI crop
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

    # gray + blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)

    # edges
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # optional mask
    if USE_TRAPEZOID_MASK:
        mask = _build_trapezoid_mask(edges)
        edges_roi = cv2.bitwise_and(edges, mask)
    else:
        edges_roi = edges

    # Hough
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0, :]:
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            vert_len = abs(y2 - y1)
            max_y = max(y1, y2)

            # vertical span
            if vert_len < MIN_VERT_LEN_FRAC * h_roi:
                continue

            # slope limit
            if abs(slope) < MIN_ABS_SLOPE:
                continue

            # lower-half filter
            if max_y < 0.5 * h_roi:
                continue

            # brightness filter (reject dark car edges)
            mean_int = _line_mean_intensity(gray, x1, y1, x2, y2)
            if mean_int < LINE_BRIGHT_MIN:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    left_params = _average_line_robust(left_lines, w_roi, side="left")
    right_params = _average_line_robust(right_lines, w_roi, side="right")

    def x_at_y(m, b, y):
        return (y - b) / m

    valid = left_params is not None and right_params is not None
    offset = 0.0
    lane_center = None
    vehicle_center = w_roi / 2.0

    if valid:
        mL, bL = left_params
        mR, bR = right_params
        y_bottom = h_roi - 1
        x_left = x_at_y(mL, bL, y_bottom)
        x_right = x_at_y(mR, bR, y_bottom)
        lane_center = (x_left + x_right) / 2.0
        offset_pixels = vehicle_center - lane_center
        offset = float(offset_pixels / (w_roi / 2.0))

    return {
        "roi": roi,
        "edges_roi": edges_roi,
        "left_lines": left_lines,
        "right_lines": right_lines,
        "left_params": left_params,
        "right_params": right_params,
        "lane_center": lane_center,
        "vehicle_center": vehicle_center,
        "offset": offset,
        "valid": valid,
    }


def compute_lane_offset(frame, debug=False):
    """public API"""
    d = detect_lanes_raw(frame)

    offset = d["offset"]
    valid = d["valid"]

    if not debug:
        return offset, valid, None

    roi = d["roi"]
    h_roi, w_roi, _ = roi.shape

    if not valid:
        dbg = cv2.cvtColor(d["edges_roi"], cv2.COLOR_GRAY2BGR)
        return offset, False, dbg

    debug_img = roi.copy()

    # draw candidates
    for (x1, y1, x2, y2) in d["left_lines"]:
        cv2.line(debug_img, (x1, y1), (x2, y2), (128, 0, 0), 1)   # dark red
    for (x1, y1, x2, y2) in d["right_lines"]:
        cv2.line(debug_img, (x1, y1), (x2, y2), (0, 128, 0), 1)   # dark green

    def x_at_y(m, b, y):
        return (y - b) / m

    mL, bL = d["left_params"]
    mR, bR = d["right_params"]
    y1 = h_roi
    y2 = int(h_roi * 0.4)
    x1L = int(x_at_y(mL, bL, y1))
    x2L = int(x_at_y(mL, bL, y2))
    x1R = int(x_at_y(mR, bR, y1))
    x2R = int(x_at_y(mR, bR, y2))

    cv2.line(debug_img, (x1L, y1), (x2L, y2), (0, 0, 255), 3)    # left red
    cv2.line(debug_img, (x1R, y1), (x2R, y2), (0, 255, 0), 3)    # right green

    lane_cx = int(d["lane_center"])
    veh_cx = int(d["vehicle_center"])
    cv2.line(debug_img, (lane_cx, 0), (lane_cx, h_roi - 1), (255, 0, 0), 2)      # lane center blue
    cv2.line(debug_img, (veh_cx, 0), (veh_cx, h_roi - 1), (0, 255, 255), 2)      # vehicle center yellow

    return offset, True, debug_img
