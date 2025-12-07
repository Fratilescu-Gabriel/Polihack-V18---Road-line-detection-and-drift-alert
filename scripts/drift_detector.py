from collections import deque
import numpy as np


class DriftDetector:

    def __init__(self, history_len=25):
        """
        history_len: how many recent frames we look at (~0.5s if ~30 FPS)
        """
        self.history = deque(maxlen=history_len)
        self.state = "STRAIGHT"
        self.frames_in_state = 0

        # thresholds
        self.DRIFT_START_OFFSET = 0.20   # start of noticeable deviation
        self.LANE_CHANGE_OFFSET = 0.55   # clearly in neighbour lane
        self.BACK_TO_CENTER = 0.12      # considered centered again

        self.MIN_DRIFT_FRAMES = 5       # must last this many frames to be real
        self.MAX_DRIFT_FRAMES = 45      # if still far after this, treat as lane change

    def _compute_features(self):
        if not self.history:
            return 0.0, 0.0, 0.0

        offsets = np.array(self.history, dtype=float)
        current = float(offsets[-1])
        mean_abs = float(np.mean(np.abs(offsets)))
        deriv = np.diff(offsets, prepend=offsets[0])
        mean_deriv = float(np.mean(deriv))
        return current, mean_abs, mean_deriv

    def update(self, offset, valid=True):
        if not valid:
            # if detection failed, assume no big change this frame
            if self.history:
                offset = self.history[-1]
            else:
                offset = 0.0

        self.history.append(offset)
        current, mean_abs, mean_deriv = self._compute_features()

        sign = np.sign(current)  # -1 = left, +1 = right, 0 = centered-ish

        # state machine
        if self.state == "STRAIGHT":
            self.frames_in_state += 1

            # potential drift start
            if abs(current) > self.DRIFT_START_OFFSET and abs(mean_deriv) > 0.01:
                self.state = "DRIFTING_LEFT" if current < 0 else "DRIFTING_RIGHT"
                self.frames_in_state = 0

        elif self.state in ("DRIFTING_LEFT", "DRIFTING_RIGHT"):
            self.frames_in_state += 1

            # done drifting
            if abs(current) < self.BACK_TO_CENTER:
                self.state = "STRAIGHT"
                self.frames_in_state = 0

            elif abs(current) > self.LANE_CHANGE_OFFSET and self.frames_in_state > self.MIN_DRIFT_FRAMES:
                # promote to lane change
                if "LEFT" in self.state:
                    self.state = "LANE_CHANGE_LEFT"
                else:
                    self.state = "LANE_CHANGE_RIGHT"
                self.frames_in_state = 0

            elif self.frames_in_state > self.MAX_DRIFT_FRAMES:
                if "LEFT" in self.state:
                    self.state = "LANE_CHANGE_LEFT"
                else:
                    self.state = "LANE_CHANGE_RIGHT"
                self.frames_in_state = 0

        elif self.state in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            self.frames_in_state += 1

            if abs(current) < self.BACK_TO_CENTER:
                self.state = "STRAIGHT"
                self.frames_in_state = 0

        return self.state
