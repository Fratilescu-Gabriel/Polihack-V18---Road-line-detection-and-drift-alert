class EgoLaneTracker:
    def __init__(self, commit_frames=6):
        self.commit_frames = commit_frames

        self.tracked_idx = None   # lane we draw
        self.pending_idx = None   # candidate new lane
        self.pending_count = 0

        self.state = "NO_LANE"    # NO_LANE / STRAIGHT / LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT

    def update(self, det):
        """update state"""

        lanes = det["lanes"]
        geom_idx = det["geom_lane_idx"]

        if not det["valid"] or not lanes or geom_idx is None:
            # no geometry
            self.state = "NO_LANE"
            self.pending_idx = None
            self.pending_count = 0
            return {
                "tracked_idx": self.tracked_idx,
                "state": self.state,
                "offset_norm": 0.0,
            }

        # init
        if self.tracked_idx is None:
            self.tracked_idx = geom_idx
            self.pending_idx = None
            self.pending_count = 0
            self.state = "STRAIGHT"
        else:
            if geom_idx == self.tracked_idx:
                # still in same lane
                self.pending_idx = None
                self.pending_count = 0
                self.state = "STRAIGHT"
            else:
                # candidate lane change
                if self.pending_idx is None or self.pending_idx != geom_idx:
                    self.pending_idx = geom_idx
                    self.pending_count = 1
                else:
                    self.pending_count += 1

                direction = "LANE_CHANGE_RIGHT" if geom_idx > self.tracked_idx else "LANE_CHANGE_LEFT"
                self.state = direction

                # commit when stable
                if self.pending_count >= self.commit_frames:
                    # lane change over -> follow new lane
                    self.tracked_idx = self.pending_idx
                    self.pending_idx = None
                    self.pending_count = 0
                    self.state = "STRAIGHT"

        # offset vs tracked lane
        offset_norm = 0.0
        if self.tracked_idx is not None and 0 <= self.tracked_idx < len(lanes):
            lane = lanes[self.tracked_idx]
            center = lane["center_x"]
            width = lane["width_px"]
            ego_x = det["ego_center_x"]
            if width > 1e-3:
                offset_norm = (ego_x - center) / (0.5 * width)

        return {
            "tracked_idx": self.tracked_idx,
            "state": self.state,
            "offset_norm": float(offset_norm),
        }
