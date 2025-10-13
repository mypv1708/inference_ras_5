import cv2
import numpy as np
from config import Config


def process_freeze(obj_id, head, head_c, bbox, cfg: Config, head_history, freeze_counters, frame):
    """Process freeze detection for a single silkworm - optimized version"""
    # Early return for low confidence
    if head_c < cfg.pose_conf:
        return

    prev = head_history.get(obj_id)
    if prev is not None:
        # OPTIMIZED: Use squared distance to avoid sqrt
        dx, dy = head[0] - prev[0], head[1] - prev[1]
        dist_squared = dx*dx + dy*dy
        pixel_thresh_squared = cfg.pixel_thresh * cfg.pixel_thresh
        
        # OPTIMIZED: Single dict access
        current_count = freeze_counters.get(obj_id, 0)
        if dist_squared <= pixel_thresh_squared:
            freeze_counters[obj_id] = current_count + 1
        else:
            freeze_counters[obj_id] = 0
    else:
        freeze_counters[obj_id] = 0

    head_history[obj_id] = head

    # OPTIMIZED: Only draw if freeze detected
    if freeze_counters[obj_id] >= cfg.freeze_frames_thresh:
        # OPTIMIZED: Direct bbox access
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cfg.freeze_color, 2)