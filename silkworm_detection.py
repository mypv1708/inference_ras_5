import cv2
from typing import List, Tuple
from config import Config
from object_tracker import SilkwormTracker


def draw_silkworm(frame, obj_id, head, body, tail, bbox, 
                  head_c, body_c, tail_c, cfg: Config):
    """Draw silkworm with bounding box, keypoints and connections"""
    x1, y1, x2, y2 = bbox

    cv2.rectangle(frame, (x1,y1), (x2,y2), cfg.bbox_color, 2)
    cv2.putText(frame, f"ID {obj_id}", (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.bbox_color, 2)

    if head_c >= cfg.pose_conf:
        cv2.circle(frame, head, 3, cfg.head_color, -1)
    if body_c >= cfg.pose_conf:
        cv2.circle(frame, body, 3, cfg.body_color, -1)
    if tail_c >= cfg.pose_conf:
        cv2.circle(frame, tail, 3, cfg.tail_color, -1)

    if head_c >= cfg.pose_conf and body_c >= cfg.pose_conf:
        cv2.line(frame, head, body, cfg.line_color, 2)
    if body_c >= cfg.pose_conf and tail_c >= cfg.pose_conf:
        cv2.line(frame, body, tail, cfg.line_color, 2)


def process_detections(result, frame, cfg, tracker: SilkwormTracker) -> List[Tuple]:
    """Process detection results and extract silkworms with stable IDs."""
    boxes, kpts = result.boxes, result.keypoints
    
    if boxes is None or kpts is None:
        return tracker.update([])
    
    # Extract data once
    points_all = kpts.xy.cpu().numpy().astype(int)
    confs_all = kpts.conf.cpu().numpy()
    bboxes = boxes.xyxy.cpu().numpy().astype(int)
    
    # Collect valid detections
    detections = []
    for idx in range(len(bboxes)):
        pts, confs, bbox = points_all[idx], confs_all[idx], bboxes[idx]
        
        # Get keypoint indices
        head_idx = cfg.head_kp_index
        body_idx = head_idx + 1
        tail_idx = head_idx + 2
        
        # Validate keypoints
        if tail_idx >= pts.shape[0] or tail_idx >= confs.shape[0]:
            continue
        
        # Extract keypoints
        head = tuple(pts[head_idx])
        body = tuple(pts[body_idx])
        tail = tuple(pts[tail_idx])
        head_c = confs[head_idx]
        body_c = confs[body_idx]
        tail_c = confs[tail_idx]
        
        # Add to detections if confidence is high enough
        if head_c >= cfg.pose_conf:
            detections.append((head, body, tail, bbox))
    
    # Update tracker and get stable IDs
    tracked_silkworms = tracker.update(detections)
    
    # Draw silkworms with stable IDs
    for obj_id, head, body, tail, bbox in tracked_silkworms:
        # Get confidence for drawing (use average confidence for simplicity)
        head_c = 0.8  # Default confidence for drawing
        body_c = 0.8
        tail_c = 0.8
        
        draw_silkworm(frame, obj_id, head, body, tail, bbox, head_c, body_c, tail_c, cfg)
    
    return tracked_silkworms
