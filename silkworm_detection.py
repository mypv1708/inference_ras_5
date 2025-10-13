import cv2
from config import Config


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
