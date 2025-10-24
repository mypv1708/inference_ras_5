import cv2
from ultralytics import YOLO

from config import Config
from freeze_detection import process_freeze
from overlap_detection import process_overlap
from silkworm_detection import draw_silkworm    


def main():
    """Secondary entry point WITHOUT heatmap and WITHOUT trajectory drawing.

    This keeps detection, ID assignment (from tracker inside YOLO),
    optional freeze and overlap checks, and writing raw frames with basic
    silkworm drawings (bbox + keypoints + links) only.
    """
    cfg = Config()
    model = YOLO(cfg.model_path)

    cap = cv2.VideoCapture(cfg.source_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {cfg.source_video}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(cfg.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    unique_ids, overlap_counters, head_history, freeze_counters = set(), {}, {}, {}

    for result in model.track(
        source=cfg.source_video,
        tracker=cfg.tracker_config,
        conf=cfg.detect_conf,
        iou=cfg.iou_thresh,
        persist=True,
        stream=True,
        imgsz=cfg.imgsz
    ):
        frame = result.orig_img
        boxes, kpts = result.boxes, result.keypoints
        silkworms = []

        if boxes.id is not None and kpts is not None:
            ids = boxes.id.int().cpu().tolist()
            unique_ids.update(ids)
            points_all = kpts.xy.cpu().numpy().astype(int)
            confs_all = kpts.conf.cpu().numpy()
            bboxes = boxes.xyxy.cpu().numpy().astype(int)

            for idx, obj_id in enumerate(ids):
                pts, confs, bbox = points_all[idx], confs_all[idx], bboxes[idx]
                
                # Use configurable keypoint indices
                head_idx = cfg.head_kp_index
                body_idx = head_idx + 1
                tail_idx = head_idx + 2
                
                # Check if we have enough keypoints
                if tail_idx >= pts.shape[0] or tail_idx >= confs.shape[0]:
                    continue
                    
                head = tuple(pts[head_idx])
                body = tuple(pts[body_idx])
                tail = tuple(pts[tail_idx])
                head_c = confs[head_idx]
                body_c = confs[body_idx]
                tail_c = confs[tail_idx]

                # Basic drawing only (no trajectory path)
                draw_silkworm(frame, obj_id, head, body, tail, bbox, head_c, body_c, tail_c, cfg)

                # Keep logic & counting, but avoid extra drawing to reduce load
                if head_c >= cfg.pose_conf:
                    silkworms.append((obj_id, head, body, tail, bbox))
                    process_freeze(obj_id, head, head_c, bbox, cfg, head_history, freeze_counters, frame)

        # Overlap checks (only when we have at least 2 silkworms)
        if len(silkworms) >= 2:
            process_overlap(silkworms, overlap_counters, frame, cfg)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved: {cfg.output_video}")
    print(f"Tổng số object (id) duy nhất: {len(unique_ids)}")


if __name__ == "__main__":
    main()