import cv2
import argparse
from ultralytics import YOLO

from config import Config
from freeze_detection import process_freeze
from overlap_detection import process_overlap
from silkworm_detection import draw_silkworm


def parse_args():
    parser = argparse.ArgumentParser(description="Silkworm realtime camera inference")
    parser.add_argument("--camera", type=int, default=None, help="Override camera index (default from config)")
    parser.add_argument("--list", action="store_true", help="List available camera indices and exit")
    return parser.parse_args()


def list_cameras(max_index: int = 5):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    print(f"Available cameras: {available if available else 'None detected (tried 0..' + str(max_index) + ')'}")


def main():
    """Realtime camera inference WITHOUT heatmap and WITHOUT trajectory drawing.

    Uses the same logic as main_2.py but reads from a webcam (source=0),
    shows a live window, and optionally writes frames to the configured output.
    """
    args = parse_args()
    if args.list:
        list_cameras()
        return

    cfg = Config()
    model = YOLO(cfg.model_path)

    camera_index = args.camera if args.camera is not None else cfg.camera_index
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được camera (source={camera_index})")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out = None
    if cfg.write_output:
        out = cv2.VideoWriter(cfg.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    unique_ids, overlap_counters, head_history, freeze_counters = set(), {}, {}, {}

    for result in model.track(
        source=camera_index,
        tracker=cfg.tracker_config,
        conf=cfg.detect_conf,
        iou=cfg.iou_thresh,
        persist=True,
        stream=True,
        imgsz=cfg.imgsz,
        device=cfg.device,
        vid_stride=cfg.vid_stride,
    ):
        frame = result.orig_img
        boxes, kpts = result.boxes, result.keypoints
        silkworms = []

        if boxes is not None and boxes.id is not None and kpts is not None:
            ids = boxes.id.int().cpu().tolist()
            unique_ids.update(ids)
            points_all = kpts.xy.cpu().numpy().astype(int)
            confs_all = kpts.conf.cpu().numpy()
            bboxes = boxes.xyxy.cpu().numpy().astype(int)

            for idx, obj_id in enumerate(ids):
                pts, confs, bbox = points_all[idx], confs_all[idx], bboxes[idx]
                if cfg.head_kp_index >= pts.shape[0]:
                    continue
                head, body, tail = tuple(pts[0]), tuple(pts[1]), tuple(pts[2])
                head_c, body_c, tail_c = confs[0], confs[1], confs[2]

                draw_silkworm(frame, obj_id, head, body, tail, bbox, head_c, body_c, tail_c, cfg)

                if head_c >= cfg.pose_conf:
                    silkworms.append((obj_id, head, body, tail, bbox))
                    process_freeze(obj_id, head, head_c, bbox, cfg, head_history, freeze_counters, frame)

        if len(silkworms) >= 2:
            process_overlap(silkworms, overlap_counters, frame, cfg)

        if out is not None:
            out.write(frame)
        if cfg.display:
            cv2.imshow("Silkworm Realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    if out is not None:
        out.release()
    if cfg.display:
        cv2.destroyAllWindows()
    print(f"Video saved: {cfg.output_video}")
    print(f"Tổng số object (id) duy nhất: {len(unique_ids)}")


if __name__ == "__main__":
    main()


