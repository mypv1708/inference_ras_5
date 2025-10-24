import cv2
import argparse
import time
from ultralytics import YOLO

from config import Config
from freeze_detection import process_freeze
from overlap_detection import process_overlap
from silkworm_detection import draw_silkworm


def parse_args():
    # Load config first to get default values
    cfg = Config()
    
    parser = argparse.ArgumentParser(description="Raspberry Pi optimized silkworm detection")
    parser.add_argument("--camera", type=int, default=cfg.camera_index, help="Camera index")
    parser.add_argument("--model", type=str, default=cfg.model_path, help="Model path override")
    parser.add_argument("--imgsz", type=int, default=cfg.imgsz, help="Inference image size")
    parser.add_argument("--fps", type=int, default=cfg.fps, help="Target FPS")
    parser.add_argument("--skip", type=int, default=cfg.vid_stride, help="Skip every N frames")
    parser.add_argument("--conf", type=float, default=cfg.detect_conf, help="Detection confidence")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--benchmark", action="store_true", help="Show performance metrics")
    parser.add_argument("--list", action="store_true", help="List available cameras and exit")
    return parser.parse_args()


def list_cameras(max_index: int = 5):
    """List available cameras."""
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    print(f"📷 Available cameras: {available if available else 'None detected (tried 0..' + str(max_index) + ')'}")
    return available


def main():
    """Raspberry Pi optimized version - keeps all logic from main.py but optimized for Pi."""
    args = parse_args()
    
    if args.list:
        list_cameras()
        return
    
    cfg = Config()
    
    # ✅ Use NCNN model if available, fallback to PT
    model_path = args.model if args.model else cfg.model_path
    if "ncnn" in model_path.lower():
        print("🚀 Using NCNN model for better Pi performance")
    else:
        print("⚠️ Using PyTorch model - may be slow on Pi")
    
    model = YOLO(model_path)

    # ✅ Setup camera with Pi-friendly settings (like pose_person_realtime.py)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Không mở được camera {args.camera}, thử camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không mở được camera: {args.camera} và camera 0")
    
    # ✅ Pi-optimized camera settings (from pose_person_realtime.py)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # ✅ Test camera read
    ret, test_frame = cap.read()
    if not ret:
        print("❌ Camera không đọc được frame, thử lại...")
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    print(f"📹 Camera: 640x480 @ {args.fps} FPS")
    print(f"🎯 Inference size: {args.imgsz}")
    print(f"⚙️ Config values: conf={args.conf}, skip={args.skip}, pose_conf={cfg.pose_conf}")

    # ✅ Performance tracking
    frame_count = 0
    inference_times = []
    start_time = time.time()
    
    # ✅ Keep ALL logic from main.py
    unique_ids, overlap_counters, head_history, freeze_counters = set(), {}, {}, {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Không đọc được frame")
                break

            # ✅ Skip frames for performance
            if frame_count % args.skip != 0:
                frame_count += 1
                continue

            inference_start = time.time()
            
            # ✅ Use predict with all parameters from main.py for consistency
            device = cfg.device if cfg.device is not None else 'cpu'
            results = model.predict(
                frame, 
                imgsz=args.imgsz, 
                conf=args.conf,
                iou=cfg.iou_thresh,
                verbose=False,
                device=device
            )
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            result = results[0]
            boxes, kpts = result.boxes, result.keypoints
            silkworms = []

            # ✅ Keep EXACT logic from main.py
            if boxes is not None and kpts is not None:
                # ✅ Use simple ID assignment like pose_person_realtime.py
                ids = list(range(len(boxes)))  # fallback nếu không có tracking id
                unique_ids.update(ids)
                points_all = kpts.xy.cpu().numpy().astype(int)
                confs_all = kpts.conf.cpu().numpy()
                bboxes = boxes.xyxy.cpu().numpy().astype(int)

                for idx, obj_id in enumerate(ids):
                    pts, confs, bbox = points_all[idx], confs_all[idx], bboxes[idx]
                    
                    # ✅ Use configurable keypoint indices (from main.py)
                    head_idx = cfg.head_kp_index
                    body_idx = head_idx + 1
                    tail_idx = head_idx + 2
                    
                    # ✅ Check if we have enough keypoints (from main.py)
                    if tail_idx >= pts.shape[0] or tail_idx >= confs.shape[0]:
                        continue
                        
                    head = tuple(pts[head_idx])
                    body = tuple(pts[body_idx])
                    tail = tuple(pts[tail_idx])
                    head_c = confs[head_idx]
                    body_c = confs[body_idx]
                    tail_c = confs[tail_idx]

                    # ✅ Basic drawing only (no trajectory path) - from main.py
                    draw_silkworm(frame, obj_id, head, body, tail, bbox, head_c, body_c, tail_c, cfg)

                    # ✅ Keep logic & counting, but avoid extra drawing to reduce load - from main.py
                    if head_c >= cfg.pose_conf:
                        silkworms.append((obj_id, head, body, tail, bbox))
                        process_freeze(obj_id, head, head_c, bbox, cfg, head_history, freeze_counters, frame)

            # ✅ Overlap checks (only when we have at least 2 silkworms) - from main.py
            if len(silkworms) >= 2:
                process_overlap(silkworms, overlap_counters, frame, cfg)

            # ✅ Add performance info (like pose_person_realtime.py)
            if args.benchmark:
                # ✅ Use same FPS calculation as pose_person_realtime.py
                fps = 1000 / inference_time if inference_time > 0 else 0
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Detections: {len(silkworms)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ✅ Display (like pose_person_realtime.py)
            if not args.no_display:
                cv2.imshow("Silkworm Detection (Pi Optimized)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            frame_count += 1
            
            # ✅ FPS control
            elapsed = time.time() - start_time
            if elapsed > 0:
                current_fps = frame_count / elapsed
                if current_fps > args.fps:
                    time.sleep(0.1)  # Slow down if too fast

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ✅ Performance summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"✅ Processed {frame_count} frames in {total_time:.1f}s")
    print(f"✅ Average FPS: {avg_fps:.1f}")
    print(f"✅ Total objects (unique IDs): {len(unique_ids)}")
    
    if args.benchmark and inference_times:
        avg_inference = sum(inference_times) / len(inference_times)
        print(f"✅ Average inference: {avg_inference*1000:.1f}ms")


if __name__ == "__main__":
    main()
