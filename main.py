#!/usr/bin/env python3
"""
Optimized Silkworm Detection for Raspberry Pi
Clean code with improved performance and logic
"""

import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO

from config import Config
from freeze_detection import process_freeze
from overlap_detection import process_overlap
from silkworm_detection import draw_silkworm
from object_tracker import SilkwormTracker


def parse_args():
    """Parse command line arguments with config defaults."""
    cfg = Config()
    
    parser = argparse.ArgumentParser(description="Optimized silkworm detection for Raspberry Pi")
    parser.add_argument("--camera", type=int, default=cfg.camera_index, help="Camera index")
    parser.add_argument("--model", type=str, default=cfg.model_path, help="Model path")
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
    print(f"Available cameras: {available if available else 'None detected (tried 0..' + str(max_index) + ')'}")
    return available


def setup_camera(camera_index: int, target_fps: int = 15):
    """Setup camera with optimized settings."""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}, trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open any camera (tried {camera_index} and 0)")
    
    # Optimized camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # Test camera read
    ret, _ = cap.read()
    if not ret:
        print("Camera cannot read frames, retrying...")
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    return cap


def get_model_info(model_path: str):
    """Get model information and performance hints."""
    if "ncnn" in model_path.lower():
        return "NCNN model - optimized for Pi"
    elif "onnx" in model_path.lower():
        return "ONNX model - optimized for Pi"
    else:
        return "PyTorch model - may be slow on Pi"


def process_detections(result, frame, cfg, tracker):
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


def draw_performance_info(frame, inference_time, silkworms, frame_count):
    """Draw performance information on frame."""
    fps = (1.0 / inference_time) if inference_time > 0 else 0
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {len(silkworms)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    """Main function - optimized silkworm detection."""
    args = parse_args()
    
    if args.list:
        list_cameras()
        return
    
    cfg = Config()
    
    # Model setup
    model_path = args.model if args.model else cfg.model_path
    print(get_model_info(model_path))
    
    model = YOLO(model_path)
    
    # Camera setup
    cap = setup_camera(args.camera, args.fps)
    print(f"Camera: 640x480 @ {args.fps} FPS")
    print(f"Inference size: {args.imgsz}")
    print(f"Config: conf={args.conf}, skip={args.skip}, pose_conf={cfg.pose_conf}")
    
    # Performance tracking
    frame_count = 0
    inference_times = []
    start_time = time.time()
    
    # Detection state
    total_detections, overlap_counters, head_history, freeze_counters = 0, {}, {}, {}
    # Select device once
    device = cfg.device if cfg.device is not None else 'cpu'
    
    # Initialize tracker
    tracker = SilkwormTracker(
        max_distance=cfg.max_distance,
        max_disappeared=cfg.max_disappeared
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame")
                break
            
            # Skip frames for performance
            if frame_count % args.skip != 0:
                frame_count += 1
                continue
            
            # Inference
            inference_start = time.time()
            results = model.predict(
                frame, 
                imgsz=args.imgsz, 
                conf=args.conf,
                iou=cfg.iou_thresh,
                verbose=False,
                device=device
            )
            inference_time = time.time() - inference_start
            if args.benchmark:
                inference_times.append(inference_time)
            
            # Process detections
            result = results[0]
            silkworms = process_detections(result, frame, cfg, tracker)
            
            # Update total detections (sum of detections across frames)
            if result.boxes is not None:
                total_detections += len(result.boxes)
            
            # Freeze detection (use stable tracker IDs; avoid indexing YOLO confidences by ID)
            for silkworm in silkworms:
                obj_id, head, body, tail, bbox = silkworm
                head_c = 1.0  # treat as confident detection for freeze logic gating
                process_freeze(obj_id, head, head_c, bbox, cfg, head_history, freeze_counters, frame)
            
            # Overlap detection
            if len(silkworms) >= 2:
                process_overlap(silkworms, overlap_counters, frame, cfg)
            
            # Performance display
            if args.benchmark:
                draw_performance_info(frame, inference_time, silkworms, frame_count)
            
            # Display
            if not args.no_display:
                cv2.imshow("Silkworm Detection (Optimized)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            
            frame_count += 1
            
            # FPS control
            elapsed = time.time() - start_time
            if elapsed > 0:
                current_fps = frame_count / elapsed
                if current_fps > args.fps:
                    time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    
    # Performance summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"Processed {frame_count} frames in {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total detections: {total_detections}")
    
    if args.benchmark and inference_times:
        avg_inference = np.mean(inference_times)
        print(f"Average inference: {avg_inference*1000:.1f}ms")


if __name__ == "__main__":
    main()
