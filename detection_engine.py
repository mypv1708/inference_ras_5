"""
Main detection engine
Handles the core detection loop and processing pipeline with threaded capture
Optimized for Raspberry Pi 4/5 (no manual resize - YOLO handles letterbox)
"""

import time
from threading import Thread, Lock
from ultralytics import YOLO
import cv2

from config import Config
from camera_utils import setup_camera, get_model_info
from silkworm_detection import process_detections
from display_utils import draw_performance_info, draw_scale_bar, display_frame
from performance_monitor import PerformanceMonitor
from object_tracker import SilkwormTracker
from freeze_detection import process_freeze
from overlap_detection import process_overlap


class DetectionEngine:
    """Main detection engine for silkworm detection."""

    def __init__(self, args, cfg: Config):
        self.args = args
        self.cfg = cfg

        # Initialize
        self.model = self._setup_model()
        self.cap = self._setup_camera()
        self.tracker = self._setup_tracker()
        self.performance_monitor = PerformanceMonitor()

        # Threaded capture
        self.running = True
        self._frame_lock = Lock()
        self._latest_frame = None
        self._capture_thread = Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # States
        self.overlap_counters = {}
        self.head_history = {}
        self.freeze_counters = {}
        self.device = cfg.device if cfg.device is not None else 'cpu'

        print(f"Camera: 640x480 @ {args.fps} FPS")
        print(f"Inference size: {args.imgsz}")
        print(f"Config: conf={args.conf}, skip={args.skip}, pose_conf={cfg.pose_conf}")

        # ✅ Raspberry Pi CPU optimization
        if self.device == 'cpu':
            try:
                import torch
                torch.set_num_threads(2)  # best for Pi4/5
                print("[Torch] threads:", torch.get_num_threads())
            except Exception:
                pass

        # ✅ Reduce camera buffer latency
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass

    def _setup_model(self):
        model_path = self.args.model if self.args.model else self.cfg.model_path
        print(get_model_info(model_path))
        return YOLO(model_path)

    def _setup_camera(self):
        return setup_camera(self.args.camera, self.args.fps)

    def _setup_tracker(self):
        return SilkwormTracker(
            max_distance=self.cfg.max_distance,
            max_disappeared=self.cfg.max_disappeared
        )

    def run(self):
        try:
            while True:
                # Get frame
                with self._frame_lock:
                    frame = None if self._latest_frame is None else self._latest_frame.copy()
                    self._latest_frame = None
                if frame is None:
                    time.sleep(0.001)
                    continue

                # Skip for performance
                if (self.performance_monitor.total_frames_read % max(1, self.args.skip)) != 0:
                    self.performance_monitor.start_frame()
                    continue

                # Process
                self._process_frame(frame)
                self.performance_monitor.start_frame()

                if self.performance_monitor.should_cleanup():
                    self.performance_monitor.cleanup()

                self._control_fps()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self._cleanup()
            self._print_summary()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame
            self.performance_monitor.record_frame_read()

    # ✅ Optimized for Raspberry Pi — no manual resize
    def _process_frame(self, frame):
        inference_start = time.time()
        results = self.model(
            frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.cfg.iou_thresh,
            device=self.device,
            verbose=False
        )
        inference_time = time.time() - inference_start

        result = results[0]
        silkworms = process_detections(result, frame, self.cfg, self.tracker)

        if result.boxes is not None:
            self.performance_monitor.record_detections(len(result.boxes))

        # Freeze logic
        for obj_id, head, body, tail, bbox in silkworms:
            process_freeze(obj_id, head, 1.0, bbox, self.cfg,
                           self.head_history, self.freeze_counters, frame)

        # Overlap
        if len(silkworms) >= 2:
            process_overlap(silkworms, self.overlap_counters, frame, self.cfg)

        # Display performance
        if self.args.benchmark:
            self.performance_monitor.record_inference_time(inference_time)
            draw_performance_info(frame, inference_time, silkworms,
                                  self.performance_monitor.frame_count, self.performance_monitor)

        draw_scale_bar(frame)

        if not self.args.no_display:
            if display_frame(frame):
                raise KeyboardInterrupt("Exit")

    def _control_fps(self):
        current_fps = self.performance_monitor.get_current_fps()
        if current_fps > self.args.fps:
            time.sleep(0.1)

    def _cleanup(self):
        self.running = False
        try:
            if self._capture_thread.is_alive():
                self._capture_thread.join(timeout=0.5)
        except:
            pass

        self.cap.release()
        if not self.args.no_display:
            cv2.destroyAllWindows()

    def _print_summary(self):
        if self.args.benchmark:
            self.performance_monitor.print_summary()
