#!/usr/bin/env python3
"""
Optimized Silkworm Detection for Raspberry Pi
Clean, modular code with improved performance and logic
"""

import argparse

from config import Config
from camera_utils import list_cameras
from detection_engine import DetectionEngine


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


def main():
    """Main function - optimized silkworm detection."""
    args = parse_args()
    
    if args.list:
        list_cameras()
        return
    
    cfg = Config()
    
    # Initialize and run detection engine
    engine = DetectionEngine(args, cfg)
    engine.run()


if __name__ == "__main__":
    main()
