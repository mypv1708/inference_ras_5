import yaml
import os
from typing import Tuple


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, config_path)

        # Load configuration from YAML file
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Video configuration
        self.camera_index: int = config['video'].get('camera_index', 0)
        self.vid_stride: int = config['video'].get('vid_stride', 1)
        self.fps: int = config['video'].get('fps', 15)
        
        # Model configuration
        self.model_path: str = config['model']['path']
        self.device = config['model'].get('device', None)
        self.imgsz: int = int(config['model'].get('imgsz', 640))
        
        # Thresholds
        self.pose_conf: float = config['thresholds']['pose_conf']
        self.detect_conf: float = config['thresholds']['detect_conf']
        self.iou_thresh: float = config['thresholds']['iou_thresh']
        self.overlap_frames_thresh: int = config['thresholds']['overlap_frames_thresh']
        self.point_segment_dist_thresh: float = float(config['thresholds'].get('point_segment_dist_thresh', 12.0))
        
        # Freeze detection
        self.pixel_thresh: int = config['freeze']['pixel_thresh']
        self.freeze_frames_thresh: int = config['freeze']['freeze_frames_thresh']
        
        # Tracker configuration
        self.head_kp_index: int = config['tracker']['head_kp_index']
        self.max_distance: float = config['tracker'].get('max_distance', 50.0)
        self.max_disappeared: int = config['tracker'].get('max_disappeared', 10)
        
        # Drawing configuration
        self.freeze_color: Tuple[int, int, int] = tuple(config['drawing']['freeze_color'])
        self.overlap_color: Tuple[int, int, int] = tuple(config['drawing']['overlap_color'])
        self.head_color: Tuple[int, int, int] = tuple(config['drawing']['head_color'])
        self.body_color: Tuple[int, int, int] = tuple(config['drawing']['body_color'])
        self.tail_color: Tuple[int, int, int] = tuple(config['drawing']['tail_color'])
        self.line_color: Tuple[int, int, int] = tuple(config['drawing']['line_color'])
        self.bbox_color: Tuple[int, int, int] = tuple(config['drawing']['bbox_color'])