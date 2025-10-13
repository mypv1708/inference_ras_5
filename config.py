import yaml
import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    def __init__(self, config_path: str = "config.yaml"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, config_path)

        # Load configuration from YAML file
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Video configuration
        self.source_video: str = config['video']['source']
        self.output_video: str = config['video']['output']
        self.camera_index: int = config['video'].get('camera_index', 0)
        self.display: bool = config['video'].get('display', True)
        self.write_output: bool = config['video'].get('write_output', True)
        self.vid_stride: int = config['video'].get('vid_stride', 1)
        
        # Model configuration
        self.model_path: str = config['model']['path']
        self.tracker_config: str = config['model']['tracker_config']
        self.device = config['model'].get('device', None)
        self.imgsz: int = int(config['model'].get('imgsz', 640))
        
        # Thresholds
        self.pose_conf: float = config['thresholds']['pose_conf']
        self.detect_conf: float = config['thresholds']['detect_conf']
        self.iou_thresh: float = config['thresholds']['iou_thresh']
        self.overlap_frames_thresh: int = config['thresholds']['overlap_frames_thresh']
        
        # Freeze detection
        self.pixel_thresh: int = config['freeze']['pixel_thresh']
        self.freeze_frames_thresh: int = config['freeze']['freeze_frames_thresh']
        
        # Tracker configuration
        self.head_kp_index: int = config['tracker']['head_kp_index']
        self.kp_conf_threshold: float = config['tracker']['kp_conf_threshold']
        self.max_missing_frames: int = config['tracker']['max_missing_frames']
        self.history_len: int = config['tracker']['history_len']
        self.smooth_alpha: float = config['tracker']['smooth_alpha']
        self.max_speed: float = config['tracker']['max_speed']
        self.buffer_size: int = config['tracker']['buffer_size']
        self.min_delta: float = config['tracker']['min_delta']
        
        # Drawing configuration
        self.draw_color: Tuple[int, int, int] = tuple(config['drawing']['trajectory_color'])
        self.draw_thickness: int = config['drawing']['trajectory_thickness']
        self.bbox_color: Tuple[int, int, int] = tuple(config['drawing']['bbox_color'])
        self.freeze_color: Tuple[int, int, int] = tuple(config['drawing']['freeze_color'])
        self.overlap_color: Tuple[int, int, int] = tuple(config['drawing']['overlap_color'])
        self.head_color: Tuple[int, int, int] = tuple(config['drawing']['head_color'])
        self.body_color: Tuple[int, int, int] = tuple(config['drawing']['body_color'])
        self.tail_color: Tuple[int, int, int] = tuple(config['drawing']['tail_color'])
        self.line_color: Tuple[int, int, int] = tuple(config['drawing']['line_color'])
        
        # Grid filter
        self.grid_size: int = config['grid']['size']
        
        # Heatmap configuration
        self.heatmap_enabled: bool = config['heatmap']['enabled']
        self.heatmap_grid_shape: tuple = tuple(config['heatmap']['grid_shape'])
        self.heatmap_opacity: float = config['heatmap']['opacity']
        self.heatmap_overlap_thresh: float = config['heatmap']['overlap_thresh']
        self.heatmap_draw_grid: bool = config['heatmap']['draw_grid']
        self.heatmap_grid_alpha: float = config['heatmap']['grid_alpha']
        self.heatmap_max_dets_per_frame: int = config['heatmap'].get('max_detections_per_frame', 100)
