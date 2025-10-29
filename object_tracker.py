"""
Simple Object Tracker for Silkworm Detection
Maintains stable IDs across frames using distance-based matching
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class SilkwormTracker:
    """Simple tracker for silkworms with improved matching stability."""
    
    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 10,
                 iou_weight: float = 0.6, distance_weight: float = 0.4,
                 new_track_min_center_dist: float = 15.0, new_track_max_iou: float = 0.3,
                 ema_alpha: float = 0.6):
        """
        Initialize silkworm tracker.
        
        Args:
            max_distance: Maximum distance to match objects between frames
            max_disappeared: Max frames an object can be missing before removing
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        self.new_track_min_center_dist = new_track_min_center_dist
        self.new_track_max_iou = new_track_max_iou
        self.ema_alpha = ema_alpha
        
        # Track objects with simplified data
        # {object_id: {'head': (x,y), 'body': (x,y), 'bbox': (x1,y1,x2,y2), 'disappeared': count}}
        self.objects: Dict[int, Dict] = {}
        self.next_id = 0
        self.frame_count = 0
        self.cleanup_interval = 100  # Cleanup every 100 frames
        
    def _calculate_body_center(self, head, body, tail):
        """Calculate body center from keypoints."""
        return ((head[0] + body[0] + tail[0]) / 3, (head[1] + body[1] + tail[1]) / 3)
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        a_area = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
        b_area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
        union = a_area + b_area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
    
    def _calculate_bbox_area(self, bbox):
        """Calculate bounding box area."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _periodic_cleanup(self):
        """Periodic cleanup to prevent memory accumulation."""
        self.frame_count += 1
        if self.frame_count % self.cleanup_interval == 0:
            # Remove old disappeared objects
            to_remove = []
            for obj_id, obj in self.objects.items():
                if obj['disappeared'] > self.max_disappeared:
                    to_remove.append(obj_id)
            
            for obj_id in to_remove:
                del self.objects[obj_id]
            
            # Reset frame count to prevent overflow
            if self.frame_count > 10000:
                self.frame_count = 0
        
    def update(self, detections: List[Tuple]) -> List[Tuple]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (head, body, tail, bbox) tuples
            
        Returns:
            List of (obj_id, head, body, tail, bbox) tuples with stable IDs
        """
        # Periodic cleanup
        self._periodic_cleanup()
        
        if len(detections) == 0:
            # No detections - increment disappeared count for all objects
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]
            return []
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            tracked_objects = []
            for detection in detections:
                head, body, tail, bbox = detection
                obj_id = self.next_id
                self.next_id += 1
                
                body_center = self._calculate_body_center(head, body, tail)
                
                self.objects[obj_id] = {
                    'head': head,
                    'body': body_center,
                    'bbox': bbox,
                    'disappeared': 0
                }
                tracked_objects.append((obj_id, head, body, tail, bbox))
            return tracked_objects
        
        # Hybrid matching using IoU and center distance
        object_ids = list(self.objects.keys())
        matched_object_indices = []
        matched_detection_indices = []
        
        # Calculate cost matrix for all possible matches
        cost_matrix = np.full((len(object_ids), len(detections)), np.inf)
        
        for obj_idx, obj_id in enumerate(object_ids):
            obj = self.objects[obj_id]
            
            for det_idx, detection in enumerate(detections):
                head, body, tail, bbox = detection
                body_center = self._calculate_body_center(head, body, tail)
                det_center = self._bbox_center(bbox)
                
                # Distance component (normalized to [0,1])
                center_dist = self._calculate_distance(self._bbox_center(obj['bbox']), det_center)
                dist_cost = min(1.0, center_dist / max(self.max_distance, 1e-6))
                
                # IoU component (converted to cost)
                iou = self._bbox_iou(obj['bbox'], bbox)
                iou_cost = 1.0 - iou
                
                # Combined cost
                cost = self.iou_weight * iou_cost + self.distance_weight * dist_cost
                cost_matrix[obj_idx, det_idx] = cost
        
        # Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        try:
            obj_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            # Filter matches by distance threshold
            for obj_idx, det_idx in zip(obj_indices, det_indices):
                # Convert cost back to rough distance gate using components
                if cost_matrix[obj_idx, det_idx] <= 0.8:  # heuristic threshold
                    matched_object_indices.append(obj_idx)
                    matched_detection_indices.append(det_idx)
        except ImportError:
            # Fallback to greedy matching if scipy not available
            for _ in range(min(len(object_ids), len(detections))):
                min_dist = np.inf
                best_obj_idx = -1
                best_det_idx = -1
                
                for obj_idx in range(len(object_ids)):
                    if obj_idx in matched_object_indices:
                        continue
                    for det_idx in range(len(detections)):
                        if det_idx in matched_detection_indices:
                            continue
                        if cost_matrix[obj_idx, det_idx] < min_dist:
                            min_dist = cost_matrix[obj_idx, det_idx]
                            best_obj_idx = obj_idx
                            best_det_idx = det_idx
                
                if min_dist <= 0.8:
                    matched_object_indices.append(best_obj_idx)
                    matched_detection_indices.append(best_det_idx)
                else:
                    break
        
        # Update matched objects
        tracked_objects = []
        for obj_idx, det_idx in zip(matched_object_indices, matched_detection_indices):
            obj_id = object_ids[obj_idx]
            head, body, tail, bbox = detections[det_idx]
            
            # Calculate new body center
            body_center = self._calculate_body_center(head, body, tail)
            
            # EMA smoothing for center and head
            prev_head = self.objects[obj_id]['head']
            prev_body = self.objects[obj_id]['body']
            alpha = self.ema_alpha
            sm_head = (int(alpha * head[0] + (1 - alpha) * prev_head[0]),
                       int(alpha * head[1] + (1 - alpha) * prev_head[1]))
            sm_body = (alpha * body_center[0] + (1 - alpha) * prev_body[0],
                       alpha * body_center[1] + (1 - alpha) * prev_body[1])

            # Update object
            self.objects[obj_id].update({
                'head': sm_head,
                'body': sm_body,
                'bbox': bbox,
                'disappeared': 0
            })
            
            tracked_objects.append((obj_id, head, body, tail, bbox))
        
        # Register new objects for unmatched detections (gate near existing)
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detection_indices:
                head, body, tail, bbox = detection
                det_center = self._bbox_center(bbox)
                too_close = False
                for existing in self.objects.values():
                    center_dist = self._calculate_distance(self._bbox_center(existing['bbox']), det_center)
                    iou = self._bbox_iou(existing['bbox'], bbox)
                    if center_dist < self.new_track_min_center_dist or iou > self.new_track_max_iou:
                        too_close = True
                        break
                if too_close:
                    continue

                obj_id = self.next_id
                self.next_id += 1

                body_center = self._calculate_body_center(head, body, tail)
                area = self._calculate_bbox_area(bbox)
                
                self.objects[obj_id] = {
                    'head': head,
                    'body': body_center,
                    'bbox': bbox,
                    'velocity': (0.0, 0.0),
                    'size': area,
                    'disappeared': 0
                }
                tracked_objects.append((obj_id, head, body, tail, bbox))
        
        # Increment disappeared count for unmatched objects
        for obj_idx, obj_id in enumerate(object_ids):
            if obj_idx not in matched_object_indices:
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]
        
        return tracked_objects
    
    def get_object_count(self) -> int:
        """Get current number of tracked objects."""
        return len(self.objects)
    
    def get_object_info(self, obj_id: int) -> Optional[Dict]:
        """Get information about a specific object."""
        return self.objects.get(obj_id)
