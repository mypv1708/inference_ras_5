"""
Simple Object Tracker for Silkworm Detection
Maintains stable IDs across frames using distance-based matching
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class SilkwormTracker:
    """Simple tracker for silkworms with multi-point tracking."""
    
    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 10):
        """
        Initialize silkworm tracker.
        
        Args:
            max_distance: Maximum distance to match objects between frames
            max_disappeared: Max frames an object can be missing before removing
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        
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
        
        # Simple distance-based matching
        object_ids = list(self.objects.keys())
        matched_object_indices = []
        matched_detection_indices = []
        
        # Calculate distance matrix for all possible matches
        distance_matrix = np.full((len(object_ids), len(detections)), np.inf)
        
        for obj_idx, obj_id in enumerate(object_ids):
            obj = self.objects[obj_id]
            
            for det_idx, detection in enumerate(detections):
                head, body, tail, bbox = detection
                body_center = self._calculate_body_center(head, body, tail)
                
                # Calculate distances to current positions
                head_dist = self._calculate_distance(obj['head'], head)
                body_dist = self._calculate_distance(obj['body'], body_center)
                
                # Combined distance (weighted average)
                combined_dist = (head_dist + body_dist) / 2
                distance_matrix[obj_idx, det_idx] = combined_dist
        
        # Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        try:
            obj_indices, det_indices = linear_sum_assignment(distance_matrix)
            
            # Filter matches by distance threshold
            for obj_idx, det_idx in zip(obj_indices, det_indices):
                if distance_matrix[obj_idx, det_idx] <= self.max_distance:
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
                        if distance_matrix[obj_idx, det_idx] < min_dist:
                            min_dist = distance_matrix[obj_idx, det_idx]
                            best_obj_idx = obj_idx
                            best_det_idx = det_idx
                
                if min_dist <= self.max_distance:
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
            
            # Update object
            self.objects[obj_id].update({
                'head': head,
                'body': body_center,
                'bbox': bbox,
                'disappeared': 0
            })
            
            tracked_objects.append((obj_id, head, body, tail, bbox))
        
        # Register new objects for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detection_indices:
                head, body, tail, bbox = detection
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
