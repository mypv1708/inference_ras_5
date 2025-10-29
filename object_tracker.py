"""
Simple Object Tracker for Silkworm Detection
Maintains stable IDs across frames using distance-based matching
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class SilkwormTracker:
    """Simple tracker that maintains stable IDs based on head position distance."""
    
    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 10):
        """
        Initialize tracker.
        
        Args:
            max_distance: Maximum distance to match objects between frames
            max_disappeared: Max frames an object can be missing before removing
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        
        # Track objects: {object_id: {'head': (x,y), 'bbox': (x1,y1,x2,y2), 'disappeared': count}}
        self.objects: Dict[int, Dict] = {}
        self.next_id = 0
        
    def update(self, detections: List[Tuple]) -> List[Tuple]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (head, body, tail, bbox) tuples
            
        Returns:
            List of (obj_id, head, body, tail, bbox) tuples with stable IDs
        """
        if len(detections) == 0:
            # No detections - increment disappeared count for all objects
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]
            return []
        
        # Extract head positions for matching
        detection_heads = [det[0] for det in detections]  # det[0] is head position
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            tracked_objects = []
            for i, detection in enumerate(detections):
                head, body, tail, bbox = detection
                obj_id = self.next_id
                self.next_id += 1
                
                self.objects[obj_id] = {
                    'head': head,
                    'bbox': bbox,
                    'disappeared': 0
                }
                tracked_objects.append((obj_id, head, body, tail, bbox))
            return tracked_objects
        
        # Match existing objects to new detections
        object_centroids = np.array([obj['head'] for obj in self.objects.values()])
        detection_centroids = np.array(detection_heads)
        
        # Compute distance matrix
        distances = np.linalg.norm(object_centroids[:, np.newaxis] - detection_centroids, axis=2)
        
        # Find best matches (Hungarian algorithm simplified)
        object_ids = list(self.objects.keys())
        matched_object_indices = []
        matched_detection_indices = []
        
        # Simple greedy matching
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
                    if distances[obj_idx, det_idx] < min_dist:
                        min_dist = distances[obj_idx, det_idx]
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
            
            # Update object
            self.objects[obj_id]['head'] = head
            self.objects[obj_id]['bbox'] = bbox
            self.objects[obj_id]['disappeared'] = 0
            
            tracked_objects.append((obj_id, head, body, tail, bbox))
        
        # Register new objects for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detection_indices:
                head, body, tail, bbox = detection
                obj_id = self.next_id
                self.next_id += 1
                
                self.objects[obj_id] = {
                    'head': head,
                    'bbox': bbox,
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
