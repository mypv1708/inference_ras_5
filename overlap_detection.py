import cv2
from config import Config
from utils import bbox_overlap, intersect, point_segment_distance


# Precomputed neighbor offsets to avoid recreating each call
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1), ( 0, 0), ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def process_overlap(silkworms, overlap_counters, frame, cfg: Config):
    """Optimized overlap detection with grid bucketing and early exits."""
    if len(silkworms) < 2:
        return

    # Grid bucketing with optimized center calculation
    grid = {}
    cell = getattr(cfg, 'grid_size', 100)
    for idx, (obj_id, h, b, t, box) in enumerate(silkworms):
        x1, y1, x2, y2 = box
        cx, cy = (x1+x2)//2, (y1+y2)//2
        gx, gy = cx//cell, cy//cell
        grid.setdefault((gx, gy), []).append((idx, obj_id, h, b, t, box))

    # OPTIMIZED: Pre-compute candidate pairs more efficiently
    candidate_pairs = set()
    for (gx, gy), objs in grid.items():
        for dx, dy in NEIGHBOR_OFFSETS:
            ng = (gx + dx, gy + dy)
            if ng not in grid:
                continue
                
            # OPTIMIZED: Avoid nested loops where possible
            for idx1, id1, h1, b1, t1, box1 in objs:
                for idx2, id2, h2, b2, t2, box2 in grid[ng]:
                    if idx1 < idx2:  # Avoid duplicate pairs
                        candidate_pairs.add((idx1, idx2))

    # OPTIMIZED: Batch process candidate pairs
    if not candidate_pairs:
        return
        
    # OPTIMIZED: Pre-compute segments to avoid repeated calculations
    segments_cache = {}
    
    sw_len = len(silkworms)
    for i, j in candidate_pairs:
        if i >= sw_len or j >= sw_len:
            continue
            
        id1, h1, b1, t1, box1 = silkworms[i]
        id2, h2, b2, t2, box2 = silkworms[j]
        
        # OPTIMIZED: Early bbox overlap check
        if not bbox_overlap(box1, box2):
            continue
            
        # OPTIMIZED: Cache segments computation
        key1 = (id1, h1, b1, t1)
        key2 = (id2, h2, b2, t2)
        
        if key1 not in segments_cache:
            segments_cache[key1] = [(h1, b1), (b1, t1)]
        if key2 not in segments_cache:
            segments_cache[key2] = [(h2, b2), (b2, t2)]
            
        segs1 = segments_cache[key1]
        segs2 = segments_cache[key2]
        
        # OPTIMIZED: Check intersection more efficiently
        key = tuple(sorted((id1, id2)))
        found = False
        
        # Early exit if any intersection found
        for s1 in segs1:
            if found:
                break
            for s2 in segs2:
                if intersect(s1[0], s1[1], s2[0], s2[1]):
                    found = True
                    break

        # If no hard intersection, check proximity: any keypoint of one
        # close to the other's segments within threshold
        if not found:
            thresh = cfg.point_segment_dist_thresh
            # keypoints for pair 1 and pair 2
            pts1 = [h1, b1, t1]
            pts2 = [h2, b2, t2]

            # Check points of worm1 to segments of worm2
            for p in pts1:
                if found:
                    break
                for s in segs2:
                    if point_segment_distance(p, s[0], s[1]) <= thresh:
                        found = True
                        break

            # Check points of worm2 to segments of worm1
            if not found:
                for p in pts2:
                    if found:
                        break
                    for s in segs1:
                        if point_segment_distance(p, s[0], s[1]) <= thresh:
                            found = True
                            break
        
        # OPTIMIZED: Single dict access
        current_count = overlap_counters.get(key, 0)
        if found:
            overlap_counters[key] = current_count + 1
            if overlap_counters[key] >= cfg.overlap_frames_thresh:
                # OPTIMIZED: Draw rectangles in batch
                cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), cfg.overlap_color, 2)
                cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), cfg.overlap_color, 2)
        else:
            overlap_counters[key] = max(0, current_count - 1)