from scipy.optimize import linear_sum_assignment
import numpy as np

class CustomTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0

    def compute_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes [x1= [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
        return intersection / union if union > 0 else 0

    def update(self, detections):
        """Update tracks with new detections [x1, y1, x2, y2, score, label]."""
        self.frame_count += 1
        matched_detections = set()
        matched_tracks = set()

        # Compute cost matrix (negative IoU + category mismatch penalty)
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                iou = self.compute_iou(det[:4], track['bbox'])
                # Penalize if categories don't match
                category_match = 0 if det[5] == track['label'] else 1000
                cost_matrix[i, j] = -iou + category_match
                if iou < self.iou_threshold:
                    cost_matrix[i, j] = 1000  # High cost for low IoU

        # Hungarian algorithm for assignment
        if len(detections) > 0 and len(self.tracks) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < 500:  # Valid match (not penalized)
                    matched_detections.add(i)
                    matched_tracks.add(j)
                    self.tracks[j]['bbox'] = detections[i][:4]
                    self.tracks[j]['score'] = detections[i][4]
                    self.tracks[j]['last_updated'] = self.frame_count
                    self.tracks[j]['hits'] += 1

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.track_id_counter += 1
                self.tracks.append({
                    'id': self.track_id_counter,
                    'bbox': det[:4],
                    'score': det[4],
                    'label': det[5],
                    'hits': 1,
                    'last_updated': self.frame_count
                })

        # Remove old tracks
        self.tracks = [track for track in self.tracks if self.frame_count - track['last_updated'] <= self.max_age]

        # Return tracks that are confirmed (enough hits)
        confirmed_tracks = [track for track in self.tracks if track['hits'] >= self.min_hits]
        return [[track['bbox'][0], track['bbox'][1], track['bbox'][2], track['bbox'][3], track['id'], track['score'], track['label']] for track in confirmed_tracks]