import math

class Tracker:
    def __init__(self, iou_threshold=0.3):
        self.tracks = {}          # id : [x1, y1, x2, y2]
        self.id_count = 0
        self.iou_threshold = iou_threshold

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1b, y1b, x2b, y2b = box2

        inter_x1 = max(x1, x1b)
        inter_y1 = max(y1, y1b)
        inter_x2 = min(x2, x2b)
        inter_y2 = min(y2, y2b)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)

        union = area1 + area2 - inter_area

        if union == 0:
            return 0
        return inter_area / union

    def update(self, detections):
        objects_bbs_ids = []

        # Track assignment
        used_track_ids = set()

        for det in detections:
            x1, y1, x2, y2 = det

            best_iou = 0
            best_id = None

            # Find best matching existing track
            for tid, track_box in self.tracks.items():
                iou_score = self.iou(track_box, det)
                if iou_score > best_iou and iou_score > self.iou_threshold:
                    best_iou = iou_score
                    best_id = tid

            if best_id is None:
                # New person
                best_id = self.id_count
                self.tracks[best_id] = det
                self.id_count += 1
            else:
                # Update existing
                self.tracks[best_id] = det

            used_track_ids.add(best_id)
            objects_bbs_ids.append([x1, y1, x2, y2, best_id])

        # Remove lost tracks (optional)
        self.tracks = {tid: box for tid, box in self.tracks.items() if tid in used_track_ids}

        return objects_bbs_ids
