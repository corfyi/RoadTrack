import numpy as np
from lap import lapjv
from .kalman import KalmanTracker

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

class Track:
    def __init__(self, det,id):
        bbox,conf,cls_id = det
        self.id = id
        self.death_count = 0
        self.cls_id = cls_id
        self.hit_count = 1
        self.kf = KalmanTracker(bbox)
        self.bbox = bbox
        self.pred_bbox = bbox
        self.uv_pred = self.kf.get_uv()
        self.status = 0
        self.conf = conf
        self.age = 1

    def is_confirmed(self):
        if self.status == 1:
            return True
        else:
            return False
    
    def iou_distance(self,bbox):
        return self.kf.iou_distance(bbox)

    def update(self,det = None):
        if det is not None:
            bbox,conf,cls_id = det
            self.age += 1
            self.conf = conf
            self.kf.update(bbox)
            self.bbox = bbox
            self.death_count = 0
            if conf > 0.8:
                self.cls_id = cls_id
            if self.hit_count >= 2:
                self.status = 1
            else:
                self.hit_count += 1
        else:
            self.death_count += 1
            self.status = 2

    def predict(self):
        self.kf.predict()
        self.uv_pred = self.kf.get_uv()
        self.pred_bbox = self.kf.get_xyxy_box()

    def get_uv_pred(self):
        return self.uv_pred
    
    def get_pred_bbox(self):
        return self.pred_bbox

    def get_bbox(self):
        return self.bbox
    
    def get_xy(self):
        return 0,0

class BaselineTracker:
    def __init__(self,conf_thresh = 0.6, match_thresh=0.9, cdt = 20):
        self.tracks = []
        self.frame_count = 0
        self.track_id = 0
        self.conf_thresh = conf_thresh
        self.match_thresh = match_thresh
        self.cdt = cdt

    def predict(self):
        for track in self.tracks:
            track.predict()

    def update(self, dets):
        num_d = len(dets)
        num_t = len(self.tracks)
        unmatched_d =[]
        unmatched_t = []
        cost_matrix = np.zeros((num_d,num_t))

        if num_d !=0 and num_t != 0:
            for i in range(num_d):
                for j in range(num_t):
                    cost_matrix[i,j] = self.tracks[j].iou_distance(dets[i][0])
            matched_indices,unmatched_d,unmatched_t = linear_assignment(cost_matrix, self.match_thresh)

            for det_idx,trk_idx in matched_indices:
                self.tracks[trk_idx].update(dets[det_idx])
        elif num_d == 0:
            unmatched_t = list(range(num_t)) 
        else:
            for i,(bbox,conf,cls_id) in enumerate(dets):
                unmatched_d.append(i)

        for det_idx in unmatched_d:
            bbox,conf,cls_id = dets[det_idx]
            if conf >= self.conf_thresh:
                self.tracks.append(Track(dets[det_idx],self.track_id))
                self.track_id += 1
        for trk_idx in unmatched_t:
            self.tracks[trk_idx].update()
    
        # remove dead tracks
        self.deleted_ids = [t.id for t in self.tracks if t.death_count >= self.cdt]
        self.tracks = [t for t in self.tracks if t.death_count < self.cdt]
