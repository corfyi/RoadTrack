import numpy as np
from lap import lapjv
from .kalman import KalmanTracker
from .mapper import Mapper

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
    def __init__(self, det,id,fps,mapper,wx,wy,vmax=3,x0=None):
        self.mapper = mapper
        bbox,conf,cls_id = det
        y,R = self.mapper.mapto(bbox)
        self.width_pred = abs(bbox[2]-bbox[0])
        self.height_pred = abs(bbox[3]-bbox[1])
        self.id = id
        self.conf = conf
        self.death_count = 0
        self.cls_id = cls_id
        self.hit_count = 1
        self.kf = KalmanTracker(y,R,dt=1/fps,wx=wx,wy=wy,vmax=vmax,x0=x0)
        self.bbox = bbox
        self.status = 1
        self.uv_pred = ((bbox[0]+bbox[2])/2,bbox[3])
        self.pred_bbox = bbox
        self.cost = 0
        self.age = 1
        self.iou = 1

    def distance(self,y,R,det):
        bbox,conf,cls_id = det
        maha_dist = self.kf.maha_distance(y,R)
        iou_dist = self.iou_distance(bbox)
        if iou_dist >= 0.9:
            iou_dist = 1000
        else:
            iou_dist = 0
        return maha_dist + iou_dist
    
    def iou_distance(self,bbox):
        # 计算两个box的iou
        tlwh1 = [self.pred_bbox[0], self.pred_bbox[1], self.pred_bbox[2]-self.pred_bbox[0], self.pred_bbox[3]-self.pred_bbox[1]]
        tlwh2 = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        r1 = tlwh1[0] + tlwh1[2]
        b1 = tlwh1[1] + tlwh1[3]
        r2 = tlwh2[0] + tlwh2[2]
        b2 = tlwh2[1] + tlwh2[3]

        left = max(tlwh1[0], tlwh2[0])
        top = max(tlwh1[1], tlwh2[1])
        right = min(r1, r2)
        bottom = min(b1, b2)

        if left < right and top < bottom:
            inter_area = (right - left) * (bottom - top)
            union_area = tlwh1[2] * tlwh1[3] + tlwh2[2] * tlwh2[3] - inter_area
            iou = inter_area / union_area
        else:
            iou = 0

        r = 1 - iou
        return r
    
    def is_confirmed(self):
        if self.status == 1:
            return True
        else:
            return False

    def update(self,det = None,cost = 0):
        if det is not None:
            bbox,conf,cls_id = det
            y,R = self.mapper.mapto(bbox)
            self.height_pred = abs(bbox[3]-bbox[1])
            self.width_pred = abs(bbox[2]-bbox[0])
            self.kf.update(y,R)
            self.bbox = bbox
            self.death_count = 0
            self.conf = conf
            self.age += 1
            if conf > 0.8:
                self.cls_id = cls_id
            if self.hit_count >= 3:
                self.status = 1
            else:
                self.hit_count += 1
        else:
            self.death_count += 1
            self.status = 2

        self.cost = cost

    def predict(self):
        self.kf.predict()
        xp,yp = self.get_xy()
        u,v = self.mapper.compute_uv(xp,yp)
        self.uv_pred = (u,v)
        self.pred_bbox = (u-self.width_pred/2,v-self.height_pred,u+self.width_pred/2,v)

    def get_bbox(self):
        return self.bbox
    
    
    def get_pred_bbox(self):
        return self.pred_bbox
    
    def get_uv_pred(self):
        return self.uv_pred
    
    def get_xy(self):
        x = self.kf.xp
        y = self.kf.kf.x[2][0]
        return x,y
    
    def get_speed(self):
        vx = self.kf.kf.x[1][0]
        vy = self.kf.kf.x[3][0]
        return vx,vy
    
class HomoTracker:
    def __init__(self,H,fps,conf_thresh = 0.3, match_thresh=50, cdt = 30, wx = 500, wy = 500, vmax= 300, x0 = None, iou_thresh=0.99):
        self.tracks = []
        self.frame_count = 0
        self.track_id = 0
        self.fps = fps
        self.conf_thresh = conf_thresh
        self.match_thresh = match_thresh
        self.iou_thresh = iou_thresh
        self.cdt = cdt
        self.wx = wx
        self.wy = wy
        self.x0 = x0
        self.vmax = vmax
        self.mapper = Mapper(H)
        self.deleted_ids = []

    def predict(self):
        for track in self.tracks:
            track.predict()

    def update(self, dets):
        num_d = len(dets)
        num_t = len(self.tracks)
        unmatched_d =[]
        unmatched_t = []
        cost_matrix = np.zeros((num_d,num_t))
        iou_matrix = np.zeros((num_d,num_t))
        
        flag = False
        if num_d !=0 and num_t != 0:
            for i in range(num_d):
                y,R = self.mapper.mapto(dets[i][0])
                for j in range(num_t):
                    cost_matrix[i,j] = self.tracks[j].distance(y,R,dets[i])
                    iou_matrix[i,j] = self.tracks[j].iou_distance(dets[i][0])
            matched_indices,unmatched_d,unmatched_t = linear_assignment(cost_matrix, self.match_thresh)

            for det_idx,trk_idx in matched_indices:
                bbox,conf,cls_id = dets[det_idx]
                cost = cost_matrix[det_idx,trk_idx]
                self.tracks[trk_idx].update((bbox,conf,cls_id),cost=cost)
                self.tracks[trk_idx].iou = 1
                for i in range(num_d):
                    if i == det_idx:
                        continue
                    if iou_matrix[i,trk_idx] < self.tracks[trk_idx].iou:
                        self.tracks[trk_idx].iou = iou_matrix[i,trk_idx]
        elif num_d == 0:
            unmatched_t = list(range(num_t)) 
        else:
            for i,(bbox,conf,cls_id) in enumerate(dets):
                unmatched_d.append(i)

        new_trk_detidx = []
        for det_idx in unmatched_d:
            bbox,conf,cls_id = dets[det_idx]
            if conf >= self.conf_thresh:
                if num_t == 0:
                    new_trk_detidx.append(det_idx)
                else:
                    if min(iou_matrix[det_idx,:]) > self.iou_thresh:
                        new_trk_detidx.append(det_idx)

        for trk_idx in unmatched_t:
            self.tracks[trk_idx].update()
    
        for i in new_trk_detidx:
            self.tracks.append(Track(dets[i],self.track_id,self.fps,self.mapper,self.wx,self.wy,self.vmax,self.x0))
            self.track_id += 1

        # remove dead tracks
        self.deleted_ids = [t.id for t in self.tracks if t.death_count >= self.cdt]
        self.tracks = [t for t in self.tracks if t.death_count < self.cdt]
