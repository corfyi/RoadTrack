from filterpy.kalman import KalmanFilter
import numpy as np


sigma_p = 0.05
sigma_v = 0.00625

class KalmanTracker(object):


    def __init__(self, bbox):
        
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 1, 0, 0, 0, 0, 0, 0], 
                               [0, 1, 0, 0, 0, 0, 0, 0], 
                               [0, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 1],
                               [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0]
                               ])
        self.kf.R = np.identity(4)
        self.kf.P = np.identity(8)
        self.kf.P[1,1] = 10
        self.kf.P[3,3] = 10
        self.kf.Q = np.identity(8)
        
        xywh = [(bbox[0]+bbox[2])/2, bbox[3], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        self.kf.x = np.zeros((8,1))
        for i in range(4):
            self.kf.x[2*i] = xywh[i]
        self.kf.R[0,0] = xywh[0]*0.1
        self.kf.R[1,1] = xywh[1]*0.05
        self.kf.R[2,2] = xywh[2]*0.1
        self.kf.R[3,3] = xywh[3]*0.05


    def update(self, bbox):   
        xywh = [(bbox[0]+bbox[2])/2, bbox[3], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        R_xywh = np.identity(4)
        R_xywh[0,0] = xywh[0]*0.1
        R_xywh[1,1] = xywh[1]*0.05
        R_xywh[2,2] = xywh[2]*0.1
        R_xywh[3,3] = xywh[3]*0.05
        
        for i in range(4):
            self.kf.Q[2*i,2*i] = pow(self.kf.x[2*i][0]*sigma_p,2)
            self.kf.Q[2*i+1,2*i+1] = pow(self.kf.x[2*i+1][0]*sigma_p,2)
        self.kf.update(xywh,R_xywh)

    def predict(self):
        self.kf.predict()

    def get_uv(self):
        u = self.kf.x[0][0]
        v = self.kf.x[2][0]
        return u,v
    
    def get_tlwh_box(self):
        tlwh = [self.kf.x[0][0]-self.kf.x[4][0]/2, self.kf.x[2][0]-self.kf.x[6][0], self.kf.x[4][0], self.kf.x[6][0]]
        return tlwh
    
    def get_xyxy_box(self):
        xyxy = [self.kf.x[0][0]-self.kf.x[4][0]/2, self.kf.x[2][0]-self.kf.x[6][0], self.kf.x[0][0]+self.kf.x[4][0]/2, self.kf.x[2][0]]
        return xyxy
    
    def iou_distance(self,bbox):
        # 计算两个box的iou
        tlwh1 = [self.kf.x[0]-self.kf.x[4]/2, self.kf.x[2]-self.kf.x[6], self.kf.x[4], self.kf.x[6]]
        tlwh2 = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        if tlwh1[2] > 80 and tlwh1[3] > 80:
            is_large = True
        else:
            is_large = False

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


