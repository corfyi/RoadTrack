import numpy as np
import cv2,json

def getUVError(h):
    u = 0.05*h
    v = 0.05*h
    if u>10:
        u = 10
    elif u<2:
        u = 2
    if v>10:
        v = 10
    elif v<2:
        v = 2
    return u,v

class Mapper(object):
    def __init__(self, H):
        self.iM_3x3 = H
        self.M_3x3 = np.linalg.inv(H)

    def compute_uv(self,x,y):
        xy1 = np.array([x,y,1])
        uv1 = self.M_3x3@xy1
        u = uv1[0]/uv1[2]
        v = uv1[1]/uv1[2]
        return u,v

    def compute_xy(self,u,v,h=1):
        uv1 = np.array([u,v,1])
        xy1 = self.iM_3x3@uv1
        s = 1/xy1[2]
        x = xy1[0]*s
        y = xy1[1]*s

        u_err,v_err = getUVError(h)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        C = np.zeros((2,2))
        C[0,0] = s*(self.iM_3x3[0,0] - x*self.iM_3x3[2,0])
        C[0,1] = s*(self.iM_3x3[0,1] - x*self.iM_3x3[2,1])
        C[1,0] = s*(self.iM_3x3[1,0] - y*self.iM_3x3[2,0])
        C[1,1] = s*(self.iM_3x3[1,1] - y*self.iM_3x3[2,1])
        sigma_xy = C@sigma_uv@C.T
            
        return x,y,sigma_xy
    
    
    def compute_gp_width(self,bbox):
        uv1 = np.array([bbox[0],bbox[3],1])
        xy1_s = self.iM_3x3@uv1
        s = 1/xy1_s[2]
        xy1 = xy1_s*s
        y1 = xy1[1]

        uv2 = np.array([bbox[2],bbox[3],1])
        xy2_s = self.iM_3x3@uv2
        s = 1/xy2_s[2]
        xy2 = xy2_s*s
        y2 = xy2[1]

        return abs(y2-y1) 
    
    def compute_pixel_width(self,x,y,w):
        xy1 = np.array([x,y-w/2,1])
        uv1 = self.M_3x3@xy1
        p1 = uv1[0]/uv1[2]

        xy1 = np.array([x,y+w/2,1])
        uv1 = self.M_3x3@xy1
        p2 = uv1[0]/uv1[2]

        return abs(p2-p1)
    
    def mapto(self,bbox):
        u = (bbox[0]+bbox[2])/2
        v = bbox[3]
        h = abs(bbox[3]-bbox[1])
        xp,yp,R = self.compute_xy(u,v,h)
        y = np.array([[xp],[yp]])
        return y,R