from filterpy.kalman import KalmanFilter
import numpy as np


class KalmanTracker(object):

    def __init__(self, y,R, dt=1/30,wx=0.5, wy=0.2, vmax=3,x0=None):
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.kf.R = R
        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))
    
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.kf.Q = np.dot(np.dot(G, Q0), G.T)

        self.kf.x[0] = y[0]
        self.kf.x[2] = y[1]
        self.kf.x[1] = 0
        self.kf.x[3] = 0
        if x0 is not None:
            if self.kf.x[0] > x0:
                self.kf.x[3] = -200
            else:
                self.kf.x[3] = 200

        self.xp = self.kf.x[0][0]

    def update(self, y,R):   
        self.kf.update(y,R)


    def predict(self):
        self.kf.predict()
        self.xp = self.xp * 0.9 + self.kf.x[0][0] * 0.1

    def maha_distance(self,y,R):
        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        logdet = np.log(np.linalg.det(S))
        r = mahalanobis[0,0] + logdet
        return r
    
    def eucli_distance(self,y):
        diff = y - np.dot(self.kf.H, self.kf.x)
        dist = np.dot(diff.T,diff)
        dist = np.sqrt(dist)
        return dist