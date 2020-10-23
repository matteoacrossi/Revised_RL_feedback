import gym
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm

class Optimal_Agent:
        
    def __init__(self,env):

        #setto la roba dall'ambiente
        self.dt=env.dt
        self.A=env.A
        self.Q=env.Q
        self.Qinv=np.linalg.inv(self.Q)
        self.Y=0*np.copy(self.Qinv)
        self.P=env.P
        self.F=env.F

    def predict(self):
        
        Y=self.Y
        P=self.P
        Qinv=self.Qinv
        F=self.F
        A=self.A
        dt=self.dt
        
        Y=Y+dt*( Y.dot(A.T)+A.dot(Y) + P - Y.dot(F.dot(Qinv.dot((F.T).dot(Y)))))
        K=Qinv.dot((F.T).dot(Y))
        self.Y=Y
        K=K#/100
        
        return K.flatten()
        