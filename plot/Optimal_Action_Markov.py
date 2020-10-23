import gym
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm

class Optimal_Agent:
        
    def __init__(self,env):

        #setto la roba dall'ambiente
        self.dt=env.dt
        self.E=env.E
        self.B=env.B
        self.F=env.F
        self.Finv=np.linalg.inv(self.F)
        self.sc=env.sc
        self.Env=env
        
        
    def predict(self):
        sc=self.Env.sc
        
        E=self.E
        B=self.B
        Finv=self.Finv        
        
        M=-Finv.dot(E-sc.dot(B))*(2**(-0.5))
        M=M#/self.action_scale
        self.sc=sc
        return M.flatten()
        