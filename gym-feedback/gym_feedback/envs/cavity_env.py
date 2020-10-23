import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.linalg import sqrtm
import scipy
import math
import Utilities as Tools

    
class CavityEnv(gym.Env):
      
      metadata = {'render.modes': ['human']} #non so bene a che serva ma per ora lo tengo
      #provo a definire degli attributi che dovrebbero essere le cose che vanno tenute in memoria in esecuzione
      
      def __init__(self,feedback,rewfunc=Tools.purity_like_rew,F=np.identity(2),q=1e-4,dt=1e-3,plot=False,steadyreset=False,pow=0.5,params={'k':1,'eta':1,'X_kunit':0.499}):
              
              super(CavityEnv, self).__init__() 
              
              #calcolo le matrici e altre cose
              self.params=params
              k=Tools.check_param(self.params['k'],1,True)
              eta=Tools.check_param(self.params['eta'],1,True)
              X=Tools.check_param(self.params['X_kunit'],0.5,True)
              params={'k':k,'eta':eta,'X_kunit':X}
              self.rewfunc=rewfunc
              self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Cavity',params)
              A,D,B,E=self.A,self.D,self.B,self.E
              self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(2))
              self.F=F
              self.q=q
              self.Q=q*np.identity(2)
              self.pow=pow
              self.dw=np.random.randn(2)*(dt**0.5)
              self.dt=dt
              self.plot=plot
              self.steadyreset=steadyreset
              self.time=0
              self.r=np.zeros(2)
              self.sc=np.identity(2)
              self.exc=np.zeros((2,2))
              self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw
              self.feedback=feedback
              self.params=params
              self.reward=0
              self.action_space = spaces.Box( 
                low=-np.inf, high=np.inf,shape=(4,), dtype=np.float32)
              
              self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
             
              
              
      
      def step(self, action):
              
              time=self.time
              pow=self.pow
              feedback=self.feedback
              A,D,B,E,F,Q=self.A,self.D,self.B,self.E,self.F,self.Q
              dt=self.dt
              current=self.current
              r=self.r
              sc=self.sc
              exc=self.exc
              dw=self.dw
              J=action
              J=np.array([[J[0],J[1]],[J[2],J[3]]])


              if feedback=='Bayes':
                u=-J@r
                Dyn=A-F@J
                L=(E-sc@B)/(2**0.5)

              if feedback=='Markov':
                u=J@current/dt
                Dyn=A-(2**0.5)*F@J@(B.T)
                L=(E-sc@B)/(2**0.5)+F@J
              
              #qui aggiorno le variabili
              r,sc=Tools.system_step(r,sc,A,D,B,E,F,dt,dw,u)
              exc=Tools.exc_step(exc,Dyn,L,dt)
              Qcost=(u.T)@Q@u
              rew=self.rewfunc(r,sc,exc,pow)-Qcost
              
              
              #check sul tempo e aggiorno il tempo
              if time==1e3:
                 self.Done=True
              time+=1
              self.time=time

              #salvo le variabili
              self.sc=sc
              self.r=r
              self.exc=exc
              
              #qui c'è la misura
              self.dw=np.random.randn(2)*(dt**0.5)
              dw=self.dw
              self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw
              
              #if feedback=='Bayes':
              #  output=np.concatenate((np.array(self.r).flatten(),np.array(self.sc).flatten()),axis=0)
              #if feedback=='Markov':
              #  output=np.concatenate((np.array(self.current).flatten(),np.array(self.sc).flatten()),axis=0)
              parametri=np.array(list(self.params.items()))
              parametri=parametri[:,1].astype(np.float)
              output=np.array(self.sc).flatten()
              
              return output , rew , self.Done ,{'M(t)':J,'r':r,'current':self.current,'u':u,'su':self.sc+self.exc,'exc':self.exc,'sc':self.sc,'Qcost':Qcost,'params':parametri}
    
      

      def reset(self):
              
              dt=self.dt
              self.time=0
              plot=self.plot
              
              #reinizializzo delle cose
              if plot==True:
                a=1
                d=1
                n=3
                
              if plot==False:
                a=np.random.uniform(-1,1)
                d=np.random.uniform(-1,1)
                n=np.random.uniform(-0.5,3)
                k=Tools.check_param(self.params['k'],1,True)
                eta=Tools.check_param(self.params['eta'],1,True)
                X=Tools.check_param(self.params['X_kunit'],0.5,True)
                params={'k':k,'eta':eta,'X_kunit':X}
                self.params=params
                self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Cavity',params)
                A,D,B,E=self.A,self.D,self.B,self.E
                
                
                
              self.r=np.array([a,d])
              self.sc=(2*n+1)*np.array([[1,0],[0,1]])
              
                
              #setto le covarianze iniziali e i 'Done'
              if self.steadyreset or n<0:
                    self.r=np.array([0,0])
                    self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(2))
                    self.sc=np.copy(self.sigmacss)
             
              self.Done=False
              dw=np.random.randn(2)*(dt**0.5)
              self.dw=dw
              r=self.r
              current=-(2**0.5)*(self.B.T)@r*dt+dw
              self.current=current
              self.exc=np.zeros((2,2))

              #if feedback=='Bayes':
              #  output=np.concatenate((np.array(self.r).flatten(),np.array(self.sc).flatten()),axis=0)
              #if feedback=='Markov':
              #  output=np.concatenate((np.array(self.current).flatten(),np.array(self.sc).flatten()),axis=0)
              output=np.array(self.sc).flatten()

              return output
          
      
      
      def render(self,mode='human'):
              print(self.r,self.sc,self.exc)
    

