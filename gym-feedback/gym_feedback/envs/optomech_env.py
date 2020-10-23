import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand
import numpy as np
from scipy.linalg import sqrtm
import math
import scipy
import Utilities as Tools

    
class OptomechEnv(gym.Env):
      
      metadata = {'render.modes': ['human']} #non so bene a che serva ma per ora lo tengo

      def __init__(self,feedback,P,rewfunc=Tools.purity_like_rew,F=np.identity(4),q=1e-4,dt=1e-3,plot=False,steadyreset=False,pow=0.5,params={'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':2*0.15,'detuning':1,'ne':3.5e5,'na':0,'phi':math.pi/2}):
              
              super(OptomechEnv, self).__init__()
             
              self.params=params
              wm=Tools.check_param(self.params['wm'],1,True)
              k=Tools.check_param(self.params['k'],3,True)
              g=Tools.check_param(self.params['g'],1,True)
              y=Tools.check_param(self.params['y'],1,True)
              eta=Tools.check_param(self.params['eta'],1,True)
              detuning=Tools.check_param(self.params['detuning'],1,True)
              ne=Tools.check_param(self.params['ne'],0.5,True)
              na=Tools.check_param(self.params['na'],0.5,True)
              phi=Tools.check_param(self.params['phi'],0.5,True)
              params={'wm':wm,'k':k,'y':y,'eta':eta,'g':g,'detuning':detuning,'ne':ne,'na':na,'phi':phi}
              #calcolo le matrici e altre cose
              self.rewfunc=rewfunc
              self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Optomech',params)
              A,D,B,E=self.A,self.D,self.B,self.E
              
              self.P=P #serve solo per l'agente ottimo
              self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(4))
              self.F=F
              self.q=q
              self.Q=q*np.identity(4)
              self.pow=pow
              self.dw=np.random.randn(4)*(dt**0.5)
              self.dt=dt
              self.plot=plot
              self.steadyreset=steadyreset
              self.time=0
              self.r=np.zeros(4)
              self.sc=np.identity(4)
              self.exc=np.zeros((4,4))
              self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw
              self.feedback=feedback
              self.params=params
            
              self.action_space = spaces.Box( 
                low=-np.inf, high=np.inf,shape=(16,), dtype=np.float32)
              
             
              self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
             
              
              
              
      
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
        J=np.array([[J[0],J[1],J[2],J[3]],[J[4],J[5],J[6],J[7]],[J[8],J[9],J[10],J[11]],[J[12],J[13],J[14],J[15]]])

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
        self.dw=np.random.randn(4)*(dt**0.5)
        dw=self.dw
        self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw
        
        #if feedback=='Bayes':
        #  output=np.concatenate((np.array(self.r).flatten(),np.array(self.sc).flatten()),axis=0)
        #if feedback=='Markov':
        #  output=np.concatenate((np.array(self.current).flatten(),np.array(self.sc).flatten()),axis=0)
        #output=np.array(self.sc).flatten()
        parametri=np.array(list(self.params.items()))
        parametri=parametri[:,1].astype(np.float)
        output=np.concatenate((parametri,np.array(self.sc).flatten()),axis=0)
        
        return output , rew , self.Done ,{'M(t)':J,'r':r,'current':self.current,'u':u,'su':sc+exc,'exc':exc,'sc':sc,'Qcost':Qcost,'params':parametri}
    
      

      def reset(self):
              
        dt=self.dt
        self.time=0
        plot=self.plot
        
        #reinizializzo delle cose
        if plot==True:
          a=1.
          b=1.
          c=1.
          d=1.
          n=3
          
        if plot==False:
          a=np.random.uniform(-1,1)
          b=np.random.uniform(-1,1)
          c=np.random.uniform(-1,1)
          d=np.random.uniform(-1,1)
          n=np.random.uniform(-1,5)
          wm=Tools.check_param(self.params['wm'],1,True)
          k=Tools.check_param(self.params['k'],3,True)
          g=Tools.check_param(self.params['g'],1,True)
          y=Tools.check_param(self.params['y'],1,True)
          eta=Tools.check_param(self.params['eta'],1,True)
          detuning=Tools.check_param(self.params['detuning'],1,True)
          ne=Tools.check_param(self.params['ne'],0.5,True)
          na=Tools.check_param(self.params['na'],0.5,True)
          phi=Tools.check_param(self.params['phi'],0.5,True)
          params={'wm':wm,'k':k,'y':y,'eta':eta,'g':g,'detuning':detuning,'ne':ne,'na':na,'phi':phi}
          self.params=params
          self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Optomech',params)
          A,D,B,E=self.A,self.D,self.B,self.E
          
          
          
        self.r=np.array([a,b,c,d])
        self.sc=(2*n+1)*np.identity(4)
        
          
        #setto le covarianze iniziali e i 'Done'
        if self.steadyreset or n<0:
              self.r=np.zeros(4)
              self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(4))
              self.sc=np.copy(self.sigmacss)
        
        self.Done=False
        dw=np.random.randn(4)*(dt**0.5)
        self.dw=dw
        r=self.r
        current=-(2**0.5)*(self.B.T)@r*dt+dw
        self.current=current
        self.exc=np.zeros((4,4))

        #if feedback=='Bayes':
        #  output=np.concatenate((np.array(self.r).flatten(),np.array(self.sc).flatten()),axis=0)
        #if feedback=='Markov':
        #  output=np.concatenate((np.array(self.current).flatten(),np.array(self.sc).flatten()),axis=0)
        #output=np.array(self.sc).flatten()
        parametri=np.array(list(self.params.items()))
        parametri=parametri[:,1].astype(np.float)
        output=np.concatenate((parametri,np.array(self.sc).flatten()),axis=0)

        return output
          
      
      
      def render(self,mode='human'):
              print(self.r,self.sc,self.exc)
    

