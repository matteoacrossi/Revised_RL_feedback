import numpy as np
import math
import Matrices as Mat
 
def system_step(r,sc,A,D,B,E,F,dt,dw,u):
    L=E-sc@B
    dr=A@r*dt+(2**(-0.5))*L@dw+F@u*dt
    r=r+dr
    dsc=dt*(A@sc+sc@(A.T)+D-L@(L.T))
    sc=sc+dsc
    return r,sc

def exc_step(exc,Dyn,L,dt):
    dexc=dt*(Dyn@exc+exc@(Dyn.T)+2*L@(L.T))
    exc=exc+dexc
    return exc

def purity_like_rew(r,sc,exc,pow=0.5):
    su=sc+exc
    d1=np.linalg.det(su)
    d2=np.linalg.det(sc)
    h=d1-d2+1
    h=1/(h**pow)
    return h

def Matrices_Calculator(mode,params):
    if mode=='Optomech':
        A,D,B,E=Mat.Optomech(params)
    elif mode=='Cavity':
        A,D,B,E=Mat.Cavity(params)
    else:
        print('select mode "Optomech" or "Cavity"')
    return A,D,B,E

def check_param(param,range,pos):
    if param==None:
        if pos==False:
            param=np.random.uniform(-range,range)
        if pos==True:
            param=np.random.uniform(0,range)
        return param
    else:
        return param

    
