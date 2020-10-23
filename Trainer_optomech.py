import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import tensorflow as tf
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common import make_vec_env
from gym_feedback.envs import OptomechEnv
from stable_baselines.common.schedules import LinearSchedule
import math
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,net_arch=[50,50,100],feature_extraction="mlp")

def cost(r):
    return r[0]**2+r[1]**2
def rew(r,sc,exc,pow):
    return -cost(r)    

e_c=0.01
feedback='Bayes'#'Markov' or 'Bayes'
steady=True #if True resets always with steady state conditions
N=1 #number of parallel workers
LRo=2.5e-4                              
TIMESTEPS=int(5e6)
sched_LR=LinearSchedule(1,LRo,0)
LR=sched_LR.value
qs=1e-2
randqset=False
if qs=='random':
    randqset=True
    qs=5e-4
title='feed{}_steady{}_lro{}_ts{}M_N{}_ec{}_C_Hammerer'.format(feedback,steady,LRo,TIMESTEPS/1e6,N,e_c)
#make checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=int(100000/N), save_path='./Boh/{}_q{}'.format(title,qs))
callback = checkpoint_callback
#set F matrix
zero=np.zeros((2,2))
F=np.block([[zero,zero],[zero,np.identity(2)]])
P=np.array([[1.,0,0,0],[0,1.,0,0],[0,0,0,0],[0,0,0,0]])
#set parameters and start training
params={'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':2*0.15,'detuning':None,'ne':3.5e5,'na':0,'phi':math.pi/2} #if a parameter is set to None it will be sampled from a uniform distribution at every reset
args={'feedback':feedback,'rewfunc':rew,'P':P,'F':F,'q':qs,'steadyreset':steady,'params':params}#i parametri di default son questi: rewfunc=Tools.purity_like_rew,F=np.identity(4),q=1e-4,dt=1e-3,plot=False,steadyreset=False,pow=0.5
env = make_vec_env(OptomechEnv,n_envs=N,env_kwargs=args) 
model=PPO2(CustomPolicy,env,learning_rate=LR,ent_coef=e_c,verbose=1,tensorboard_log='/home/fallani/prova/TRAIN_Boh/{}_q{}'.format(title,qs),seed=1)
model.learn(total_timesteps=TIMESTEPS,callback=callback,tb_log_name='{}_q{}'.format(title,qs))
model.save('/home/fallani/prova/MODELS{}_q{}'.format(title,qs))
        
