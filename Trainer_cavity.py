import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO1,PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import tensorflow as tf
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common import make_vec_env
from gym_feedback.envs import CavityEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,net_arch=[16],feature_extraction="mlp")
e_c=0.01
feedback='Markov'#'Markov' or 'Bayes'
steady=False#False #if True resets always with steady state conditions
N=1 #number of parallel workers
LRo=2.5e-4                             
TIMESTEPS=int(100e6)
sched_LR=LinearSchedule(1,LRo,0)
LR=sched_LR.value
qs=0
title='feed{}_steady{}_lro{}_ts{}M_N{}_ec{}_0.5_acsc'.format(feedback,steady,LRo,TIMESTEPS/1e6,N,e_c)
#make checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=int(100000/N), save_path='./Revised/{}_q{}'.format(title,qs))
callback = checkpoint_callback
#set parameters and start training
params={'k':1,'eta':1,'X_kunit':0.499} #if a parameter is set to None it will be sampled from a uniform distribution at every reset
args={'feedback':feedback,'q':qs,'steadyreset':steady,'pow':0.5,'params':params}#i parametri di default son questi: rewfunc=Tools.purity_like_rew,q=1e-4,dt=1e-3,plot=False,pow=0.5
env = make_vec_env(CavityEnv,n_envs=N,env_kwargs=args) 
model=PPO2(MlpPolicy,env,learning_rate=LR,ent_coef=e_c,verbose=1,tensorboard_log='/home/fallani/prova/TRAIN_Revised/{}_q{}'.format(title,qs),seed=1)
model.learn(total_timesteps=TIMESTEPS,callback=callback,tb_log_name='{}_q{}'.format(title,qs))
model.save('/home/fallani/prova/MODELS{}_q{}'.format(title,qs))
        
