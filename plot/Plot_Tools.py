import numpy as np

def step(model,det,env,obs,r,current,sc,su,exc,rew,azione,Qcost,opt=False):
    if opt==False:
        action, _states = model.predict(obs,deterministic=det)
    else:
        action = model.predict()
    obs, rewards, dones, info = env.step(action)
    r.append(info['r'])
    current.append(info['current'])
    sc.append(info['sc'])
    su.append(info['su'])
    exc.append(info['exc'])
    rew.append(rewards)
    azione.append(info['M(t)'])
    Qcost.append(info['Qcost'])
    return obs

                  
    

def media_cammini(N,steps,env,det,optenv,model,opt_model,Optimal_Agent):
    
    #resetto l'ambiente
    obs = env.reset()
    optenv.reset()    
    
    #inizializzo le liste per formare i vettori relativi alle singole realizzazioni
    #qui per l'agente allenato
    r=[]
    current=[]
    sc=[]
    su=[]
    exc=[]
    rew=[]
    Qcost=[]
    azione=[]
    
    r2=[]
    current2=[]
    sc2=[]
    su2=[]
    exc2=[]
    rew2=[]
    Qcost2=[]
    azione2=[]
    
    #inizializzo le tabelle
    R=[]
    Current=[]
    Sc=[]
    Su=[]
    Exc=[]
    Rew=[]
    QCOST=[]
    Azione=[]
    R2=[]
    Current2=[]
    Sc2=[]
    Su2=[]
    Exc2=[]
    Rew2=[]
    QCOST2=[]
    Azione2=[]
    
    for j in range(0,N): #cammini
    
      for k in range(0,steps): #steps di integrazione
      
        
        obs=step(model,det,env,obs,r,current,sc,su,exc,rew,azione,Qcost)
        
        step(opt_model,det,optenv,obs,r2,current2,sc2,su2,exc2,rew2,azione2,Qcost2,opt=True)
        
        
      
      #resetto l'ambiente
      
      obs=env.reset()
      optenv.reset()
      opt_model=Optimal_Agent(optenv)
    
      #salvo sulle tabelle
      R.append(r)
      Current.append(current)
      Sc.append(sc)
      Su.append(su)
      Exc.append(exc)
      Rew.append(rew)
      QCOST.append(Qcost)
      Azione.append(azione)
    
      R2.append(r2)
      Current2.append(current2)
      Sc2.append(sc2)
      Su2.append(su2)
      Exc2.append(exc2)
      Rew2.append(rew2)
      QCOST2.append(Qcost2)
      Azione2.append(azione2)
    
      #reinizializzo i vettorini delle realizzazioni
      r=[]
      current=[]
      sc=[]
      su=[]
      exc=[]
      rew=[]
      Qcost=[]
      azione=[]
        
      r2=[]
      current2=[]
      sc2=[]
      su2=[]
      exc2=[]
      rew2=[]
      Qcost2=[]
      azione2=[]  

      #fine del for su N
    
    
    #faccio la media sulle realizzazioni
    rmean=np.mean(np.array(R),axis=0)
    rmean2=np.mean(np.array(R2),axis=0)
    scmean=np.mean(np.array(Sc),axis=0)
    scmean2=np.mean(np.array(Sc2),axis=0)
    sumean=np.mean(np.array(Su),axis=0)
    sumean2=np.mean(np.array(Su2),axis=0)
    excmean=np.mean(np.array(Exc),axis=0)
    excmean2=np.mean(np.array(Exc2),axis=0)
    currentmean=np.mean(np.array(Current),axis=0)
    currentmean2=np.mean(np.array(Current2),axis=0)
    rewmean=np.mean(np.array(Rew),axis=0)
    rewmean2=np.mean(np.array(Rew2),axis=0)
    Qcostmean=np.mean(np.array(QCOST),axis=0)
    Qcostmean2=np.mean(np.array(QCOST2),axis=0)
    azionemean=np.mean(np.array(Azione),axis=0)
    azionemean2=np.mean(np.array(Azione2),axis=0)
    
    #salvo le std
    rstd=np.std(np.array(R),axis=0)
    rstd2=np.std(np.array(R2),axis=0)
    scstd=np.std(np.array(Sc),axis=0)
    scstd2=np.std(np.array(Sc2),axis=0)
    sustd=np.std(np.array(Su),axis=0)
    sustd2=np.std(np.array(Su2),axis=0)
    excstd=np.std(np.array(Exc),axis=0)
    excstd2=np.std(np.array(Exc2),axis=0)
    currentstd=np.std(np.array(Current),axis=0)
    currentstd2=np.std(np.array(Current2),axis=0)
    rewstd=np.std(np.array(Rew),axis=0)
    rewstd2=np.std(np.array(Rew2),axis=0)
    Qcoststd=np.std(np.array(QCOST),axis=0)
    Qcoststd2=np.std(np.array(QCOST2),axis=0)
    azionestd=np.std(np.array(Azione),axis=0)
    azionestd2=np.std(np.array(Azione2),axis=0)
    
    return {'rmean':rmean,'rmean2':rmean2,'scmean':scmean,'scmean2':scmean2,'sumean':sumean,'sumean2':sumean2,'excmean':excmean,'excmean2':excmean2,'currentmean':currentmean,'currentmean2':currentmean2,'rewmean':rewmean,'rewmean2':rewmean2,'Qcostmean':Qcostmean,'Qcostmean2':Qcostmean2,'azionemean':azionemean,'azionemean2':azionemean2,'rstd':rstd,'rstd2':rstd2,'scstd':scstd,'scstd2':scstd2,'sustd':sustd,'sustd2':sustd2,'excstd':excstd,'excstd2':excstd2,'currentstd':currentstd,'currentstd2':currentstd2,'rewstd':rewstd,'rewstd2':rewstd2,'Qcoststd':Qcoststd,'Qcoststd2':Qcoststd2,'azionestd':azionestd,'azionestd2':azionestd2}
