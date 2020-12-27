# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:49 2020

@author: rpira
"""

from numpy.random import randn
from random import choice
from  OPTFinal import * # Importin the optimization objective function
import pickle
import matplotlib.pyplot as plt
import numpy as np


## Making the model and the data
Portion=1
Feature='Energy'
InputData,Output,Model=Output_moddel_Data(Portion, Feature)

## Defining the k-fold and 
K_fold=3

## Definig a dummy function to make sure the Randomized search is working
c=np.array(list(range(-1000,1000,10)))
values=c**2

## Defining the optimization treashold parameters
MapeOpt=50
TreasholdCount=150
TreasureholdAccuracy=.2
count=0
k=0
m=0
z=0
# Cross over

def CrossOver(CountParam,K_fold,InputData,Output,Mode):
    Scores=[]
    for CrossCount in range(1,K_fold+1):
        mape =compile_model(CountParam,CrossCount,K_fold,InputData,Output,Model)
        Scores.append(mape)
    CrossMape=np.mean(Scores)
    return CrossMape

## Save the optimal model
ModelFile='OptModel'+Feature
ModelP = open(ModelFile, 'wb')

tot=[]
for CountParam in range(0,len(Model)):
    count=count+1
    
#    mape=choice(values)
    mape=CrossOver(CountParam,K_fold,InputData,Output,Model)
    if MapeOpt-mape>TreasureholdAccuracy:
        count=0
        z=z+1
        TreasholdCount=50*z+TreasholdCount
        print("diff=",MapeOpt-mape)	
    if mape<MapeOpt:
        print("diff=",MapeOpt-mape)
        MapeOpt=mape
        tot.append(MapeOpt)
        m=m+1
#        print(mape)
        ModelInfo=Model[CountParam]
        pickle.dump(ModelInfo, ModelP)
        print("bestModel",ModelInfo)
        
    
    if count>TreasholdCount:
       
        break
print("")
print("Resetting the counter=",z)
print("Number of iteration MAPE reduces=",m)
print("Number of iteration MAPE reduces without resettin=",m-z)
print("Opt Iteration=",z+count)
print("Treashold Count=",TreasholdCount)
print("Count=",count)

#for i in range(0,np.shape(c)[0]):
##    k=k+1
#    count=count+1
##    saved_model, mape  =compile_model(ModelInfo[i])
#    mape=values[i]
#    print("mape=",mape)
#    if mape<MapeOpt:
#        MapeOpt=mape
#        k=k+1
#        tot.append(MapeOpt)
#        print("MinMape=",mape)
#        count=0        
#    if count>TreasholdCount:
#        break
print("mape=",tot)
        
plt.plot(values)
plt.plot(tot)
#plt.xlabel('Data ')
#plt.ylabel('Predictions [Energy]')
plt.figure() 
plt.plot(tot)
#plt.xlabel('True Values [Energy]')
#plt.ylabel('Predictions [Energy]')
#plt.plot()
