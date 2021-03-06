

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
import pickle
from random import shuffle
import random
## Make the dictionary for the values
ModelInfo={}
ModelInfo['Nerouns_L1']=[2560,80,3000,1600]
ModelInfo['Nerouns_L2']=[160,60,400]
ModelInfo['Nerouns_L3']=[160,60,200]

ModelInfo['Layers']=[2,3]

ModelInfo['Dropout_Value_L1']=[0,0.2,0.4]
ModelInfo['Dropout_Value_L2']=[0,0.2,0.4]
ModelInfo['Dropout_Value_L3']=[0,0.2,0.4]

ModelInfo['Reguralization_L1']=[l1(0.1),l2(0.2),l2(0.1),l1(0.2),l2(0.3),l2(0.4)]
ModelInfo['Reguralization_L2']=[l1(0.1),l2(0.2),l2(0.1),l1(0.2),l2(0.3),l2(0.4)]
ModelInfo['Reguralization_L3']=[l1(0.1),l2(0.2),l2(0.1),l1(0.2),l2(0.3),l2(0.4)]


ModelInfo['kernel_constraint_L1']=[max_norm(3),max_norm(5)]
ModelInfo['kernel_constraint_L2']=[max_norm(3),max_norm(5)]
ModelInfo['kernel_constraint_L3']=[max_norm(3),max_norm(5)]

ModelInfo['Activation_Method']=['relu','sigmoid']
ModelInfo['Epochs']=[50]
ModelInfo['Batches']=[32]
ModelInfo['optimizer']=['Nadam','Adam']

ModelInfo['W_Initialization_Method_L1']=[keras.initializers.glorot_uniform(seed=None),keras.initializers.lecun_normal(seed=None)]
ModelInfo['W_Initialization_Method_L2']=[keras.initializers.glorot_uniform(seed=None),keras.initializers.lecun_normal(seed=None)]
ModelInfo['W_Initialization_Method_L3']=[keras.initializers.glorot_uniform(seed=None),keras.initializers.lecun_normal(seed=None)]

TotalModelInfo=[]
ModelInfoMade={}
EnergyParamF = open('EnergyParam.obj', 'wb')
pickle.dump(TotalModelInfo, EnergyParamF)
g=0
x=0
ratio=0.1			## ratio of choosing random samples from the made database
num=1000000  ## number of procceed before saving the data
SampleQuantity=num*ratio
for i1 in (ModelInfo['Nerouns_L1']):
    for i2 in (ModelInfo['Nerouns_L2']):
        for i3 in (ModelInfo['Nerouns_L3']):
            
            for j in (ModelInfo['Layers']):
                
                for ll1 in (ModelInfo['Dropout_Value_L1']):
                    for ll2 in (ModelInfo['Dropout_Value_L2']):
                        for ll3 in (ModelInfo['Dropout_Value_L3']):
                            
                            for m1 in ModelInfo['Reguralization_L1']:
                                for m2 in ModelInfo['Reguralization_L2']:
                                    for m3 in ModelInfo['Reguralization_L3']:
                                        
                                        for n1 in ModelInfo['kernel_constraint_L1']:
                                            for n2 in ModelInfo['kernel_constraint_L2']:
                                                for n3 in ModelInfo['kernel_constraint_L3']:
                                                    
                                                    for o in ModelInfo['Activation_Method']:
                                                        for p in ModelInfo['Epochs']:
                                                            for q in ModelInfo['Batches']:
                                                                for r in ModelInfo['optimizer']:
                                                                    
                                                                     for s1 in ModelInfo['W_Initialization_Method_L1']:
                                                                         for s2 in ModelInfo['W_Initialization_Method_L2']:
                                                                             for s3 in ModelInfo['W_Initialization_Method_L3']:
                                                                                 
                                                                                 ModelInfoMade['Nerouns']=[i1,i2,i3]
                                                                                 ModelInfoMade['Layers']=[j]
                                                                                 ModelInfoMade['Dropout_Value']=[ll1,ll2,ll3]
                                                                                 ModelInfoMade['Reguralization']=[m1,m2,m3]
                                                                                 
                                                                                 ModelInfoMade['kernel_constraint']=[n1,n2,n3]
                                                                                 ModelInfoMade['Activation_Method']=[o]
                                                                                 ModelInfoMade['Epochs']=[p]
                                                                                 ModelInfoMade['Batches']=[q]
                                                                                 ModelInfoMade['optimizer']=[r]
                                                                                 ModelInfoMade['W_Initialization_Method']=[s1]
                                                                                 
                                                                                 TotalModelInfo.append(ModelInfoMade.copy())
                                                                                 x=x+1
										 g=g+1
                                                        		         print(x)
                                                        		         print(g)
										 if g==1000000:
#                                                                 #####################################################Open the file and load the model
										    TotalModelInfo=random.sample(TotalModelInfo,k=int(SampleQuantity))
     										    EnergyParamF = open('EnergyParam.obj', 'rb') 
										    ModelInfoLoaded = pickle.load(EnergyParamF)
										    ModelInfoLoaded.append(TotalModelInfo)
							 			    EnergyParamF = open('EnergyParam.obj', 'wb') 
							 			    pickle.dump(ModelInfoLoaded, EnergyParamF)
										    g=0
								 		    TotalModelInfo =[]
							  
                                                    
                                        
                
                    
                                   
                                        
                            
            
        
            
    
    
EnergyParamF = open('EnergyParam.obj', 'rb')
ModelInfoLoaded = pickle.load(EnergyParamF) 
ModelInfoLoaded.append(TotalModelInfo)
shuffle(TotalModelInfo)
EnergyParamF = open('EnergyParam.obj', 'wb')
pickle.dump(ModelInfoLoaded, EnergyParamF)
