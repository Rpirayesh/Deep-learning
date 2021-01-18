# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:08:18 2020

@author: rpira
"""


from __future__ import absolute_import, division, print_function, unicode_literals
#group rank and group size and type of output
import pandas as pd
import math
import numpy as np
from sklearn.svm import SVR
####### data for crossfold
dataset= pd.read_csv('DataI300Q7893.csv') # read data set using pandas
#np.random.shuffle(dataset.values)
#dataset.sample(frac=1)
InputData=dataset.copy()
#Split features from labels
InputData.pop('P')
InputData.pop('D')
InputData.pop('Time')
InputData.pop('Energy')
OutputData=InputData.pop('Error')
#Defining the parameters
grid={}

#Define the portion for the data
K_fold=4
pp=len(InputData)
counter=math.floor(pp/K_fold)
i=0
IndEnd=(i+1)*counter
IndBeging=i*counter
test_dataset = InputData[int(IndBeging):int(IndEnd)]
train_dataset = InputData.drop(test_dataset.index)
test_labels = OutputData[int(IndBeging):int(IndEnd)]
train_labels = OutputData.drop(test_labels.index)
### Defining MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def CrossOver(K_fold,InputData,Output,ModelInfo):
    ScoresMAPE=[]
    ScoresMSE=[]
    for CrossCount in range(1,K_fold+1):
        mse,mape =compile_model(CrossCount,K_fold,InputData,Output,ModelInfo)
        ScoresMAPE.append(mape)
        ScoresMSE.append(mse)
    CrossMape=np.mean(ScoresMAPE)
    CrossMSE=np.mean(ScoresMSE)
    return CrossMape,CrossMSE

## Making the GP
#kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.Matern(10.0, (1e-3, 1e3))
regressor = SVR(kernel='linear', C=.01, gamma = .05, epsilon = 0.001)
regressor.fit(train_dataset,train_labels)#5 Predicting a new result
y_pred = regressor.predict(test_dataset)
MSE = ((y_pred-test_labels)**2).mean()
MAPE=mean_absolute_percentage_error(test_labels, y_pred)
print('MAPE=',MAPE)
print('MSE=',MSE)
#print('Std=',std)
#print('StdLen=',len(std))
