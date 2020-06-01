# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:49 2020

@author: rpira
"""

from OptimizedFunction import compile_model
import pickle
filehandler = open('filename_pi.obj', 'rb') 

ModelInfo = pickle.load(filehandler)

saved_model, mape  =compile_model(ModelInfo[110])
#randomizw(ModelInfo)
##loss, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
#hyperparm=100,000
#nodesnumber=1000
#h=4
#m=np.shape(ModelInfo)[0]/h
#for u in range(m):
#    for z i range(h):
#        
#        saved_model, mape(z)  =compile_model(ModelInfo[110])
#        minimm(mape(z))
#    treasure
#        
        
 
