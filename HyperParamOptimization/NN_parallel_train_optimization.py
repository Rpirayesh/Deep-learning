# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:49 2020

@author: rpira
@author: JorgeDiaz
"""

from OptimizedFunction import compile_model
import pickle
import numpy as np
from mpi4py import MPI
import time

#generate random integer values
#from random import randrange

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

total_configurations = 10
per_rank = total_configurations//size
filehandler = open('filename_pi.obj', 'rb')
ModelInfo = pickle.load(filehandler)

# print run info
if rank == 0:
    print('-'*30)
    print("Number of ranks:", size)
    print("Total number of configurations: ", total_configurations)
    print("Configurations per rank: ", per_rank)
    print('-'*30)

mape_optimal = np.zeros(2)
th_quantity = 10
th_accuracy = 1

comm.Barrier()
start_time = time.time()

count = 0.0
mape_temp = 100.0
for conf in range(1 + rank*per_rank, 1 + (rank+1)*per_rank):
    count = count + 1.0
    saved_model, mape = compile_model(ModelInfo[conf])
    #mape = randrange(100)
    print("I am rank", rank, "running conf", conf, ". MAPE =", mape)
    if mape_temp - mape > th_accuracy:
        count = 0.0
    if mape < mape_temp:
        mape_temp = mape
        optimal_model_temp = saved_model
    if count > th_quantity:
        break

mape_optimal[0] = mape_temp
mape_optimal[1] = rank

comm.Barrier()

if rank == 0:
    # Process remaining configurations
    count = 0.0
    for conf in range(1 + (size)*per_rank, total_configurations+1):
        saved_model, mape  =compile_model(ModelInfo[conf])
        #mape = randrange(100)
        print("I am rank", rank, "running conf", conf, ". MAPE =", mape)
        if mape_temp - mape > th_accuracy:
            count = 0.0
        if mape < mape_temp:
            mape_temp = mape
            optimal_model_temp = saved_model
        if count > th_quantity:
            break
    mape_optimal[0] = mape_temp

mape_final = np.zeros(2)

comm.Barrier()
# Find the minimum MAPE across all ranks
mape_final = comm.reduce(mape_optimal, op=MPI.MINLOC, root=0)

comm.Barrier()
mape_final = comm.bcast(mape_final, root=0)

comm.Barrier()
if rank == mape_final[1]:
    optimal_model_final = optimal_model_temp
    print("I am rank:", mape_final[1], "and I found the optimal model", optimal_model_final)
    #print("I am rank:", mape_final[1], "and I found the optimal model")

comm.Barrier()
if rank == 0: 
    print("The optimal mape is: ", mape_final[0], "found by rank: ", mape_final[1])

    stop_time = time.time()
    total_time = int((stop_time-start_time)*1000)
    print('-'*30)
    print("Total execution time: ", total_time, "ms")
    print('-'*30)
