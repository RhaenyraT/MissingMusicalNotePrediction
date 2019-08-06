import SupportFiles_bach
import csv
import time
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
shift                  =  5
timesteps_in_one_Batch =  10   # 1 seconds of music training samples used per gradient-update predict the next  0.5 seconds


#***************************************************************************************************************************************************************
#INITIALIZATION
print("Opening BIG npz")
PianoRoll_sparse =scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/bach.npz")
PianoRoll=PianoRoll_sparse.tocsr()
print(PianoRoll.shape)
#***************************************************************************************************************************************************************
#MAKE INPUT AND TARGET SEQUENCES

print("Splitting into Train & Test is over")
SupportFiles_bach.MakeInputTarget(timesteps_in_one_Batch,PianoRoll,shift)
