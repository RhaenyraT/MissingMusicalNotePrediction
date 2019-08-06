import SupportFiles
import csv
import time
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
shift                  =  5
split                  =  0.90
timesteps_in_one_Batch =  30   # 2 seconds of music training samples used per gradient-update predict the next  0.5 seconds


#***************************************************************************************************************************************************************
#INITIALIZATION
print("Opening BIG npz")
PianoRoll_sparse =scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/Inputs/LongPianoroll_reallylong.npz")
PianoRoll=PianoRoll_sparse.tocsr()
print(PianoRoll.shape)
#***************************************************************************************************************************************************************
#MAKE INPUT AND TARGET SEQUENCES
Train,Test=SupportFiles.Splitdata(split,PianoRoll)
print(Train.shape)
print(Test.shape)
print("Splitting into Train & Test is over")
SupportFiles.MakeInputTarget(timesteps_in_one_Batch,Train,shift)
SupportFiles.MakeTestInputTarget(timesteps_in_one_Batch,Test,shift)
