import numpy as np
import scipy
from scipy import sparse,vstack

def MakeInputTarget(timesteps_in_one_Batch,Train,shift):
    SequenceLength=timesteps_in_one_Batch
    
    Number_of_Batches_Train=Train.shape[0]-SequenceLength-shift
    Xtrain= np.array([]).reshape(0,Train.shape[1])  
    Ytrain=np.array([]).reshape(0,Train.shape[1])

    for t in range(0,Number_of_Batches_Train):
        
        if t%1000==0:
            print(t)
        seq_in = Train[t:t+timesteps_in_one_Batch,:]
        seq_out = Train[shift+t+timesteps_in_one_Batch,:]
        Xtrain=scipy.sparse.vstack((Xtrain,seq_in))
        Ytrain=scipy.sparse.vstack((Ytrain,seq_out))

    print(Xtrain.shape)
    print(Ytrain.shape)

    Xtrain_D=Xtrain.toarray()
    Input = Xtrain_D.reshape((Number_of_Batches_Train, SequenceLength, Train.shape[1]))
    Target = Ytrain.toarray()
    print(Input.shape)
    print(Target.shape)

    scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/BachX.npz", Xtrain, compressed=True)
    scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/BachY.npz", Ytrain, compressed=True) 
    print("Train data is saved Location")
    return 

