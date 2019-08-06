import numpy as np
def SortROCMatt(Targ,Val):
    Ind       =  np.where(~Targ.any(axis=0))[0]
    #Matt_Targ =  Targ[:][Ind]
    Roc_Targ  =  np.delete(Targ, (Ind), axis=1)
    #Matt_Val  =  Val[:][Ind]
    Roc_Val   =  np.delete(Val, (Ind), axis=1)   
    # np.sum(Roc_Targ, axis=1)
    # np.sum(Matt_Targ, axis=1) 
    # np.savetxt("D:/EC2/Inputs/Roc_Targ.csv", Roc_Targ, delimiter=",") 
    # np.savetxt("D:/EC2/Inputs/Matt_Targ.csv", Matt_Targ, delimiter=",") 
    return  Roc_Targ,Roc_Val