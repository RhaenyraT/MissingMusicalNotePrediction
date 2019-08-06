import numpy as np
import scipy 
from scipy import sparse

Input_unshaped  = scipy.sparse.load_npz("D:/EC2/Inputs/TargetInputOutput/Training_20_s5_3.npz")
Input_unshapedD = Input_unshaped.toarray()
# ColumnSumTrain  = Input_unshapedD.sum(axis=0)
# np.savetxt("D:/EC2/SavedModel/note/SumTrain.csv", ColumnSumTrain, delimiter=",") 
print(Input_unshapedD.shape)

Target_unshaped   = scipy.sparse.load_npz("D:/EC2/Inputs/TargetInputOutput/TrainingTarget_20_s5_3.npz")
Target_unshapedD  = Target_unshaped.toarray()
# ColumnSumTarget  = Target_unshapedD.sum(axis=0)
# np.savetxt("D:/EC2/SavedModel/note/SumTarget.csv", ColumnSumTarget, delimiter=",") 

Input=Input_unshapedD.reshape(86902,20,Input_unshapedD.shape[1])  #,
print(Input.shape)

tt=range(0,86875,625)
print(tt.shape)

for yy in range(len(tt)):
    x_train=