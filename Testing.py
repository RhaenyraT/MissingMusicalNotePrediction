import numpy as np
import scipy 
import math
from scipy import sparse
a=np.array([[1, 2], [3, 4],[5,6],[7,8],[9,10],[11,12]])
print(a.shape)
b=a.reshape(3,2,a.shape[1])


SplitValidate=  math.floor(0.75*b.shape[0])
X_train      =  b[:SplitValidate,:][:][:]
X_Validate   =  b[SplitValidate+1:,:][:][:]
