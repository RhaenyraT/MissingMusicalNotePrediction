import numpy as np
from sklearn.metrics import roc_auc_score
y_true   = np.array([0 ,   1  , 0 ,  0  , 0 , 0 , 1  , 0 , 0 , 0 ,  1 , 0 ,  0])
y_scores = np.array([0.1 ,0.5 , 0 , 0.1 , 0 , 0 , 0 , 0.3 ,0 , 0 , 0 , 0.5, .2])
#try:
YY=roc_auc_score(y_true, y_scores)
#except ValueError:
#    pass
print(YY)