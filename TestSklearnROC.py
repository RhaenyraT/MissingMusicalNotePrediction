import numpy as np
import RocMatt
from sklearn.metrics import roc_auc_score
y_true=np.genfromtxt("D:/EC2/Inputs/ValTarget.csv",delimiter=",")
y_scores=np.genfromtxt("D:/EC2/Inputs/ValPred.csv",delimiter=",")
Matt_Targ,Roc_Targ,Matt_Val,Roc_Val=RocMatt.SortROCMatt(y_true,y_scores)
y=roc_auc_score(Roc_Targ, Roc_Val)
print(y)
#y_true = np.array([0, 0, 0, 0])
#y_scores = np.array([.1, 0.2, 0, 0])
# try:
#     roc_auc_score(y_true, y_scores)
# except ValueError:
#     pass