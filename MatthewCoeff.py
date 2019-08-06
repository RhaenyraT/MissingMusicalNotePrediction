import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

a = np.array([[0, 1], [1, 0]])
b = np.array([[0.5, 0.5], [0.3, 0.9]])
roc_val    = sklearn.metrics.roc_auc_score(a, b)
ww=a.flatten()
vv=b.flatten()
REroc_val    = sklearn.metrics.roc_auc_score(ww,vv)
#REcorr       = sklearn.metrics.matthews_corrcoef(ww,vv)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(ww,vv)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print('AUC score for unshaped is %f' % roc_val)
print('AUC score for REshaped is %f' % REroc_val)
print('other j value for REshaped is %f' % optimal_threshold)