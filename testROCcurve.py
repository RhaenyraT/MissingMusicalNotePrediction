import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import MatthewCoeff
import numpy as np
from sklearn import metrics
import keras.backend as K
def MattCoeff(t,y_true,y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
y = np.array([1, 1, 0, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores)
print(fpr)
print(tpr)
print(thresholds)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y,scores)
j_scores = tpr-fpr
j_ordered = sorted(zip(j_scores,thresholds))
yy= MattCoeff(j_ordered[-1][1],y,scores)
print(yy)