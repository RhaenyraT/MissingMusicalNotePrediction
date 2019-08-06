import keras
from sklearn.metrics import roc_auc_score
import RocMatt

class roc_callback(keras.callbacks.Callback):
    def __init__(self,validation_data):

        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        #np.savetxt("D:/EC2/Inputs/ValTarget.csv", self.y_val, delimiter=",") 
        
    def on_train_begin(self, logs={}):
        self.AucScore = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        Roc_Targ,Roc_Val=RocMatt.SortROCMatt(self.y_val,y_pred_val)
        #np.savetxt("D:/EC2/Inputs/ValPred.csv", y_pred_val, delimiter=",") 
        roc_val = sklearn.metrics.roc_auc_score(Roc_Targ, Roc_Val)
        print(roc_val)
        self.AucScore.append(roc_val)
        return