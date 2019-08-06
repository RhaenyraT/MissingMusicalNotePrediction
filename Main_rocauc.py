import scipy
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from sklearn.metrics import roc_curve, auc
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.models import model_from_json
import glob
import csv
import time
import numpy as np
import scipy 
from scipy import sparse
import keras.backend as K
#***************************************************************************************************************************************************************
#INITIALIZATION


NLayers                =  1
dropout                =  0.3
Neurons                = np.array([512,512])
NEpochs                = 1
Timesteps              = 10
shift                  = 5
learning_rate          = 0.1
decay_rate             = learning_rate / NEpochs
#***************************************************************************************************************************************************************

print("Importing Input Target files")
PianoRollS      = scipy.sparse.load_npz("D:/EC2/Inputs/Three.npz")
PianoRoll       = PianoRollS.toarray()
Split=round(0.9*PianoRoll.shape[0])
train=PianoRoll[:Split,:]
test=PianoRoll[Split+1:,:]
Number_of_Batches_Train=train.shape[0]-Timesteps-shift
Number_of_Batches_Test =test.shape[0]-Timesteps-shift

Input_unshaped = scipy.sparse.load_npz("D:/EC2/Inputs/TargetInputOutput/Training_10_s5_3.npz")
Input_unshapedD=Input_unshaped.toarray()
print(Input_unshapedD.shape)

Target_unshaped = scipy.sparse.load_npz("D:/EC2/Inputs/TargetInputOutput/TrainingTarget_10_s5_3.npz")
Target_unshapedD=Target_unshaped.toarray()
Input=Input_unshapedD.reshape(Number_of_Batches_Train,Timesteps,Input_unshapedD.shape[1])  #,
print(Input.shape)
#***************************************************************************************************************************************************************
SplitValidate=round(0.79*Input.shape[0])
X_train      =  Input[:SplitValidate,:][:][:]
X_Validate   =  Input[SplitValidate+1:,:][:][:]
Y_train      =  Target_unshapedD[:SplitValidate,:][:]
Y_Validate   =  Target_unshapedD[SplitValidate+1:,:][:]
print(X_train.shape)
print(X_Validate.shape)
print(Y_train.shape)
print(Y_Validate.shape)
#***********************************************************************************************************************************************************
# BUILD MODEL & TRAIN

print("Making LSTM model")

model = Sequential()
#model.add(Bidirectional(LSTM(Neurons[0],input_shape=(Input.shape[1],Input.shape[2])),stateful=True)) 
model.add(LSTM(200,input_shape=(X_train.shape[1],X_train.shape[2]))) #return_sequences=True,
model.add(Dropout(0.3))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(256))
# model.add(Dense(256))
# model.add(Dropout(0.3))
model.add(Dense(X_train.shape[2]))
model.add(Activation('sigmoid'))

class roc_callback(keras.callbacks.Callback):
    def __init__(self,validation_data):

        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = sklearn.metrics.roc_auc_score(self.y_val, y_pred_val)
        print('\roc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['binary_accuracy'])
history=model.fit(X_train, Y_train, validation_data=(X_Validate, Y_Validate),  batch_size=171, epochs=NEpochs, verbose=1,callbacks=[roc_callback(validation_data=(X_Validate, Y_Validate))])
print(history.history.keys())
print("Fitting is over")

#Corr                 =  history.history['matthews_correlation']
Loss                 =  history.history['loss']
ValLoss              =  history.history['val_loss']
#ValCorr               =  history.history['val_matthews_correlation']
#np.savetxt("D:/EC2/SavedModel/testmatthew/Corr.csv", Corr, delimiter=",") 
np.savetxt("D:/EC2/SavedModel/testmatthew/Loss.csv", Loss, delimiter=",") 
#np.savetxt("D:/EC2/SavedModel/testmatthew/ValCorr.csv", ValCorr, delimiter=",") 
np.savetxt("D:/EC2/SavedModel/testmatthew/ValLoss.csv", ValLoss, delimiter=",") 

# serialize model to JSON
model_json = model.to_json()
with open("D:/EC2/SavedModel/testmatthew/model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")
# serialize weights to HDF5
model.save_weights("D:/EC2/SavedModel/testmatthew/model.h5")
print("Saved weights to disk")

#***************************************************************************************************************************************************************

# PREDICTION 

print("Importing and reshaping Test Input")
TestInput_unshaped = scipy.sparse.load_npz("D:/EC2/Inputs/TargetInputOutput/Test_10_s5_3.npz")
TestInput_Dense=TestInput_unshaped.toarray()
print(TestInput_Dense.shape)
TestInput=TestInput_Dense.reshape(Number_of_Batches_Test,Timesteps,TestInput_Dense.shape[1])

print(TestInput.shape)

print("Starting to Predict Now")
PredictedRawProbabilities = model.predict(TestInput)
print(PredictedRawProbabilities.shape)
 
np.savetxt("D:/EC2/SavedModel/testmatthew/PredictedRawProbabilities.csv", PredictedRawProbabilities, delimiter=",") 

print("DONE ! Now save the files and STOP THE INSTANCE POORIMAA !!")
print("SAVE  model.json  ,  model.h5  PredictedPianoroll  PredictedRawProbabilities  !!")
# #***************************************************************************************************************************************************************


#momentum = 0.8
#sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# class roc_callback(Callback):
#     def __init__(self,validation_data):
#         self.x_val = validation_data[0]
#         self.y_val = validation_data[1]


#     def on_train_begin(self, logs={}):
#         return

#     def on_train_end(self, logs={}):
#         return

#     def on_epoch_begin(self, epoch, logs={}):
#         return

#     def on_epoch_end(self, epoch, logs={}):
#         y_pred_val = self.model.predict(self.x_val)
#         roc_val = sklearn.metrics.roc_auc_score(self.y_val, y_pred_val)
#         print('\rroc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
#         return

#     def on_batch_begin(self, batch, logs={}):
#         return

#     def on_batch_end(self, batch, logs={}):
#         return