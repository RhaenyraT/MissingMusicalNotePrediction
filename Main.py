import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras import optimizers
from keras.optimizers import SGD
import glob
import csv
import time
import numpy as np
import scipy 
from scipy import sparse
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

#***************************************************************************************************************************************************************
#INITIALIZATION

NEpochs                =400
Timesteps              = 10
shift                  = 5
#learning_rate          = 0.1
#decay_rate             = learning_rate / NEpochs
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
Target=Target_unshaped.toarray()
print(Target.shape)

Input=Input_unshapedD.reshape(Number_of_Batches_Train,Timesteps,Input_unshapedD.shape[1])  
print(Input.shape)

#***************************************************************************************************************************************************************

# BUILD MODEL & TRAIN

print("Making LSTM model")

model = Sequential()
#model.add(Bidirectional(LSTM(Neurons[0],input_shape=(Input.shape[1],Input.shape[2])),stateful=True)) 
model.add(LSTM(100,input_shape=(Input.shape[1],Input.shape[2]))) #return_sequences=True,
model.add(Dropout(0.3))
#model.add(LSTM(50, return_sequences=True))
#model.add(Dropout(0.3))
#model.add(LSTM(50))
#model.add(Dense(256))
#model.add(Dropout(0.3))
model.add(Dense(Input.shape[2]))
model.add(Activation('sigmoid'))
#momentum = 0.8
#sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['binary_accuracy'])

history=model.fit(Input, Target, batch_size=23, epochs=NEpochs, validation_split=0.25,verbose=1,)
print(history.history.keys())
print("Fitting is over")


Acc                  =  history.history['binary_accuracy']
Loss                 =  history.history['loss']
ValLoss              =  history.history['val_loss']
ValAcc               =  history.history['val_binary_accuracy']
np.savetxt("D:/EC2/SavedModel/Batchsize23/Acc.csv", Acc, delimiter=",") 
np.savetxt("D:/EC2/SavedModel/Batchsize23/Loss.csv", Loss, delimiter=",") 
np.savetxt("D:/EC2/SavedModel/Batchsize23/ValAcc.csv", ValAcc, delimiter=",") 
np.savetxt("D:/EC2/SavedModel/Batchsize23/ValLoss.csv", ValLoss, delimiter=",") 


# serialize model to JSON
model_json = model.to_json()
with open("D:/EC2/SavedModel/Batchsize23/model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")
# serialize weights to HDF5
model.save_weights("D:/EC2/SavedModel/Batchsize23/model.h5")
print("Saved weights to disk")

#***************************************************************************************************************************************************************

# PREDICTION 

print("Importing and reshaping Test Input")
TestInput_unshaped = scipy.sparse.load_npz("D:/EC2/Inputs/TargetInputOutput/Test_10_s5_3.npz")
TestInput_Dense=TestInput_unshaped.toarray()
print(TestInput_Dense.shape)
#TestInput=TestInput_Dense.reshape(Number_of_Batches_Test,Timesteps,TestInput_Dense.shape[1])
TestInput=TestInput_Dense.reshape(Number_of_Batches_Test,Timesteps,TestInput_Dense.shape[1])

print(TestInput.shape)

# print("Starting to Predict Now")
# x=TestInput[140:160][:][:]
# print(x.shape)
# Predicted_Pianoroll= np.zeros((Timesteps,TestInput_Dense.shape[1]))
# prediction=np.array([]).reshape(0,TestInput_Dense.shape[1])
# for i in range(300):
#     preds = model.predict(x)
#     #print (preds)
#     y = np.where(preds > 0.4)
#     Predicted_Pianoroll[y]=1
#     prediction=np.vstack((prediction,Predicted_Pianoroll))
#     xnew=Predicted_Pianoroll
#     x=xnew.reshape(1,Timesteps,TestInput_Dense.shape[1])
    #x=xnew.reshape(40,1,TestInput_Dense.shape[1])

PredictedRawProbabilities = model.predict(TestInput)
#print(prediction.shape)
print(PredictedRawProbabilities.shape)
 
np.savetxt("D:/EC2/SavedModel/Batchsize23/PredictedRawProbabilities.csv", PredictedRawProbabilities, delimiter=",") 
#np.savetxt("D:/EC2/SavedModel/PredictedTarget.csv", prediction, delimiter=",") 

print("DONE ! Now save the files and STOP THE INSTANCE POORIMAA !!")
print("SAVE  !!")
# #***************************************************************************************************************************************************************
