import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.models import model_from_json
import glob
import csv
import time
import numpy as np
import scipy 
from scipy import sparse
import os
os.path.expanduser("D:/EC2/path")

Timesteps       = 10
print("Importing Input Target files")
PianoRollS      = scipy.sparse.load_npz("D:/EC2/Inputs/LongPianoroll_reallylong.npz")
PianoRoll       = PianoRollS.toarray()
Split=round(0.75*PianoRoll.shape[0])
train=PianoRoll[:Split,:]
test=PianoRoll[Split+1:,:]
Number_of_Batches_Train=train.shape[0]-Timesteps-5
Number_of_Batches_Test =test.shape[0]-Timesteps-5

print("Importing and reshaping Test Input")
TestInput_unshaped = scipy.sparse.load_npz("D:/EC2/Inputs/TestInput_10_s5.npz")
TestInput_Dense=TestInput_unshaped.toarray()
print(TestInput_Dense.shape)
TestInput=TestInput_Dense.reshape(Number_of_Batches_Test,Timesteps,TestInput_Dense.shape[1])
print(TestInput.shape)


    
json_file = open("D:/EC2/SavedModel/model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("D:/EC2/SavedModel/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


print("Starting to Predict Now")
x=TestInput[140:150][:][:]
print(x.shape)
Predicted_Pianoroll= np.zeros((10,TestInput_Dense.shape[1]))
prediction=np.array([]).reshape(0,TestInput_Dense.shape[1])
for i in range(300):
    preds = loaded_model.predict(x)
    print (preds)
    y = np.where(preds > 0.4)
    Predicted_Pianoroll[y]=1
    prediction=np.vstack((prediction,Predicted_Pianoroll))
    xnew=Predicted_Pianoroll
    x=xnew.reshape(1,Timesteps,TestInput_Dense.shape[1])


PredictedRawProbabilities = loaded_model.predict(TestInput)
print(prediction.shape)
print(PredictedRawProbabilities.shape)
 
np.savetxt("D:/EC2/SavedModel/PredictedRawProbabilities.csv", PredictedRawProbabilities, delimiter=",") 
np.savetxt("D:/EC2/SavedModel/PredictedPianoroll.csv", prediction, delimiter=",") 

print("DONE ! Now save the files and STOP THE INSTANCE POORIMAA !!")
print("SAVE  PredictedPianoroll  ,  PredictedRawProbabilities  !!")