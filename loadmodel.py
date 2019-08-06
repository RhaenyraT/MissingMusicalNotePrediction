# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 00:07:35 2019

@author: Apoorva.Radhakrishna
"""
import numpy as np
import sklearn
import math
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import model_from_json
from keras import regularizers
from keras.regularizers import l2
import RocMatt
import glob
import csv
import time
import scipy 
from scipy import sparse
import RocMatt
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.models import model_from_json

# load json and create model
json_file = open('C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/model.h5")
print("Loaded model from disk")

Timesteps=10

print("Importing and reshaping Test Input")
TestInput_unshaped = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/BachX.npz")
TestInput_Dense    = TestInput_unshaped.toarray()
print(TestInput_Dense.shape)
TestInput          = TestInput_Dense.reshape(6056,Timesteps,TestInput_Dense.shape[1])

TestOuput_unshaped = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/BachY.npz")
TestOuput_Dense    = TestOuput_unshaped.toarray()
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
PredictedRawProbabilities = loaded_model.predict(TestInput)

np.savetxt('C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/1L_bach_PredictedRawProbabilities.csv', PredictedRawProbabilities, delimiter=",") 
