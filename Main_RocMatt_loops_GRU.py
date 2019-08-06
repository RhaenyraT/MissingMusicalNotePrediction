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
from keras.layers.recurrent import SimpleRNN
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
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
#***************************************************************************************************************************************************************
#INITIALIZATION

NEpochs                = 15
Timesteps              = 10
shift                  = 5

#***************************************************************************************************************************************************************

print("Importing Input Target files")
Train      = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/Train.npz")
Test       = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/Test.npz")
Val        = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/Val.npz")

print(Train.shape)
print(Val.shape)
print(Test.shape)

TrainArr       = Train.toarray()
ColumnSumTrainArr  = TrainArr.sum(axis=0)
#np.savetxt("D:/EC2/SavedModel/note/ColumnSumTrainArr.csv", ColumnSumTrainArr, delimiter=",") 
TestArr       = Test.toarray()
ColumnSumTestArr  = TestArr.sum(axis=0)
#np.savetxt("D:/EC2/SavedModel/note/ColumnSumTestArr.csv", ColumnSumTestArr, delimiter=",") 
ValArr       = Val.toarray()
ColumnSumValArr  = ValArr.sum(axis=0)
#np.savetxt("D:/EC2/SavedModel/note/ColumnSumValArr.csv", ColumnSumValArr, delimiter=",") 


Number_of_Batches_Train  =  Train.shape[0]-Timesteps-shift #86902 #
Number_of_Batches_Test   =  Test.shape[0]-Timesteps-shift #28950 #
Number_of_Batches_Val    =  Val.shape[0]-Timesteps-shift 

Input_unshaped    = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/TrainX_10_s5.npz")
Input_unshapedD   = Input_unshaped.toarray()
print(Input_unshapedD.shape)
X_train=Input_unshapedD.reshape(Number_of_Batches_Train,Timesteps,Input_unshapedD.shape[1])  #,
print(X_train.shape)
Target_unshaped   = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/TrainY_10_s5.npz")
Y_train           = Target_unshaped.toarray()
print(Y_train.shape)

ValX_unshaped   = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/ValX_10_s5.npz")
ValX_unshapedD  = ValX_unshaped.toarray()
print(ValX_unshapedD.shape)
X_Validate=ValX_unshapedD.reshape(Number_of_Batches_Val,Timesteps,ValX_unshapedD.shape[1])  #,
print(X_Validate.shape)
ValY_unshaped   = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/ValY_10_s5.npz")
Y_Validate  = ValY_unshaped.toarray()
print(Y_Validate.shape)


#***************************************************************************************************************************************************************

print("Importing and reshaping Test Input")
TestInput_unshaped = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/TestX_10_s5.npz")
TestInput_Dense    = TestInput_unshaped.toarray()
print(TestInput_Dense.shape)
TestInput          = TestInput_Dense.reshape(Number_of_Batches_Test,Timesteps,TestInput_Dense.shape[1])



#***********************************************************************************************************************************************************
repetitions=5
neurons=[200]
dropout=[0]
regularization=[.0001]
loss= np.empty([NEpochs, repetitions])
valLoss= np.empty([NEpochs, repetitions])
val_AUC= np.empty([0, X_train.shape[2]])
# BUILD MODEL & TRAIN
for yy in range(0,len(regularization)):
    for uu in range(0,repetitions):
        print("Making LSTM model")
        
        model = Sequential()
        #model.add(Bidirectional(LSTM(Neurons[0],input_shape=(Input.shape[1],Input.shape[2])),stateful=True)) 
        #.add(Bidirectional(LSTM(neurons[0],input_shape=(X_train.shape[1],X_train.shape[2])))) #return_sequences=True,
        model.add(GRU(neurons[0],input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Dropout(dropout[0]))
        #model.add(LSTM(50)) #, return_sequences=True
        #model.add(Dropout(0.3))
        # model.add(LSTM(256))
        # model.add(Dense(256))
        # model.add(Dropout(0.3))
        model.add(Dense(X_train.shape[2],kernel_regularizer=regularizers.l2(regularization[yy])))
        model.add(Activation('sigmoid'))
        
        class roc_callback(keras.callbacks.Callback):
            def __init__(self,validation_data):
        
                self.x_val = validation_data[0]
                self.y_val    = validation_data[1]
                
            def on_train_begin(self, logs={}):
                return
        
            def on_train_end(self, logs={}):
                return
        
            def on_epoch_begin(self, epoch, logs={}):
                return
        
            def on_epoch_end(self, epoch, logs={}):
         
                y_pred_val           =  self.model.predict(self.x_val)
                #Roc_Targ,Roc_Val    =  RocMatt.SortROCMatt(self.y_val,y_pred_val)
                Roc_Targ             =  self.y_val
                Roc_valcol           = np.empty(shape=[0, 1])
                for h in range(0,Roc_Targ.shape[1]):
                    if np.sum(Roc_Targ[:,h]==0)==Roc_Targ.shape[0]:
                        roc_valcol=0
                    else:
                        roc_valcol       =  sklearn.metrics.roc_auc_score(Roc_Targ[:,h], y_pred_val[:,h])
                    Roc_valcol       =  np.vstack((Roc_valcol,roc_valcol))
        
                #Roc_TargF            =  Roc_Targ.flatten()
                #Roc_ValF             =  y_pred_val.flatten()
        
                #roc_val              =  sklearn.metrics.roc_auc_score(Roc_TargF, Roc_ValF)
        
                # fpr, tpr, thresholds =  sklearn.metrics.roc_curve(Roc_TargF,Roc_ValF)
                # optimal_idx          =  np.argmax(tpr - fpr)
                # optimal_threshold    =  thresholds[optimal_idx]
                # Roc_ValM             =  np.zeros((Roc_ValF.shape[0], ))
                # Indice               =  np.where(Roc_ValF > optimal_threshold)
                # Roc_ValM[Indice]     =  1
                # mattval              =  sklearn.metrics.matthews_corrcoef(Roc_TargF, Roc_ValM)
        
                # logs['val_matthews_correlation'] = mattval 
               # logs['val_AUC']                  = roc_val
                logs['val_AUCcol']               = Roc_valcol
                return
            
        model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['binary_accuracy'])
        history=model.fit(X_train, Y_train, validation_data=(X_Validate, Y_Validate),  batch_size=205, epochs=NEpochs, verbose=1,callbacks=[roc_callback(validation_data=(X_Validate, Y_Validate))])
        #history=model.fit(X_train, Y_train, validation_data=(X_Validate, Y_Validate),  batch_size=413, epochs=NEpochs, verbose=1)

        print(history.history.keys())
        print("Fitting is over")
        
        Loss         =  history.history['loss']
        ValLoss       =  history.history['val_loss']
        #ValCorr               =  history.history['val_matthews_correlation']
        Val_AUC      =  history.history['val_AUCcol']
        
        loss[:,uu]=Loss
        valLoss[:,uu]=ValLoss
        for gg in range(0,NEpochs):
            temp=np.asarray(Val_AUC[gg])
            val_AUC=np.vstack((val_AUC,np.transpose(temp)))
        

    np.savetxt('C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/ValAUCRegularizer'+ str(yy) + '.txt', val_AUC)
    np.savetxt('C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/LossRegularizer'+ str(yy) + '.txt', loss)
    np.savetxt('C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/ValLossRegularizer'+ str(yy) + '.txt', valLoss)
#        np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/Loss_40.csv", loss, delimiter=",") 
#        np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/Val_AUC_40.csv", val_AUC, delimiter=",") 
#        #np.savetxt("D:/EC2/SavedModel/100100/ValCorr.csv", ValCorr, delimiter=",") 
#        np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/ValLoss_40.csv", valLoss, delimiter=",") 
#***************************************************************************************************************************************************************
    
    # PREDICTION 
    
#    print("Importing and reshaping Test Input")
#    TestInput_unshaped = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/Inputs/Test_1_s5_3.npz")
#    TestInput_Dense=TestInput_unshaped.toarray()
#    print(TestInput_Dense.shape)
#    TestInput=TestInput_Dense.reshape(Number_of_Batches_Test,Timesteps,TestInput_Dense.shape[1])
#    
#    print(TestInput.shape)
    
    print("Starting to Predict Now")
    PredictedRawProbabilities = model.predict(TestInput)
    print(PredictedRawProbabilities.shape)

    np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/PredictedRawProbabilities.csv", PredictedRawProbabilities, delimiter=",") 


# serialize model to JSON
#model_json = model.to_json()
#with open("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/model.json", "w") as json_file:
    #json_file.write(model_json)
print("Saved model to disk")
# serialize weights to HDF5
#model.save_weights("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/SavedModels/model.h5")
print("Saved weights to disk")
print("DONE ! Now save the files and STOP THE INSTANCE POORIMAA !!")
print("SAVE  model.json  ,  model.h5  PredictedPianoroll  PredictedRawProbabilities  !!")
# #***************************************************************************************************************************************************************
