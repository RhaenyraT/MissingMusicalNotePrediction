from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import glob
#from mido import MidiFile, MidiTrack, Message
#from mido import MetaMessage
import csv
import time
import numpy as np
np.set_printoptions(threshold=np.nan)


PianoRoll = np.genfromtxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/PianoRolls/piano_roll_set9.csv",delimiter=",")
#PianoRoll=np.array([[0,1,1,0],[0,1,0,0],[0,0,0,0],[1,1,0,1],[0,1,0,0],[0,0,0,1],[0,0,1,1],[1,1,0,1],[0,1,0,1],[0,1,0,1],[1,0,1,1]])

#***************************************************************************************************************************************************************
#      MAKE INPUT AND TARGET SEQUENCES
#  In this case, 5,000 time steps is too long; LSTMs work better with 200-to-400 time steps based on some
#  papers I’ve read. Therefore, we need to split the 5,000 time steps into multiple shorter sub-sequences.      
#***************************************************************************************************************************************************************
X=np.empty((0,PianoRoll.shape[1]), int)
Y=np.empty((0,PianoRoll.shape[1]), int)
Split=round(.7*PianoRoll.shape[0])
Train=PianoRoll[:Split,:]
Test=PianoRoll[Split+1:,:]


# SequenceLength=400
# Number_of_Batches_Train=Train.shape[0]//SequenceLength
# Number_of_Batches_Test= Test.shape[0]//SequenceLength
# Xtrain=Train[:(SequenceLength*Number_of_Batches_Train)][:]
# Ytrain=Train[1:(SequenceLength*Number_of_Batches_Train+1)][:]
# Xtest=Test[:(SequenceLength*Number_of_Batches_Test)][:]
# Ytest=Test[1:(SequenceLength*Number_of_Batches_Test+1)][:]
# print(Xtrain.shape)
# print(Ytrain.shape)
# Input = Xtrain.reshape((Number_of_Batches_Train, SequenceLength, Train.shape[1]))
# Target = Ytrain.reshape((Number_of_Batches_Train, SequenceLength, Train.shape[1]))
# TestInput = Xtest.reshape((Number_of_Batches_Test, SequenceLength, Test.shape[1]))
# TestTarget = Ytest.reshape((Number_of_Batches_Test, SequenceLength, Test.shape[1]))


Xtrain=Train[:Train.shape[0]-1][:]
Ytrain=Train[1:Train.shape[0]+1][:]    
Xtest=Test[:Test.shape[0]-1][:]
Ytest=Test[1:Test.shape[0]+1][:]    
Input = Xtrain.reshape((1, Xtrain.shape[0], Xtrain.shape[1]))
Target = Ytrain.reshape((1, Ytrain.shape[0], Ytrain.shape[1]))
TestInput = Xtest.reshape((1, Xtest.shape[0], Xtest.shape[1]))
TestTarget = Ytest.reshape((1, Ytest.shape[0], Ytest.shape[1]))




#***************************************************************************************************************************************************************
#       BUILD MODEL !  
#***************************************************************************************************************************************************************

model = Sequential()
model.add(LSTM(500,return_sequences=True, input_shape=(Xtrain.shape[0], Xtrain.shape[1])))
model.add(Dropout(0.2))
# model.add(LSTM(500, return_sequences=True))
# model.add(Dropout(0.2))
model.add(Dense(PianoRoll.shape[1]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(Input, Target, 1, epochs=500)


predicted = model.predict(TestInput)
Prediction_reshaped=predicted.reshape(predicted.shape[1],67)

Prediction_reshaped_01= np.zeros((Prediction_reshaped.shape[0], Prediction_reshaped.shape[1]))
for l in range(0,Prediction_reshaped.shape[0]):
     p = np.argmax(Prediction_reshaped[l,:])
     Prediction_reshaped_01[l,p]=1

np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/PianoRolls/FinalPrediction.csv", Prediction_reshaped_01, delimiter=",") 
np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/PianoRolls/PredictionProbabilities.csv", Prediction_reshaped, delimiter=",") 
print(11)         


lowNote=22


# 1) Find ON indices, transpose and sort by Note

indices=np.where(Prediction_reshaped_01>0)
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/indices_Big.csv", indices, delimiter=",") 
indices_trans=np.transpose(indices)
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/indices_trans_Big.csv", indices_trans, delimiter=",") 
indices_sort=indices_trans[indices_trans[:,1].argsort()]
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/indices_sort_Big.csv", indices_sort, delimiter=",") 
NotesOn=np.unique(indices_sort[:,1])
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/indices_sort_Big.csv", NotesOn, delimiter=",") 

print(333)
#***********************************************************************************************************************
# 2)  Sort within notes sort in Ascending

Main=np.empty((0,2), int)
for noteNum in range(0,len(NotesOn)):
    temp=np.empty((0,1), int)
    for indexNum in range(0,len(indices_sort)):
        if indices_sort[indexNum][1]==NotesOn[noteNum]:
            temp=np.vstack((temp,indices_sort[indexNum][0]))
    Temp=sorted(temp)
    main=np.column_stack((Temp,(NotesOn[noteNum]*np.ones((len(Temp),1)))))
    Main=np.vstack((Main,main))
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/WhichRow_Note_Big.csv", Main, delimiter=",") 
print(444)
#***********************************************************************************************************************
# 3)  Find Start End and Sequences within ON notes

Note_Start_End=np.zeros((1,3))
note_Start_End=np.zeros((1,3))
indices_trans=Main

i=0
while i<indices_trans.shape[0]:
    #print(i)
    k=0
    if len(indices_trans[i:])==1:  
        note_Start_End[0][0]=indices_trans[i][1]+lowNote     
        note_Start_End[0][1]= indices_trans[i][0]
        note_Start_End[0][2]=1
        #print(indices_trans[i][1]+lowNote,indices_trans[i][0],0)  
        break

    if indices_trans[i][1]!=indices_trans[i+1][1] and len(indices_trans[i+1])!=1:
        note_Start_End[0][0]=indices_trans[i][1]+lowNote     
        note_Start_End[0][1]= indices_trans[i][0]
        note_Start_End[0][2]=1
        #print(indices_trans[i][1]+lowNote,indices_trans[i][0],0)  

    if indices_trans[i][1]==indices_trans[i+1][1] and (indices_trans[i+1][0]-indices_trans[i][0]!=1):
        note_Start_End[0][0]=indices_trans[i][1]+lowNote     
        note_Start_End[0][1]= indices_trans[i][0]
        note_Start_End[0][2]=1
        #print(indices_sort[i][1]+lowNote,indices_sort[i][0],0)
        
    if indices_trans[i][1]==indices_trans[i+1][1] and (indices_trans[i+1][0]-indices_trans[i][0]==1):
        for k in range(0,len(indices_trans[i:])):
            
            if len(indices_trans[i+k:])==1:
                note_Start_End[0][0]=indices_trans[i][1]+lowNote     
                note_Start_End[0][1]= indices_trans[i][0]
                note_Start_End[0][2]=k+1+indices_trans[i][0]                
                #print(indices_trans[i][1]+lowNote,indices_trans[i][0],k+1)
                break

            if not (indices_trans[i+k][1]==indices_trans[i+1+k][1] and (indices_trans[i+1+k][0]-indices_trans[i+k][0]==1)):
                note_Start_End[0][0]=indices_trans[i][1]+lowNote     
                note_Start_End[0][1]= indices_trans[i][0]
                note_Start_End[0][2]=k+1+indices_trans[i][0]
                #print(indices_sort[i][1]+lowNote,indices_sort[i][0],k+1)
                break
    
    Note_Start_End=np.vstack((Note_Start_End,note_Start_End))
    i += 1+k
print(555)
#***********************************************************************************************************************

#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/Reversed_Note_Start_End_Big.csv", Note_Start_End, delimiter=",") 
Note_Start_End_Sorted=Note_Start_End[Note_Start_End[:,1].argsort()]
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/Reversed_Note_Start_End_Sorted_Big.csv", Note_Start_End_Sorted, delimiter=",") 

# 4)  [ Note, Start time, 1] 
Start_times_temp=np.column_stack((Note_Start_End_Sorted[:,0],Note_Start_End_Sorted[:,1]))
Start_times=np.column_stack((Start_times_temp,(np.ones((len(Start_times_temp),1)))))
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/Start_times_Big.csv", Start_times, delimiter=",") 

# 5)  [ Note, End time, 0] 
End_times_temp=np.column_stack((Note_Start_End_Sorted[:,0],Note_Start_End_Sorted[:,2]))
End_times=np.column_stack((End_times_temp,(np.zeros((len(Start_times_temp),1)))))
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/End_times_Big.csv", End_times, delimiter=",") 

# 6)  TotalStack = [ Note, Start time ]  vertical stacked with [ Note, End time] 
TotalStack=np.vstack((Start_times,End_times))
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/TotalStack_Big.csv", TotalStack, delimiter=",") 

# 7)  TotalStack Sorted by the TIME Axis 
TotalStack_sorted=TotalStack[TotalStack[:,1].argsort()]
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/TotalStack_sorted_Big.csv", TotalStack_sorted, delimiter=",") 
print(666)
#***********************************************************************************************************************
# 8)        MAKE MIDI FILE 

file = open('C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/PianoRolls/MidiFile_set9_1l_500n_500e.csv','w') 
file.write('0, 0, Header, 1, 2, 480\n')
file.write('1, 0, Start_track\n')
file.write('1, 0, Tempo, 600000\n')
file.write('1, 1000000, End_track\n')
file.write('2, 0, Start_track\n')
for keys in range(0,len(TotalStack_sorted)):
    if TotalStack_sorted[keys][2]==1:
        file.write('2, %d, Note_on_c, 0, %d,57\n' % (TotalStack_sorted[keys][1]*8,TotalStack_sorted[keys][0]))
    if TotalStack_sorted[keys][2]==0:
        file.write('2, %d, Note_on_c, 0, %d,0\n' % (TotalStack_sorted[keys][1]*8,TotalStack_sorted[keys][0]))
file.write('2,1000000, End_track\n')
file.write('0, 0, End_of_file')
file.close()
print(777)
#***********************************************************************************************************************
#**************************************************************************************************************