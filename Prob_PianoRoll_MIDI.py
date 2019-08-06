import numpy as np
import scipy 
from scipy import sparse
from sklearn.metrics import roc_curve

print("Changing raw probabilities to values")
PredictedRawProbabilities=np.genfromtxt("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/3L_best_0_PredictedRawProbabilities.csv",delimiter=",")
Prob_target=np.genfromtxt("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/Target.csv",delimiter=",")

Predicted_Pianoroll= np.zeros((PredictedRawProbabilities.shape[0], PredictedRawProbabilities.shape[1]))
testy=Prob_target.flatten()
pred=PredictedRawProbabilities.flatten()
fpr, tpr, thresholds = roc_curve(testy, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

for l in range(0,PredictedRawProbabilities.shape[0]):
     #p = np.argmax(PredictedRawProbabilities[l,:])
     #if p>optimal_threshold:
     #     Predicted_Pianoroll[l,p]=1
     temp=PredictedRawProbabilities[l,:]>0.25
     Predicted_Pianoroll[l,:]=temp.astype(int)
Pianoroll=sparse.csr_matrix(Predicted_Pianoroll)
scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/PianoRolls/Depth/3/3L_Best_multi_MIDI_thp25.npz", Pianoroll, compressed=True)

#***********************************************************************************************************************
#                                                    SWARA PARINAMA  
# *********************************************************************************************************************** 
#                        Maintain   Varna(syllable)        Swara(accent)      Maatra(duration)     
#                                   Balam(time-duration)   Sama(even tone )   Santana(continuity)
#***********************************************************************************************************************

#  import MIDO music library

from mido import MidiFile, MidiTrack, Message
from mido import MetaMessage
import numpy as np
import mido
import csv
import glob
import time
import scipy
from scipy import sparse
np.set_printoptions(threshold=np.nan)

#***************************************************************************************************************************************************************
#       MIDI TO MATRICE ! 
#        GET PIANO ROLL
#***************************************************************************************************************************************************************

#  1)   Decide Sampling
time = float(0)
prev = float(0)

#***************************************************************************************************************************************************************
#        NOW REVERSE ENGINEERING!!!! 
#  GET BACK Note_Start_End FROM PIANO ROLL
#***************************************************************************************************************************************************************
#PianoRoll_sparse =scipy.sparse.load_npz("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Results/p5Sequence_Predictionp1/Simple_500Epoch_1Layer_200Neurons/Mod_PredictedPianoroll.npz") 
#PianoRoll_sparse=scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/PianoRolls/Depth/3/3L_Best_PredictedRoll.npz")
#piano_roll=np.genfromtxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Results/Context/p1sec_Simple_250Epoch_1Layer_200Neurons/PianorollTestTarget.csv",delimiter=",") 
piano_roll = Predicted_Pianoroll


# 1) Find ON indices, transpose and sort by Note

indices=np.where(piano_roll>0)
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/indices_Big.csv", indices, delimiter=",") 
indices_trans=np.transpose(indices)
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/indices_trans_Big.csv", indices_trans, delimiter=",") 
indices_sort=indices_trans[indices_trans[:,1].argsort()]
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/indices_sort_Big.csv", indices_sort, delimiter=",") 
NotesOn=np.unique(indices_sort[:,1])
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/NotesOn.csv", NotesOn, delimiter=",") 

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
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/WhichRow_Note_Big.csv", Main, delimiter=",") 
print(444)
#***********************************************************************************************************************
# 3)  Find Start End and Sequences within ON notes
lowNote=22
Note_Start_End=np.zeros((0,3))
note_Start_End=np.zeros((1,3))
indices_trans=Main

i=0
while i<indices_trans.shape[0]:
    #print(i)
    k=0
    if len(indices_trans[i:])==1:    #IF THIS IS THE LAST LINE
        note_Start_End[0][0]=indices_trans[i][1]+lowNote         
        note_Start_End[0][1]= indices_trans[i][0]
        note_Start_End[0][2]=1+indices_trans[i][0]
        #print(indices_trans[i][1]+lowNote,indices_trans[i][0],0)  
        break
#IF NOTE IS CHANGING AND IT IS NOT LAST LINE
    if indices_trans[i][1]!=indices_trans[i+1][1] and len(indices_trans[i+1])!=1:
        note_Start_End[0][0]=indices_trans[i][1]+lowNote         
        note_Start_End[0][1]= indices_trans[i][0]
        note_Start_End[0][2]=1+indices_trans[i][0]
        #print(indices_trans[i][1]+lowNote,indices_trans[i][0],0)  
#IF NEXT NOTE IS SAME BUT TIME IS NOT CONTINUOUS, MEANING NOTE OCCURS SOMEWHERE ELSE IN SONG
    if indices_trans[i][1]==indices_trans[i+1][1] and (indices_trans[i+1][0]-indices_trans[i][0]!=1):
        note_Start_End[0][0]=indices_trans[i][1]+lowNote         
        note_Start_End[0][1]= indices_trans[i][0]
        note_Start_End[0][2]=1+indices_trans[i][0]
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

#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/Reversed_Note_Start_End_Big.csv", Note_Start_End, delimiter=",") 
Note_Start_End_Sorted=Note_Start_End[Note_Start_End[:,1].argsort()]
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/Reversed_Note_Start_End_Sorted_Big.csv", Note_Start_End_Sorted, delimiter=",") 

# 4)  [ Note, Start time, 1] 
Start_times_temp=np.column_stack((Note_Start_End_Sorted[:,0],Note_Start_End_Sorted[:,1]))
Start_times=np.column_stack((Start_times_temp,(np.ones((len(Start_times_temp),1)))))
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/Start_times_Big.csv", Start_times, delimiter=",") 

# 5)  [ Note, End time, 0] 
End_times_temp=np.column_stack((Note_Start_End_Sorted[:,0],Note_Start_End_Sorted[:,2]))
End_times=np.column_stack((End_times_temp,(np.zeros((len(Start_times_temp),1)))))
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/End_times_Big.csv", End_times, delimiter=",") 

# 6)  TotalStack = [ Note, Start time ]  vertical stacked with [ Note, End time] 
TotalStack=np.vstack((Start_times,End_times))
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/TotalStack_Big.csv", TotalStack, delimiter=",") 

# 7)  TotalStack Sorted by the TIME Axis 
TotalStack_sorted=TotalStack[TotalStack[:,1].argsort()]
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/TotalStack_sorted.csv", TotalStack_sorted, delimiter=",") 
print(666)
#***********************************************************************************************************************
# 8)        MAKE MIDI FILE 

#IF SAMPLING IS 0.01 CHANGE TEMPO TO 600000 AND MULTIPY LINE 151 AND 153 BY 8.

file = open('C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/PianoRolls/Depth/3/3L_Best_multi_MIDI_thp25.csv','w') 
file.write('0, 0, Header, 1, 2, 480\n')
file.write('1, 0, Start_track\n')
file.write('1, 0, Tempo, 600000\n')
file.write('1, 4000000, End_track\n')
file.write('2, 0, Start_track\n')
for keys in range(0,len(TotalStack_sorted)):
    if TotalStack_sorted[keys][2]==1:
        file.write('2, %d, Note_on_c, 0, %d,57\n' % (TotalStack_sorted[keys][1]*80,TotalStack_sorted[keys][0]))
    if TotalStack_sorted[keys][2]==0:
        file.write('2, %d, Note_on_c, 0, %d,0\n' % (TotalStack_sorted[keys][1]*80,TotalStack_sorted[keys][0]))
file.write('2,4000000, End_track\n')
file.write('0, 0, End_of_file')
file.close()
print(777)
#***********************************************************************************************************************
#***********************************************************************************************************************