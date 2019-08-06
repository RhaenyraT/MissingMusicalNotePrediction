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
np.set_printoptions(threshold=np.nan)

#***************************************************************************************************************************************************************
#       MIDI TO MATRICE ! 
#        GET PIANO ROLL
#***************************************************************************************************************************************************************

#  1)   Decide Sampling
Sampling=.01
time = float(0)
prev = float(0)

files_dir="C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/"
train_files = glob.glob("%s*.mid" %(files_dir))

notes = []
Note_StartTime_Velocity=np.zeros((1,3))

#***********************************************************************************************************************
#   2)     FIND  [Note, StartTime, Velocity], Get Highest and Lowest Note

for file_dir in train_files:
    file_path = "%s" %(file_dir)
    mid = MidiFile(file_path)    
                  
    for msg in mid:
    	### this time is in seconds, not ticks
        time += msg.time

        if not msg.is_meta:
			### only interested in piano channel
            if msg.channel == 0:
                if msg.type == 'note_on':
                    #note in vector form to train on
                    note = msg.bytes() 
					# message is in the form of [type, note, velocity]
                    note = note[1:2]
                    note.append(time)  # [ note, total time, velocity]
                    note.append(msg.velocity)
                    note=np.array([note])                    
                    prev = time
                    Note_StartTime_Velocity=np.vstack((Note_StartTime_Velocity,note))    
    
#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/Note_timeElapsed_Velocity_Big.csv", Note_StartTime_Velocity, delimiter=",")
highNote=int(max(Note_StartTime_Velocity[1:,0]))
print(time)
print(highNote)
lowNote=int(min(Note_StartTime_Velocity[1:,0]))
print(lowNote)
TotalTime=int(Note_StartTime_Velocity[-1][1]//Sampling)
print(000)
#***********************************************************************************************************************
#   3)    FIND   [ Note , StartTime , Length it was ON ]

Note_StartTime_Length = []
for i, message in enumerate(Note_StartTime_Velocity):
    if message[2] != 0: #if note type is 'note_on'
        start_time = int(message[1]/Sampling)
        for event in Note_StartTime_Velocity[i:]: 
            if event[0] == message[0] and event[2] == 0:
                length = int(event[1]/Sampling) - start_time
                break
                
        Note_StartTime_Length.append([int(message[0]), start_time, length])

#np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/Note_StartTime_Length_Big.csv", Note_StartTime_Length, delimiter=",") 

print(111)
#***********************************************************************************************************************
#   4)    FIND  PIANO ROLL !!

piano_roll = np.zeros((TotalTime, highNote-lowNote+1), dtype=np.float32)
for row in Note_StartTime_Length:
    piano_roll[row[1]:(row[1]+row[2]), row[0]-lowNote] = 1
np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/piano_roll_Big.csv", piano_roll_Big, delimiter=",") 

print(222)
#***************************************************************************************************************************************************************
#        NOW REVERSE ENGINEERING!!!! 
#  GET BACK Note_Start_End FROM PIANO ROLL
#***************************************************************************************************************************************************************

# 1) Find ON indices, transpose and sort by Note

indices=np.where(piano_roll>0)
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

file = open('C:/Users/Poori/Desktop/Parinama/MakePianoRoll/ExcelFiles/MidiFile_Big.txt','w') 
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
#***********************************************************************************************************************