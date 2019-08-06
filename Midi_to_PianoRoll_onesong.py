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
Sampling=.01
time = float(0)
prev = float(0)

files_dir="C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/singlesong/"
train_files = glob.glob("%s*.mid" %(files_dir))

notes = []
Note_StartTime_Velocity=np.zeros((1,3))

#***********************************************************************************************************************
#   2)     FIND  [Note, StartTime, Velocity], Get Highest and Lowest Note

for file_dir in train_files:
    file_path = "%s" %(file_dir)
    mid = MidiFile(file_path)    
    #print(file_path) 
    #print(time)              
    for msg in mid:
        #print(msg)
    	### this time is in seconds, not ticks
        time += msg.time
        print(msg)
        if not msg.is_meta:
			### only interested in piano channel
            if msg.channel == 0:
                if msg.type == 'note_on':
                    #note in vector form to train on
                    note = msg.bytes() 
                    if not note[1]>88:
					# message is in the form of [type, note, velocity]
                        note = note[1:2]
                        note.append(time)  # [ note, total time, velocity]
                        note.append(msg.velocity)
                        note=np.array([note])                    
                        prev = time
                        Note_StartTime_Velocity=np.vstack((Note_StartTime_Velocity,note))    
    print(time)
np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/Note_timeElapsed_Velocity.csv", Note_StartTime_Velocity, delimiter=",")
highNote=int(max(Note_StartTime_Velocity[1:,0]))

#print(highNote)
lowNote=int(min(Note_StartTime_Velocity[1:,0]))
#print(lowNote)
TotalTime=int(Note_StartTime_Velocity[-1][1]//Sampling)

#***********************************************************************************************************************
#   3)    FIND   [ Note , StartTime , Length it was ON ]
print("Starting the Rest")
Note_StartTime_Length = []
for i, message in enumerate(Note_StartTime_Velocity[1:][:]):
    if message[2] != 0: #if note type is 'note_on'
        start_time = int(message[1]/Sampling)
        for event in Note_StartTime_Velocity[i:]: 
            if event[0] == message[0] and event[2] == 0:
                length = int(event[1]/Sampling) - start_time
                break
                
        Note_StartTime_Length.append([int(message[0]), start_time, length])

np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/Note_StartTime_Length.csv", Note_StartTime_Length, delimiter=",") 

print(111)
#***********************************************************************************************************************
#   4)    FIND  PIANO ROLL !!

#piano_roll = np.zeros((TotalTime, 88-22+1))
piano_roll = np.zeros((TotalTime, 88))
for row in Note_StartTime_Length:
    #piano_roll[row[1]:(row[1]+row[2]), row[0]-22] = 1
    piano_roll[row[1]:(row[1]+row[2]), row[0]-1] = 1
Pianoroll=sparse.csr_matrix(piano_roll)
scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/bor_ps1.npz", Pianoroll, compressed=True)

#p.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/Proll_bach_0847_p1.csv", piano_roll, delimiter=",") 

print(222)
#***************************************************************************************************************************************************************


#print(TotalTime)
#piano_roll = np.zeros(( TotalTime, 1), dtype=np.float32)
#PianoRoll  =np.empty((0,1), int)
#i=0
#for row in Note_StartTime_Length: 
#    i=i+1
#    print(i)
#    piano_roll[row[1]:(row[1]+row[2])]=row[0]
#    PianoRoll=np.vstack((PianoRoll,piano_roll))  
#    
#np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/pianoroll_borps1.csv", PianoRoll, delimiter=",")
#print(333)