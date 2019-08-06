import numpy as np
import SupportFiles

PianoRoll = np.genfromtxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/PianoRolls/piano_roll_set9.csv",delimiter=",")

#***************************************************************************************************************************************************************
#MAKE INPUT AND TARGET SEQUENCES
split                  =  0.7
timesteps_in_one_Batch =  1
Train,Test=SupportFiles.Splitdata(split,PianoRoll)
Input,Target,TestInput,TestTarget,Number_of_Batches_Train=SupportFiles.MakeInputTarget(timesteps_in_one_Batch,Train,Test)
