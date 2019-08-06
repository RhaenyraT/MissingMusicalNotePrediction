import numpy as np
import scipy
from scipy import sparse,vstack
def Splitdata(num,PianoRoll):
    Split=round(num*PianoRoll.shape[0])
    Train=PianoRoll[:Split,:]
    Test=PianoRoll[Split+1:,:]
    return Train,Test


def MakeInputTarget(timesteps_in_one_Batch,Train,shift):
    SequenceLength=timesteps_in_one_Batch
    
    Number_of_Batches_Train=Train.shape[0]-SequenceLength-shift
    Xtrain= np.array([]).reshape(0,Train.shape[1])  
    Ytrain=np.array([]).reshape(0,Train.shape[1])

    for t in range(0,Number_of_Batches_Train):
        
        if t%1000==0:
            print(t)
        seq_in = Train[t:t+timesteps_in_one_Batch,:]
        seq_out = Train[shift+t+timesteps_in_one_Batch,:]
        Xtrain=scipy.sparse.vstack((Xtrain,seq_in))
        Ytrain=scipy.sparse.vstack((Ytrain,seq_out))

    print(Xtrain.shape)
    print(Ytrain.shape)

    Xtrain_D=Xtrain.toarray()
    Input = Xtrain_D.reshape((Number_of_Batches_Train, SequenceLength, Train.shape[1]))
    Target = Ytrain.toarray()
    print(Input.shape)
    print(Target.shape)

    scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/Inputs/Training_30_s5_3.npz", Xtrain, compressed=True)
    scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/Inputs/TrainingTarget_30_s5_3.npz", Ytrain, compressed=True) 
    print("Train data is saved Location")
    return 

def MakeTestInputTarget(timesteps_in_one_Batch,Test,shift):
        
    Number_of_Batches_Test= Test.shape[0]-timesteps_in_one_Batch-shift
    Xtest=np.array([]).reshape(0,Test.shape[1])
    Ytest=np.array([]).reshape(0,Test.shape[1])

    for t in range(0,Number_of_Batches_Test):
        
        if t%1000==0:
            print(t)
        seq_in_test  = Test[t:t + timesteps_in_one_Batch,:]
        seq_out_test = Test[shift+t+timesteps_in_one_Batch,:]
        Xtest        = scipy.sparse.vstack((Xtest,seq_in_test))
        Ytest        = scipy.sparse.vstack((Ytest,seq_out_test))
         
    print(Xtest.shape)
    print(Ytest.shape)

    Xtest_D=Xtest.toarray()
    TestInput = Xtest_D.reshape((Number_of_Batches_Test, timesteps_in_one_Batch, Test.shape[1]))
    TestTarget=Ytest.toarray()
    print(TestInput.shape)
    print(TestTarget.shape)

    scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/Inputs/Test_30_s5_3.npz", Xtest, compressed=True)
    scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/Inputs/TestTarget_30_s5_3.npz", Ytest, compressed=True) 
    print("Test data is saved Location")

    return 

def buildMIDIfile(piano_roll):
    lowNote=22
    # 1) Find ON indices, transpose and sort by Note

    indices=np.where(piano_roll>0)
    indices_trans=np.transpose(indices)
    indices_sort=indices_trans[indices_trans[:,1].argsort()]
    NotesOn=np.unique(indices_sort[:,1])

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

#***********************************************************************************************************************
# 3)  Find Start End and Sequences within ON notes

    Note_Start_End=np.zeros((1,3))
    note_Start_End=np.zeros((1,3))
    indices_trans=Main

    i=0
    while i<indices_trans.shape[0]:
    
        k=0
        if len(indices_trans[i:])==1:  
            note_Start_End[0][0]=indices_trans[i][1]+lowNote     
            note_Start_End[0][1]= indices_trans[i][0]
            note_Start_End[0][2]=1
            break

        if indices_trans[i][1]!=indices_trans[i+1][1] and len(indices_trans[i+1])!=1:
            note_Start_End[0][0]=indices_trans[i][1]+lowNote     
            note_Start_End[0][1]= indices_trans[i][0]
            note_Start_End[0][2]=1

        if indices_trans[i][1]==indices_trans[i+1][1] and (indices_trans[i+1][0]-indices_trans[i][0]!=1):
            note_Start_End[0][0]=indices_trans[i][1]+lowNote     
            note_Start_End[0][1]= indices_trans[i][0]
            note_Start_End[0][2]=1
            
        if indices_trans[i][1]==indices_trans[i+1][1] and (indices_trans[i+1][0]-indices_trans[i][0]==1):
            for k in range(0,len(indices_trans[i:])):
                
                if len(indices_trans[i+k:])==1:
                    note_Start_End[0][0]=indices_trans[i][1]+lowNote     
                    note_Start_End[0][1]= indices_trans[i][0]
                    note_Start_End[0][2]=k+1+indices_trans[i][0]                
                    break

                if not (indices_trans[i+k][1]==indices_trans[i+1+k][1] and (indices_trans[i+1+k][0]-indices_trans[i+k][0]==1)):
                    note_Start_End[0][0]=indices_trans[i][1]+lowNote     
                    note_Start_End[0][1]= indices_trans[i][0]
                    note_Start_End[0][2]=k+1+indices_trans[i][0]
                    break
        
        Note_Start_End=np.vstack((Note_Start_End,note_Start_End))
        i += 1+k
    
    #***********************************************************************************************************************

    Note_Start_End_Sorted=Note_Start_End[Note_Start_End[:,1].argsort()]

    # 4)  [ Note, Start time, 1] 
    Start_times_temp=np.column_stack((Note_Start_End_Sorted[:,0],Note_Start_End_Sorted[:,1]))
    Start_times=np.column_stack((Start_times_temp,(np.ones((len(Start_times_temp),1)))))

    # 5)  [ Note, End time, 0] 
    End_times_temp=np.column_stack((Note_Start_End_Sorted[:,0],Note_Start_End_Sorted[:,2]))
    End_times=np.column_stack((End_times_temp,(np.zeros((len(Start_times_temp),1)))))

    # 6)  TotalStack = [ Note, Start time ]  vertical stacked with [ Note, End time] 
    TotalStack=np.vstack((Start_times,End_times))

    # 7)  TotalStack Sorted by the TIME Axis 
    TotalStack_sorted=TotalStack[TotalStack[:,1].argsort()]
    
    #***********************************************************************************************************************
    # 8)        MAKE MIDI FILE 

    file = open('C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Results/TestTarget.csv','w') 
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
    return
    #***********************************************************************************************************************
 