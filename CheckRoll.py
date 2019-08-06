import numpy as np
import scipy
from scipy import sparse

PianoRoll_sparse =scipy.sparse.load_npz("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/Bach_p1.npz")
piano_roll = PianoRoll_sparse.toarray()

CheckSum=piano_roll.sum(axis=1)
np.savetxt("C:/Users/Poori/Desktop/Parinama/MakePianoRoll/SmallDataSet/Testing/CheckSum.csv", CheckSum, delimiter=",") 
