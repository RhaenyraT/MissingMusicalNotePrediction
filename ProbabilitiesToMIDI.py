import numpy as np
import scipy 
from scipy import sparse
from sklearn.metrics import roc_curve

print("Changing raw probabilities to values")
PredictedRawProbabilities=np.genfromtxt("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/3L_Best_PredictedRawProbabilities.csv",delimiter=",")
Prob_target=np.genfromtxt("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/batch_Target.csv",delimiter=",")

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
     temp=PredictedRawProbabilities[l,:]>optimal_threshold
     Predicted_Pianoroll[l,:]=temp.astype(int)
Pianoroll=sparse.csr_matrix(Predicted_Pianoroll)
scipy.sparse.save_npz("C:/Users/Apoorva.Radhakrishna/Desktop/Shareernalam/validation/PianoRolls/Depth/3/3L_Best_PredictedRoll.npz", Pianoroll, compressed=True)

