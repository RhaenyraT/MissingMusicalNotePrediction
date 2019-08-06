from __future__ import print_function
import numpy as np
import hyperas
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras import regularizers
from keras.callbacks import EarlyStopping
import sklearn
from sklearn.metrics import roc_curve, auc

def data():
    import scipy 
    from scipy import sparse
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
    Timesteps              = 10
    shift                  = 5
    Col2Select=np.arange(0,87,1)
    Number_of_Batches_Train  =  Train.shape[0]-Timesteps-shift #86902 #
    Number_of_Batches_Test   =  Test.shape[0]-Timesteps-shift #28950 #
    Number_of_Batches_Val    =  Val.shape[0]-Timesteps-shift 

    Input_unshaped    = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/TrainX_10_s5.npz")
    Input_unshapedD   = Input_unshaped.toarray()
    print(Input_unshapedD.shape)
    Input_unshapedDsmall=Input_unshapedD[:,Col2Select]
    print(Input_unshapedDsmall.shape)

    x_train=Input_unshapedDsmall.reshape(Number_of_Batches_Train,Timesteps,Input_unshapedDsmall.shape[1])  #,
    print(x_train.shape)

    Target_unshaped   = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/TrainY_10_s5.npz")
    Target_unshapedD   = Target_unshaped.toarray()
    Target_unshapedDsmall=Target_unshapedD[:,Col2Select]
    y_train           = Target_unshapedDsmall
    print(y_train.shape)

    ValX_unshaped   = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/ValX_10_s5.npz")
    ValX_unshapedD  = ValX_unshaped.toarray()
    print(ValX_unshapedD.shape)
    ValX_unshapedDsmall=ValX_unshapedD[:,Col2Select]
    print(ValX_unshapedDsmall.shape)
    x_val=ValX_unshapedDsmall.reshape(Number_of_Batches_Val,Timesteps,ValX_unshapedDsmall.shape[1])  #,
    print(x_val.shape)

    ValY_unshaped   = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/ValY_10_s5.npz")
    ValY_unshapedD  = ValY_unshaped.toarray()
    print(ValY_unshapedD.shape)
    ValY_unshapedDsmall=ValY_unshapedD[:,Col2Select]
    print(ValY_unshapedDsmall.shape)
    y_val  = ValY_unshapedDsmall
    print(y_val.shape)
    
    
    print("Importing and reshaping Test Input")
    TestInput_unshaped = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/TestX_10_s5.npz")
    TestInput_Dense    = TestInput_unshaped.toarray()
    print(TestInput_Dense.shape)
    x_test          = TestInput_Dense.reshape(Number_of_Batches_Test,Timesteps,TestInput_Dense.shape[1])
    
    TestOutput_unshaped   = scipy.sparse.load_npz("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/Inputs/mixie/TestY_10_s5.npz")
    TestOutput_unshapedD  = TestOutput_unshaped.toarray()
    y_test               = TestOutput_unshapedD
    return x_train, y_train,x_val,y_val, x_test, y_test


def create_model(x_train, y_train ,x_val,y_val,x_test, y_test):
    import keras
    from hyperopt import Trials, STATUS_OK, tpe
    from keras.layers import LSTM
    from keras.datasets import mnist
    from keras.layers.core import Dense, Dropout, Activation
    from keras.models import Sequential
    from keras.utils import np_utils
    from hyperas import optim
    from hyperas.distributions import choice, uniform
    from keras.layers import Bidirectional
    from keras import regularizers
    from keras.callbacks import EarlyStopping
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM({{choice([100])}}),merge_mode='sum',input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Activation({{choice(['relu','tanh','sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(x_train.shape[2], kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('sigmoid'))



    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                  optimizer={{choice(['rmsprop'])}})
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    result = model.fit(x_train, y_train,validation_data=(x_val, y_val), 
              batch_size={{choice([205])}},
              epochs=25,
              verbose=2,callbacks=[early_stopping])
    RawProbabilities = model.predict(x_test)
    AUC_Score=sklearn.metrics.roc_auc_score(y_test.flatten(), RawProbabilities.flatten())
    #get the lowest validation loss of the training epochs
    validation_loss = np.amin(result.history['val_loss']) 
    print('Lowest validation loss of epoch:', validation_loss)
    print('AUC Score:', AUC_Score)
    return {'loss': -AUC_Score, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials())
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)