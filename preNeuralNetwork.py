# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 07:49:59 2019
"""
from argparse import ArgumentParser
import keras.models as models
import keras.layers as layers
import keras.utils as utils
from preprocessor import load_dataset
import numpy as np
import os.path
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from predict import pred

#For F1 score
from keras import backend as K

#Tensorboard imports
import time
from tensorflow.python.keras.callbacks import TensorBoard

#To load and save models
from keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint

#Model path
model_dir = "saved_models"
log_dir = "logs"


def main():
    """
    Author Cory Kromer-Edwards
    Edits by: ...
    The main method that will build and run the model.
    """
    parser = build_parser()
    options = parser.parse_args()
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    
    model_name = "model" + ",".join(str(x) for x in options.layers) + " d-" + str(options.dropout_prob) + " lr-" + str(options.learning_rate)
    
    model_path = model_dir + "/" + model_name + ".h5"
        
        
    #Collect the dataset. Input is boolean for whether to debug which we need to be
    #set to false.
    (X, Y) = load_dataset(options.debug > 4)
       
    if options.debug > 3:
        plt.hist(Y, bins = [0, 1, 2, 3, 4]) 
        plt.title("histogram") 
        plt.show()
    
    if options.debug > 1:
        print("")
        print("Orignial X shape: ", X.shape)
        print("Orignial Y shape: ", Y.shape)
        print("")
        if options.debug > 2:
            test_index = 23
            print("")
            print("X at 23: ", X[test_index])
            print("Y at 23: ", Y[test_index])
            print("")
          
    
    #Normalize X values
#    minCol = np.min(X, axis=0)
#    maxMinCol = np.max(X, axis=0) - minCol
#    X = (X - minCol) / maxMinCol
        
    #Turn Y into a one hot vector
    Y = utils.to_categorical(Y, num_classes=4)
    
    #Uniformly randomly select train, validation, and test sets
    num_datapoints = X.shape[0]
    train_ind = np.random.choice(num_datapoints, int(num_datapoints * 0.7))
    val_ind = np.random.choice(num_datapoints, int(num_datapoints * 0.2))
    test_ind = np.random.choice(num_datapoints, int(num_datapoints * 0.1))
    
    x_train = X[train_ind, :]
    x_val = X[val_ind, :]
    x_test = X[test_ind, :]
    
    y_train = Y[train_ind]
    y_val = Y[val_ind]
    y_test = Y[test_ind]
    
    if options.debug > 2:
        print("")
        print("x train shape: ", x_train.shape)
        print("x val shape: ", x_val.shape)
        print("x test shape: ", x_test.shape)
        
        print("y train shape: ", y_train.shape)
        print("y val shape: ", y_val.shape)
        print("y test shape: ", y_test.shape)
        print("")
	
    if not os.path.isfile(model_path):        
        model = models.Sequential()
        for i in range(len(options.layers)):
            model.add(layers.Dense(options.layers[i], activation="relu"))
            model.add(layers.Dropout(options.dropout_prob))
            
        model.add(layers.Dense(4, activation="softmax"))
        
        model.compile(optimizer=Adam(lr=options.learning_rate), #"rmsprop", 
                      loss="categorical_crossentropy",
                      metrics=["accuracy", f1])
    else:
        model = load_model(model_path, custom_objects={'f1': f1})
        if options.debug > 0:
            print("model loaded sucessfully")
    
    #To run tensorboard, run this line in your terminal:
    #tensorboard --logdir=logs/
    tensor_board = TensorBoard(log_dir=log_dir+"/{0}_{1}".format(time.asctime(time.localtime(time.time())).replace(":", "-"), model_name))
    
    #Will save the model every epoch if it is the best model so far in terms of validation f1 score.
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_f1', save_best_only=True, mode='auto', period=1)
    
    model.fit(x_train, y_train, epochs=options.num_epochs, batch_size=options.batch_size,
              validation_data=(x_val, y_val), callbacks=[tensor_board, model_checkpoint])
    
    results = model.evaluate(x_test, y_test, batch_size=options.batch_size)
    print("Test (accuracy, f1, loss): ", results)
	
    if options.debug > 1:
        print("")
        if options.debug > 2:
            print(model.summary())
            
        pred(1, model_path)



#Code found here: https://stackoverflow.com/a/45305384
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  

def build_parser():
    """
    Author Cory Kromer-Edwards
    Edits by: ...
    Builds the parser based on input variables from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int,
                        dest='batch_size', help='Batch size',
                        metavar='BATCH_SIZE', default=64)
    
    parser.add_argument('-r', '--learning_rate', type=float,
                        dest='learning_rate', help='Learning rate',
                        metavar='LEARNING_RATE', default=0.001)
    
    parser.add_argument('-n', '--num_epochs', type=int,
                        dest='num_epochs', help='Number of epochs',
                        metavar='NUM_EPOCHS', default=5)
    
    parser.add_argument('--debug', type=int,
                        dest='debug', help='Print debug, and extra, information. Values [1-5)',
                        metavar='DEBUG', default=1)
    
    parser.add_argument('-d', '--dropout_prob', type=float,
                        dest='dropout_prob', help='The keep probability in the dropout layer. (1=no drop out)',
                        metavar='DROPOUT', default=0.5)
    
    parser.add_argument('-l', '--layers', type=int,
                        dest='layers', help='The number of elements given is the number of hidden layers, and the \
                          integer given for each layer is the number of neurons.',
                          metavar='LAYERS', nargs='+', required=True)
    
    return parser


if __name__ == '__main__':
    main()