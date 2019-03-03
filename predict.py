# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:23:59 2019
"""

from argparse import ArgumentParser
from preprocessor import load_dataset
import numpy as np
import random

#For F1 score
from keras import backend as K

#To load and save models
from keras.models import load_model

#Model path
model_dir = "saved_models"


def main():
    """
    Author Cory Kromer-Edwards
    Edits by: ...
    The main method that will build and run the model.
    """
    parser = build_parser()
    options = parser.parse_args()
    
    model_name = "model" + ",".join(str(x) for x in options.layers) + " d-" + str(options.dropout_prob) + " lr-" + str(options.learning_rate)
    
    model_path = model_dir + "/" + model_name + ".h5"
    
    pred(options.num_subjects, model_path)
        
        


def pred(num_subjects, model_path):
    """
    Author Cory Kromer-Edwards
    Edits by: ...
    The main method that will build and run the model.
    """
    model = load_model(model_path, custom_objects={'f1': f1})
    print("model loaded sucessfully")
	
    for _ in range(num_subjects):
        print("")
        data_point = random.randint(1, 4300)
        (X_subject, Y_subject) = load_dataset(False, data_point, True)
        print("Looking at person on row: ", data_point + 1)
        prediction = model.predict(X_subject)
        for drug_id in range(len(X_subject)):
            print("Drug id: ", X_subject[drug_id, 1])
            print("Actual Y: ", Y_subject[drug_id])
            print("Predicted max Y: ", np.argmax(prediction[drug_id]))
            print("Predicted Y output: ", prediction[drug_id])
            print("")
    

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
    
    parser.add_argument('-r', '--learning_rate', type=float,
                        dest='learning_rate', help='Learning rate',
                        metavar='LEARNING_RATE', default=0.001)
    
    parser.add_argument('-d', '--dropout_prob', type=float,
                        dest='dropout_prob', help='The keep probability in the dropout layer. (1=no drop out)',
                        metavar='DROPOUT', default=0.5)
    
    parser.add_argument('-l', '--layers', type=int,
                        dest='layers', help='The number of elements given is the number of hidden layers, and the \
                          integer given for each layer is the number of neurons.',
                          metavar='LAYERS', nargs='+', required=True)
    
    parser.add_argument('-n', '--num_subjects', type=int,
                        dest='num_subjects', help='The number of subjects that you want to predict.',
                          metavar='NUM_SUBJECTS', default=1)
    
    return parser


if __name__ == '__main__':
    main()

