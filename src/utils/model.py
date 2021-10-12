import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib  # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap
import os
import logging

### Here we are just building the model but not training it , basically just the skeleton of it

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),             ## Flatten is nothing but reducing 2 or more dimensions into 1 dimension array
              tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
              tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
              tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)          ## The sequential API allows you to create models layer-by-layer for most problems. 


    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=METRICS)

    # Compile our model, this will create a Python object which will build the ANN.
    # The compilation steps also asks you to define the loss function and kind of optimizer you want to use.
    #  If you try to use predict with this compiled model your accuracy will be 10%, pure random output.     

    return model_clf ## <<< untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)
