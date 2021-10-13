import os
import tensorflow as tf
import numpy as np
import pandas as pd
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
from src.utils.save_plots import save_plot
from src.utils.model import get_log_path
import argparse

def training(config_path):                          ## Here we are defining the training function which is called from main function
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)


    # So, now we need to train our model so that the parameters get tuned to provide 
    # the correct outputs for a given input.
    # We do this by feeding inputs at the input layer and then getting an output, 
    # we then calculate the loss function using 
    # the output and use backpropagation to tune the model parameters.
    # This will fit the model parameters to the data.

    log_dir = get_log_path()

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    ## Early early_stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    ## Check points saving dir
    CKPT_path = config["artifacts"]["CKPT_path"]
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, checkpointing_cb]


    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_data=VALIDATION_SET, callbacks=CALLBACKS_LIST)

   
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_name = config["artifacts"]["model_name"]

    save_model(model, model_name, model_dir_path)

    plots_dir = config["artifacts"]["plots_dir"]
    plots_name = config["artifacts"]["plots_name"]

    save_plot(pd.DataFrame(history.history), plots_name , plots_dir)




    file_writer = tf.summary.create_file_writer(logdir=log_dir)

    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handritten digit samples", images, max_outputs=25, step=0)



if __name__ == '__main__':

    # Creating a a container called args to store our arguements
    args = argparse.ArgumentParser(description="It calls the training function with the configuration file as its arguement")

    args.add_argument("--config", "-c", default="config.yaml")          ## Here --config and -c provides insight about the type of arguement the function accepts

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)