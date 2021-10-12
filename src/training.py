import os
import pandas as pd
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
from src.utils.save_plots import save_plot
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


    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_data=VALIDATION_SET)

   
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_name = config["artifacts"]["model_name"]

    save_model(model, model_name, model_dir_path)

    plots_dir = config["artifacts"]["plots_dir"]
    plots_name = config["artifacts"]["plots_name"]

    save_plot(pd.DataFrame(history.history), plots_name , plots_dir)
    print("*"*20)
    print(history.history)
    print("*"*20)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)