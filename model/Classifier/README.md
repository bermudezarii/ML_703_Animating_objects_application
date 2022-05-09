# Trainable Base Model Template

This code works as a template for any transfer learning type of training applied to the base model (adult effusion TNI). In this document you will find the instructions to run the code succesfully.

## Requirements

1. Create a new virtual environment: `virtualenv --python=python3 venv`
2. Activate the environment using `source venv/bin/activate` if using Linux or `./venv/Scripts/activate` if using Windows.
3. Install the requirements using: `pip install -r requirements.txt`
4. To deactivate **after** running experiments enter: `deactivate`

## Instructions

1. Move the positive and negative images to their respective folder in [./dataset/](./dataset/)
2. Run [create_dataset.py](./code/create_dataset.py) to create the CSV files of the training and testing data used on the training.
3. Run [mean_std_dataset.py](./code/mean_std_dataset.py) to calculate the mean and standard deviation of the training dataset. Then copy each list to the [hyperaparameters](./code/hyperparameters.py) file on the lines 18 and 19.
4. Change any hyperparameters for your particular requirements.
5. Login into wandb typing `wandb login` here you will keep track of training status of your experiments. You can change the name of your experiment on [run.py](./code/run.py) in the line 18.
6. Run the code by entering in the code folder `cd code/` and then running the file: `python run.py`
7. To use the model for inference you can run the file `python inference.py`

## Notes

1. In case that any modification to the code wants to be performed, it will have to be done by changing the files. Future work will include command line parameters.