## Project idea:
Image classification of tomato plant leaves

## Goal:
The goal of this project is to determine the health status of a tomato plant based on images of its leaves. Different CNNs will be trained and compared.

The classification is divided into 10 health / infestation classes.

## Data set:
https://data.mendeley.com/datasets/tywbtsjrjv/1

## Usage
The parameters *FIT* and *TEST* determine whether the system should train or test a model. The model is selected via the *MODEL_CHOICE* parameter. When testing a model, a folder is automatically created in `/results`, along with a report and a confusion matrix.

The models used are defined in the `/models` subfolder and are assigned via the `model_manager` selection.

Once a CNN has been trained, it is automatically saved in `/models/SavedModels`.

The training results can be viewed using Tensorboard: **tensorboard --logdir=logs**