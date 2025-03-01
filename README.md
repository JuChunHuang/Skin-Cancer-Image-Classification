# Skin-Cancer-Image-Classification

This repository contains code for training and evaluating a Convolutional Neural Network (CNN) for image classification using PyTorch.

## Directory Structure

    skin_cancer_classification/

    ├── data/
    │ └── dataset.py
    ├── model/
    │ └── simple_cnn.py
    ├── main.py
    ├── config.json
    ├── utils.py
    ├── poc.ipynb
    └── README.md

## Installation

Install the required packages:

```
pip install torch torchvision numpy scikit-learn matplotlib
```

## Configuration

Edit the `configs.json` file to set the paths to your training and test datasets, as well as other hyperparameters:

```
{
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "train_dir": "path/to/train",
    "val_dir": "path/to/val",
    "test_dir": "path/to/test",
    "plots_dir": "performance"
}
```

## Running the Code

- Train and Evaluate the Model:

  ```
  python main.py
  ```

  This will train the model using the specified configuration and save the loss and accuracy plots in the plots directory.

- View Results:

  After training, the script will print the following metrics for the test dataset:

  - Accuracy
  - Precision
  - Recall
  - F1 Score

  Additionally, the loss and accuracy plots will be saved in the plots directory.

## Code Overview

| Scripts         | Description                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| `dataset.py`    | Contains the function to load and preprocess the dataset and get dataloaders. |
| `simple_cnn.py` | Defines the CNN model architecture.                                           |
| `main.py`       | Main script to train, validate, and test the model.                           |
| `utils.py`      | Contains utility functions, such as saving plots.                             |
| `config.json`   | Configuration file for setting hyperparameters and dataset paths.             |
| `poc.ipynb`     | Jupyter notebook for prototyping and experimenting with the model and data.   |
