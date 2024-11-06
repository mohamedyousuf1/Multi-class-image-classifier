# Multi class image Classification using CNN Models in Pytorch

This project implements a polyp shape classification system using various CNN models. The code is designed to automatically identify classes based on the folder structure of the dataset and train a model to classify images into these classes. The assigned label for each class will be generated in classes.json file inside the output file

## Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- scikit-learn
- tqdm
- PIL (Pillow)
- matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mohamedyousuf1/Multi-class-image-classifier.git
   cd Multi-class-image-classifier-main

2. Create a virtual environment and activate it:
   ```bash
    python3 -m venv venv
    source venv/bin/activate

3. Install the required packages:
   ```bash
    pip install torch torchvision scikit-learn tqdm pillow matplotlib
    
## Dataset Structure
The dataset should be organized in the following structure:


The code will automatically identify the classes based on the subdirectory names under the `train` and `val` directories.

## Running the Code

1. Prepare your dataset in the structure mentioned above.
    ```
    Dataset/
        train/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
                image1.jpg
                image2.jpg
                ...
            ...
        val/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
                image1.jpg
                image2.jpg
                ...
            ...
    ```
    The number of calsses and classes names will be taken from the names of the folders inside Dataset folder.


2. Run the training script, for example:
   ```bash
   python main.py --train-path /path/to/Dataset/train --validation-path /path/to/Dataset/val --model densenet3d --batch-size 16 --lr 0.001 --num-epochs 100 --device cuda --output-dir /path/to/output

## Arguments
- `--batch-size`: Batch size for dataloaders (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--num-epochs`: Number of training epochs (default: 100)
- `--model`: Type of model to use (default: 'cnn3d')
- `--num-workers`: Number of workers for dataloaders (default: 4)
- `--loss`: Loss function to use (choices: ['bce', 'mse', 'focal'], default: 'bce')
- `--optimizer`: Optimizer to use (choices: ['adam', 'sgd'], default: 'adam')
- `--train-path`: Path to the training data
- `--validation-path`: Path to the validation data
- `--device`: Device to use for training (default: 'cuda')
- `--resume`: Path to the model checkpoint to resume training
- `--weight-decay`: Weight decay for the optimizer (default: 0)
- `--output-dir`: Output directory to save images and results (default a unique folder contains the date and time of the starting experiment will be created)

## Output
The results will be saved in the specified output directory. The following files will be generated:

- `args.json`: JSON file containing the arguments used for training.
- `model_summary.txt`: Text file containing the model architecture and the number of parameters.
- `results.csv`: CSV file containing the training and validation metrics for each epoch.
- `classes.json`: JSON file containing the class-to-index mapping.
- `metrics_val_vs_train.png`: Plot of the training and validation metrics over epochs.
- Checkpoints of the best models based on validation loss, F1 score, and recall.


## Notes
- Ensure that your dataset is properly structured as mentioned above.
- The code will automatically identify the classes based on the folder names under the `train` and `val` directories.
- You can resume training from a checkpoint by specifying the `--resume` argument with the path to the checkpoint file.

