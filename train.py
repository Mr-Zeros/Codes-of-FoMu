from __future__ import print_function

import argparse
import os
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd

# Internal Imports for dataset and utility functions
from datasets.dataset import *
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Argument Parser to handle input configurations
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis')

# Directory where results will be saved
parser.add_argument('--results_dir', type=str, default='./results/', help='Directory to store results (Default: ./results)')
# Name of the experiment
parser.add_argument('--exp_name', type=str, default='3-center survival', help='Name of the experiment')

# Input dimensions for different types of data
parser.add_argument('--path_input_dim', type=int, default=512, help='Input dimension for pathology data')
parser.add_argument('--rad_input_dim', type=int, default=256, help='Input dimension for radiology data')

# Checkpointing and other pathing parameters
parser.add_argument('--n_classes', type=int, default=4, help='Number of output classes')

# Random seed for reproducibility
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible experiment')

# Number of folds for cross-validation
parser.add_argument('--k', type=int, default=5, help='Number of folds for cross-validation (default: 5)')

# Flag to enable logging data via tensorboard
parser.add_argument('--log_data', action='store_true', default=True, help='Log data using TensorBoard')

# Optimizer and Training Parameters
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer to use (default: adam)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1, due to varying bag sizes)')
parser.add_argument('--gc', type=int, default=32, help='Gradient accumulation step')
parser.add_argument('--max_epochs', type=int, default=80, help='Maximum number of epochs for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--weight_con', type=float, default=0.5, help='Weight coefficient for loss function')

# Slide-level loss function options
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'],
                    default='nll_surv', help='Slide-level classification loss function')

# Fraction of labels to use for training (can be used for semi-supervised learning)
parser.add_argument('--label_frac', type=float, default=1.0, help='Fraction of training labels (default: 1.0)')

# Coefficients for various losses
parser.add_argument('--bag_weight', type=float, default=0.7, help='Weight coefficient for bag-level loss')
parser.add_argument('--reg', type=float, default=1e-5, help='L2 regularization weight decay')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='Weight for uncensored patient data')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None',
                    help='Which submodules to apply L1 regularization to')

# Strength of L1 regularization
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1 regularization strength')

# Weighted sampling for imbalanced data
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')

# Early stopping flag
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')

# Parse the command line arguments
args = parser.parse_args()

# Set the device for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an experiment code based on the arguments and experiment name
args = get_custom_exp_code(args)

# Print the experiment code for reference
print("Experiment Name:", args.exp_code)

# Generate a unique nickname based on the current time
from datetime import datetime
nick_name = str(datetime.now().strftime("%Y%m%d%H%M%S"))

# Settings dictionary for easy access to the hyperparameters
settings = {
    'batch_size': args.batch_size,
    'num_splits': args.k,
    'max_epochs': args.max_epochs,
    'results_dir': args.results_dir,
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'bag_loss': args.bag_loss,
    'bag_weight': args.bag_weight,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size_wsi': args.model_size_wsi,
    'model_size_omic': args.model_size_omic,
    'use_drop_out': args.drop_out,
    'weighted_sample': args.weighted_sample,
    'gc': args.gc,
    'opt': args.opt
}

# Load datasets (placeholders for dataset paths and file locations)
print('\nLoading Dataset')

# Dataset instances for each center (YZ, KH, ZD, CY)
dataset_YZ = CustomDataset(csv_file='', wsi_dir='', rad_dir='')
dataset_KH = CustomDataset(csv_file='', wsi_dir='', rad_dir='')
dataset_ZD = CustomDataset(csv_file='', wsi_dir='', rad_dir='')
dataset_CY = CustomDataset(csv_file='', wsi_dir='', rad_dir='')

# DataLoader instances for each dataset
train_loader = DataLoader(dataset_YZ, batch_size=1, shuffle=True)
val_loader1 = DataLoader(dataset_KH, batch_size=1, shuffle=False)
val_loader2 = DataLoader(dataset_ZD, batch_size=1, shuffle=False)
val_loader3 = DataLoader(dataset_CY, batch_size=1, shuffle=False)

# Group the validation loaders into a list
val_loader = [val_loader1, val_loader2, val_loader3]

# Assign the nickname to args for saving files with unique names
args.nick_name = nick_name

# Train the model and evaluate its performance
val_latest, cindex_latest = train(train_loader, val_loader, args)

# Save the validation results to a pickle file with the experiment nickname
results_pkl_path = os.path.join(args.results_dir, f"_latest_val_results_{args.nick_name}.pkl")
save_pkl(results_pkl_path, val_latest)

# Save the C-index value from the latest fold to a CSV file
results_latest_df = pd.DataFrame({'val_cindex': [cindex_latest]})
results_latest_df.to_csv(os.path.join(args.results_dir, f'_summary_latest_{args.nick_name}.csv'))

# Final print statement indicating the end of the process
print("Done!")
