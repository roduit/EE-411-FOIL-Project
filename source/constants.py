# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-01-20 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Constants used for this project-*-

#Import libraries
import os
import torch
import seaborn as sns

#Data path
DATA_FOLDER = '../data'
PICKLE_FOLDER = os.path.join(DATA_FOLDER, 'pickle')
FIG4_FOLDER = os.path.join(PICKLE_FOLDER, 'fig4')
FIG6_FOLDER = os.path.join(PICKLE_FOLDER, 'fig6')

#Model constants
BATCH_SIZE = 128
TEST_BATCH_SIZE  = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 1

#Constants part1
Adam_LR = 0.0001
SGD_LR = 0.1
NUM_TRAIN_SAMPLES = 50000
NUM_TEST_SAMPLES = 10000

#Plot constants
color_palette = sns.color_palette('viridis')