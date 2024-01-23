# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-01-20 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Constants used for this project-*-

#Import libraries
import os
import torch


#Model constants
BATCH_SIZE = 128
TEST_BATCH_SIZE  = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 4e3

#Constants part1
Adam_LR = 0.0001
NUM_SAMPLES = in