# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-01-20 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Functions to plot results-*-

#import librairies
import matplotlib.pyplot as plt
import numpy as np

def display_error(error_lists, width_model_list,noise_ratio_list, optimizer):
  for noise_level, noise_curve in enumerate(error_lists):
    acc = np.array([i.item() for i in noise_curve])
    test_error = 100 - acc
    plt.plot(width_model_list, test_error, '--',label = optimizer)
  plt.legend()
  plt.xlabel('CNN width parameter')
  plt.ylabel('Train error')
  plt.title('Train Error for SGD and Adam Optimizer')
  #plt.show()