# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-01-20 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Functions to plot results-*-

#import librairies
import matplotlib.pyplot as plt
import numpy as np

def display_error(error_lists, width_model_list,noise_ratio_list):
  for noise_level, noise_curve in enumerate(error_lists):
    acc = np.array([i.item() for i in noise_curve])
    test_error = 100 - acc
    plt.plot(width_model_list, test_error, '--',label = f'noise : {int(noise_ratio_list[noise_level]*100)}%')
  plt.legend()
  plt.xlabel('ResNet18 width parameter')
  plt.ylabel('Test error')
  plt.title('Test Error for different noise level ')
  plt.show()