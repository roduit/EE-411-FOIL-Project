# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-01-20 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Additional functions used for this project-*-

#Import libraries
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
import time
from torch.utils.data import SubsetRandomSampler
from datetime import timedelta
import IPython
import time

#Import files
import constants
from models.resnet18k import make_resnet18k
from training_utils import fit, predict


def label_noise(dataset, noise_ratio=0.1, seed = 0):
    np.random.seed(seed)

    # Make a deep copy of the dataset
    noisy_dataset = copy.deepcopy(dataset)

    total_images = len(noisy_dataset)

    num_noisy_images = int(total_images * noise_ratio)

    indices = np.random.choice(total_images, num_noisy_images, replace=False)

    num_classes = len(np.unique(noisy_dataset.targets))

    for idx in indices:
        noisy_dataset.targets[idx] = torch.randint(0, num_classes, (1,))

    return noisy_dataset

def get_dataloaders(train_dataset, test_dataset, subsample_train_indices, subsample_test_indices, noise_ratio):
    noisy_train_dataset = label_noise(train_dataset, noise_ratio=noise_ratio)
    noisy_train_dataloader = torch.utils.data.DataLoader(
        dataset=noisy_train_dataset,
        batch_size=constants.BATCH_SIZE,
        sampler=SubsetRandomSampler(subsample_train_indices),
        num_workers=2)
    
    test_subset = torch.utils.data.Subset(test_dataset, subsample_test_indices)
    test_dataloader = torch.utils.data.DataLoader(
        test_subset, 
        batch_size=constants.TEST_BATCH_SIZE, 
        shuffle=False,
        num_workers=2)
    return noisy_train_dataloader, test_dataloader

def train_models(noise_ratio_list, width_model_list,train_dataset, test_dataset, optimizer='Adam', model='ResNet'):
    #initialize lists for storing results
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    subsample_train_indices = torch.randperm(len(train_dataset))[:constants.NUM_TRAIN_SAMPLES]
    subsample_test_indices = torch.randperm(len(test_dataset))[:constants.NUM_TEST_SAMPLES]

    for noise_ratio in noise_ratio_list:

        # Display parameters
        print(f'Model with noise ratio {noise_ratio}')
        out = display(IPython.display.Pretty('Starting'), display_id=True)
        start_time = time.time()

        # Initialize lists for noise ratio
        noise_train_loss = []
        noise_train_acc = []
        noise_test_loss = []
        noise_test_acc = []

        #Define dataloaders
        noisy_train_dataloader, test_dataloader = get_dataloaders(train_dataset, test_dataset, subsample_train_indices, subsample_test_indices, noise_ratio)

        out_epoch = display(IPython.display.Pretty('Starting'), display_id=True)

        # Iterate over different widths
        for width in width_model_list:
            out.update(IPython.display.Pretty('Training for width ' + str(width) + '/' + str(width_model_list[-1])))

            #Define model
            if model == 'ResNet':  
                ResNet = make_resnet18k(k=width)
                cnn = ResNet.to(constants.DEVICE)
            else:
                CNN = make_cnn(c = width)
                cnn = CNN.to(constants.DEVICE)
            if optimizer == 'SGD':
                optimizer = torch.optim.SGD(cnn.parameters(), lr=constants.SGD_LR)
            else:
                optimizer = torch.optim.Adam(cnn.parameters(), lr=constants.Adam_LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=2, verbose=False
            )
            #Train model
            losses = fit(
                model=cnn,
                train_dataloader=noisy_train_dataloader,
                optimizer=optimizer,
                epochs=constants.NUM_EPOCHS,
                device=constants.DEVICE,
                scheduler=scheduler,
                text = out_epoch
            )

            #Evaluate model
            train_loss, acc_train = predict(model=cnn, test_dataloader=noisy_train_dataloader, device=constants.DEVICE)
            test_loss, acc = predict(model=cnn, test_dataloader=test_dataloader, device=constants.DEVICE)

            #Store results
            noise_train_loss.append(train_loss)
            noise_train_acc.append(acc_train)

            noise_test_loss.append(test_loss)
            noise_test_acc.append(acc)

        #Store results
        train_losses.append(noise_train_loss)
        train_accuracies.append(noise_train_acc)
        test_losses.append(noise_test_loss)
        test_accuracies.append(noise_test_acc)

        #Print statistics
        elapsed_time = time.time() - start_time
        elapsed_time = str(timedelta(seconds=elapsed_time))
        print(f'Noise ratio {noise_ratio} done. Duration: {elapsed_time}')
        print('******************')

    return train_losses, train_accuracies, test_losses, test_accuracies