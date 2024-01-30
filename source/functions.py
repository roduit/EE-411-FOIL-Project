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
import time
from datetime import timedelta
import IPython
import time
from torchvision import datasets
from torch.utils.data import Dataset

#Import files
import constants
from models.resnet18k_2channels import make_resnet18k_2channels
from models.resnet18k_3channels import make_resnet18k_3channels
from training_utils import fit, predict
from models.mcnn import make_cnn
from data_classes import*

#Define dictionary constant
#Datasets
dataset_classes = {
    'CIFAR100': NoisyCIFAR100,
    'CIFAR10': NoisyCIFAR10,
    'MNIST': NoisyMNIST
}

def train_models(noise_ratio_list, width_model_list, optimizer='Adam', model='ResNet',dataset_name='MNIST'):
    #initialize lists for storing results
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for noise_ratio in noise_ratio_list:

        #Get Dataloader for train and test
        if dataset_name in dataset_classes:
            DatasetClass = dataset_classes[dataset_name]
            train_dataset = DatasetClass(train=True, noise_ratio=noise_ratio,num_samples=constants.NUM_TRAIN_SAMPLES)
            test_dataset = DatasetClass(train=False, noise_ratio=noise_ratio,num_samples=constants.NUM_TEST_SAMPLES)

            train_dataloader = torch.utils.data.DataLoader(
                                dataset=train_dataset,
                                batch_size=constants.BATCH_SIZE,
                                num_workers=2)
            test_dataloader = torch.utils.data.DataLoader(
                                dataset=test_dataset, 
                                batch_size=constants.TEST_BATCH_SIZE, 
                                shuffle=False,
                                num_workers=2)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        # Display parameters
        print(f'Model with noise ratio {noise_ratio}')
        out = display(IPython.display.Pretty('Starting'), display_id=True)
        start_time = time.time()

        # Initialize lists for noise ratio
        noise_train_loss = []
        noise_train_acc = []
        noise_test_loss = []
        noise_test_acc = []

        out_epoch = display(IPython.display.Pretty('Starting'), display_id=True)

        # Iterate over different widths
        for width in width_model_list:
            out.update(IPython.display.Pretty('Training for width ' + str(width) + '/' + str(width_model_list[-1])))

            #Define model
            if model == 'ResNet' and dataset_name == 'MNIST':  
                ResNet = make_resnet18k_2channels(k=width)
                cnn = ResNet.to(constants.DEVICE)
            elif model == 'ResNet' and dataset_name == 'CIFAR10':
                ResNet = make_resnet18k_3channels(k=width,num_classes=10)
                cnn = ResNet.to(constants.DEVICE)
            elif model == 'ResNet' and dataset_name == 'CIFAR100':
                ResNet = make_resnet18k_3channels(k=width,num_classes=100)
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
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                epochs=constants.NUM_EPOCHS,
                device=constants.DEVICE,
                scheduler=scheduler,
                text = out_epoch
            )

            #Evaluate model
            train_loss, acc_train = predict(model=cnn, test_dataloader=train_dataloader, device=constants.DEVICE)
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