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
# Define model creation functions
def make_resnet18k_2channels_MNIST(k):
    return make_resnet18k_2channels(k=k).to(constants.DEVICE)

def make_resnet18k_3channels_CIFAR10(k):
    return make_resnet18k_3channels(k=k, num_classes=10).to(constants.DEVICE)

def make_resnet18k_3channels_CIFAR100(k):
    return make_resnet18k_3channels(k=k, num_classes=100).to(constants.DEVICE)

def make_cnn_CIFAR10(k):
    return make_cnn(c=k).to(constants.DEVICE)

def make_cnn_CIFAR100(k):
    return make_cnn(c=k, num_classes=100).to(constants.DEVICE)

# Define optimizer creation functions
def make_SGD(cnn):
    return torch.optim.SGD(cnn.parameters(), lr=constants.SGD_LR)

def make_Adam(cnn):
    return torch.optim.Adam(cnn.parameters(), lr=constants.Adam_LR)

model_creation_functions = {
    ('ResNet', 'MNIST'): make_resnet18k_2channels_MNIST,
    ('ResNet', 'CIFAR10'): make_resnet18k_3channels_CIFAR10,
    ('ResNet', 'CIFAR100'): make_resnet18k_3channels_CIFAR100,
    ('CNN', 'CIFAR10'): make_cnn_CIFAR10,
    ('CNN', 'CIFAR100'): make_cnn_CIFAR100
}
# Map optimizer names to creation functions
optimizer_creation_functions = {
    'SGD': make_SGD,
    'Adam': make_Adam
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
            # Create model
            if (model, dataset_name) in model_creation_functions:
                ModelCreationFunction = model_creation_functions[(model, dataset_name)]
                cnn = ModelCreationFunction(width)
            else:
                raise ValueError(f"Invalid model-dataset combination: {model}-{dataset_name}")
            # Create optimizer
            if optimizer in optimizer_creation_functions:
                OptimizerCreationFunction = optimizer_creation_functions[optimizer]
                optimizer = OptimizerCreationFunction(cnn)
            else:
                raise ValueError(f"Invalid optimizer name: {optimizer}")
            # Create scheduler
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

def model_convergence(optimizer='Adam', model='ResNet',dataset_name='MNIST',num_epochs=10, noise_ratio=0.1, width=1,scheduler=None):
    print(f'Training model')
    start_time = time.time()
    out_epoch = display(IPython.display.Pretty('Starting'), display_id=True)
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
    
    if (model, dataset_name) in model_creation_functions:
                ModelCreationFunction = model_creation_functions[(model, dataset_name)]
                cnn = ModelCreationFunction(width)
    else:
        raise ValueError(f"Invalid model-dataset combination: {model}-{dataset_name}")
    # Create optimizer
    if optimizer in optimizer_creation_functions:
        OptimizerCreationFunction = optimizer_creation_functions[optimizer]
        optimizer = OptimizerCreationFunction(cnn)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer}")
    if scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, verbose=False
        )
    #Train model
    losses = fit(
        model=cnn,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=num_epochs,
        device=constants.DEVICE,
        scheduler=scheduler,
        text = out_epoch
    )
    test_loss,test_accuracy = predict(model=cnn, test_dataloader=test_dataloader, device=constants.DEVICE)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    elapsed_time = str(timedelta(seconds=elapsed_time))
    print(f'Training done. Duration: {elapsed_time}')
    return losses, test_loss,test_accuracy
