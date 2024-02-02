# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-01-20 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Functions used to train models-*-
# -*- Source: Adapted from the course "Fundamentals of Inference and Learning" of EPFL -*-

#import librairies
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
from torch.utils.data import DataLoader
import pickle
import IPython
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    """This Function trains the model for one epoch. It is called by the fit() method.
    Args:
        model (nn.Module): The model to train
        train_dataloader (DataLoader): The training dataloader
        optimizer (torch.optim.Optimizer): The optimizer used for training
        device (torch.device): The device on which the model is trained
    Returns:
        running_loss / len(train_dataloader) (float): The average loss over the epoch
    """
    running_loss = 0.0
    model = model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # move data and target to device
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # do the forward pass
        output = model(data)

        # compute the loss
        loss = F.cross_entropy(output, target)

        # compute the gradients
        loss.backward()

        # perform the gradient step
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(train_dataloader)

def fit(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    scheduler: ReduceLROnPlateau,
    text,
    patience: int = 5,
    verbose=False
):
    """This function trains the model for a given number of epochs.
    Args:
        model (nn.Module): The model to train
        train_dataloader (DataLoader): The training dataloader
        optimizer (torch.optim.Optimizer): The optimizer used for training
        epochs (int): The number of epochs to train
        device (torch.device): The device on which the model is trained
        scheduler (ReduceLROnPlateau): The learning rate scheduler
        text (IPython.display.Pretty): The text to display the training progress
        patience (int, optional): The number of epochs to wait for improvement before stopping the training. Defaults to 5.
        verbose (bool, optional): If True, prints the loss at each epoch. Defaults to False.
    Returns:
        losses (list): The list of losses at each epoch
    """

    # keep track of the losses in order to visualize them later
    losses = []
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )
        losses.append(running_loss)

        if scheduler is not None:
            scheduler.step(running_loss)

        # Check if loss has improved
        if running_loss < best_loss:
            best_loss = running_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # If loss hasn't improved for a certain number of epochs, stop training
        if (no_improve_epochs >= patience) or running_loss < 1e-4:
          if verbose:
            print(f'Early stopping at epoch {epoch+1}')
          break

        text.update(IPython.display.Pretty('Epoch ' + str(epoch+1) +'/'+str(epochs)+ ': Loss = ' + str(running_loss) + ' Time = ' + str(time.time() - start_time)))
    return losses


def predict(
    model: nn.Module, test_dataloader: DataLoader, device: torch.device, verbose=True
):
    """This function evaluates the model on the test set.
    Args:
        model (nn.Module): The model to evaluate
        test_dataloader (DataLoader): The test dataloader
        device (torch.device): The device on which the model is evaluated
        verbose (bool, optional): If True, prints the loss and accuracy. Defaults to True.
    Returns:
        test_loss (float): The average loss on the test set
        accuracy (float): The accuracy on the test set
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="sum")
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.sampler)
    accuracy = 100.0 * correct / len(test_dataloader.sampler)

    return test_loss, accuracy.item()
