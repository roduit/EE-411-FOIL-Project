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
    """
    This function implements the core components of any Neural Network training regiment.
    In our stochastic setting our code follows a very specific "path". First, we load the batch
    a single batch and zero the optimizer. Then we perform the forward pass, compute the gradients and perform the backward pass. And ...repeat!
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
):
    """
    the fit method simply calls the train_epoch() method for a
    specified number of epochs.
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
        if no_improve_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        text.update(IPython.display.Pretty('Epoch ' + str(epoch+1) +'/'+str(epochs)+ ': Loss = ' + str(running_loss) + ' Time = ' + str(time.time() - start_time)))

    plt.title("Training curve")
    plt.plot(range(epochs), losses, "b-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


    return losses


def predict(
    model: nn.Module, test_dataloader: DataLoader, device: torch.device, verbose=True
):
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
