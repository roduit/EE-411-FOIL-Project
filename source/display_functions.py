# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-01-20 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Functions to plot results-*-

# import librairies
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import files
import constants


def visualize_dataset(dataset, num_images, label_names=None):
    """
    Visualizes examples from a given dataset.

    Args:
      dataset: The dataset containing images.
      num_images: Number of images to visualize.

    Note:
    - The dataset should be a tuple (images, labels), where `images` is a
      4D NumPy array with shape (num_samples, channels, height, width).
    """

    images, labels = dataset

    # Ensure num_images is not greater than the total number of images in the dataset
    num_images = min(num_images, len(images))

    # Randomly select num_images indices
    indices = np.random.choice(len(images), num_images, replace=False)

    # Create a grid of subplots
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))

    plt.figure(figsize=(5, 5))

    for i, idx in enumerate(indices, 1):
        plt.subplot(rows, cols, i)
        image = images[idx]
        plt.imshow(image, cmap="gray")
        if label_names is not None:
            plt.title(f"Label: {label_names[labels[idx]]}")
        else:
            plt.title(f"Label: {labels[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def display_error(error_lists, width_model_list, noise_ratio_list, train=False):
    """Function that display the error as a function of the width parameter
    Args:
        error_lists (list): list of lists containing the error for each width parameter
        width_model_list (list): list of width parameters
        noise_ratio_list (list): list of noise ratio
        train (bool, optional): True if the error is the train error, False if it is the test error. Defaults to False.
    """
    for noise_level, noise_curve in enumerate(error_lists):
        sns.lineplot(
            x=width_model_list,
            y=noise_curve,
            color=constants.color_palette[2 * noise_level],
            label=f"noise : {int(noise_ratio_list[noise_level]*100)}%",
        )
    plt.legend()
    plt.xlabel("ResNet18 width parameter")
    if train:
        plt.ylabel("Train error")
        plt.title("Train error as a function of the width parameter")
    else:
        plt.ylabel("Test error")
        plt.title("Test error as a function of the width parameter")
    plt.show()


def display_losses(losses, width_model_list, noise_ratio_list):
    """Function that display the loss as a function of the width parameter
    Args:
        losses (list): list of lists containing the loss for each width parameter
        width_model_list (list): list of width parameters
        noise_ratio_list (list): list of noise ratio
    """
    for noise_level, noise_curve in enumerate(losses):
        sns.lineplot(
            width_model_list,
            noise_curve,
            color=constants.color_palette[2 * noise_level],
            label=f"noise : {int(noise_ratio_list[noise_level]*100)}%",
        )
    plt.legend()
    plt.xlabel("ResNet18 width parameter")
    plt.ylabel("Loss")
    plt.title("Loss as a function of the width parameter")
    plt.show()


def display_optimizer_stats(adam_error, sgd_error, width_model_list, train=False):
    """Function that display the optimizer stats as a function of the width parameter
    Args:
        optimizer_stats (list): list of lists containing the optimizer stats for each width parameter
        width_model_list (list): list of width parameters
        noise_ratio_list (list): list of noise ratio
    """
    plt.plot(width_model_list, adam_error, "-", label="Adam")
    plt.plot(width_model_list, sgd_error, "-", label="SGD")
    plt.legend()
    if train:
        plt.ylabel("Train error")
        plt.title("Train Error as a function of the width parameter")
    else:
        plt.ylabel("Test error")
        plt.title("Test Error as a function of the width parameter")
    plt.xlabel("CNN width parameter")
    plt.show()


def visualize_convergence(losses, test_loss, test_accuracy, num_epochs, width):
    """Vizualize the convergence of the training process
    Args:
        losses (list): list of losses at each epoch
        test_loss (float): The average loss on the test set
        test_accuracy (float): The accuracy on the test set
        num_epochs (int): The number of epochs
        width (int): The width of the model
    """
    plt.figure(figsize=(12, 6))
    plt.title(f"Training loss for width = {width} and for {num_epochs} Epochs")
    plt.plot(np.linspace(1, num_epochs, num_epochs), losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
