# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit - Fabio Palmisano -*-
# -*- date : 2024-01-20 -*-
# -*- Last revision: 2024-02-02 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Description: Classes to load the datasets-*-

# Import libraries
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np

# import files
import constants


class NoisyCIFAR100(Dataset):
    def __init__(
        self, train=True, noise_ratio=0.1, augmentation=True, num_samples=None
    ):
        transform = transforms.ToTensor()
        if augmentation:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        self.cifar100 = datasets.CIFAR100(
            root="../data/datasets", train=train, download=True, transform=transform
        )
        self.noise_ratio = noise_ratio

        # Limit the number of samples if num_samples is provided
        if num_samples is not None:
            self.cifar100.data = self.cifar100.data[:num_samples]
            self.cifar100.targets = self.cifar100.targets[:num_samples]

    def __getitem__(self, index):
        img, target = self.cifar100[index]

        if np.random.rand() < self.noise_ratio:
            target = np.random.randint(0, 100)

        return img, target

    def __len__(self):
        return len(self.cifar100)


class NoisyCIFAR10(Dataset):
    def __init__(
        self, train=True, noise_ratio=0.1, augmentation=False, num_samples=None
    ):
        transform = transforms.ToTensor()
        if augmentation:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        self.cifar10 = datasets.CIFAR10(
            root="../data/datasets", train=train, download=True, transform=transform
        )
        self.noise_ratio = noise_ratio

        # Limit the number of samples if num_samples is provided
        if num_samples is not None:
            self.cifar10.data = self.cifar10.data[:num_samples]
            self.cifar10.targets = self.cifar10.targets[:num_samples]

    def __getitem__(self, index):
        img, target = self.cifar10[index]

        if np.random.rand() < self.noise_ratio:
            target = np.random.randint(0, 10)

        return img, target

    def __len__(self):
        return len(self.cifar10)


class NoisyMNIST(Dataset):
    def __init__(
        self, train=True, noise_ratio=0.1, augmentation=False, num_samples=None
    ):
        transform = transforms.ToTensor()
        if augmentation:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        self.mnist = datasets.MNIST(
            root="../data/datasets", train=train, download=True, transform=transform
        )
        self.noise_ratio = noise_ratio

        # Limit the number of samples if num_samples is provided
        if num_samples is not None:
            self.mnist.data = self.mnist.data[:num_samples]
            self.mnist.targets = self.mnist.targets[:num_samples]

    def __getitem__(self, index):
        img, target = self.mnist[index]

        if np.random.rand() < self.noise_ratio:
            target = np.random.randint(0, 10)

        return img, target

    def __len__(self):
        return len(self.mnist)
