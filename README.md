<div align="center">
<img src="./ressources/logo-epfl.png" alt="Example Image" width="192" height="108">
</div>

<div align="center">
Ecole Polytechnique Fédérale de Lausanne
</div> 
<div align="center">
EE-411: Fundamentals of Inference and Learning
</div> 

# Deep Double Descent: When Bigger Models and More Data Hurt

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Conda Environment](#conda-environment)
- [Contributors](#contributors)
- [Theory and results](#theory-and-results)

## Abstract 

Modern machine learning research shows that the classical bias-variance trade-off has some limits as model size increases. The purpose of this project is to reproduce a research study conducted by engineers from Harvard and OpenAI [(Link to Publication)](https://arxiv.org/abs/1912.02292). This paper shows cases where increasing model size and data volume does not always lead to better performance.

## Project structure
```
├── Introduction
│   ├── Visualize the Dataset
│   │   ├── Define class dictionaries
│   │   ├── Visualize Datasets
│   │   |   ├── MNIST
│   │   |   ├── CIFAR10
│   │   |   └── CIFAR100
│   ├── Effect of transformation
│   │   ├── Horizontal Flip
│   │   └── Random Crop 
├── Some Experiments
│   ├── Required time
│   │   ├── CIFAR10 on T4
│   │   ├── CIFAR10 on V100
│   │   ├── CIFAR100 on T4
│   │   ├── CIFAR100 on V100
│   │   ├── MNIST on T4
│   │   └── MNIST on V100
│   ├── Testing the Convergence
│   │   ├── for K = 4
│   │   └── for K = 10
│   ├── Result with/without Scheduler
│   │   ├── No label noise
│   │   └── Label noise 20%
│   ├── Convergence with reduced datasets
├── Figure 4
│   ├── MNIST
│   │   ├── pickle_functions.py
│   │   └── read_functions.py
│   └── main.ipynb
├── Figure 6 : SGD vs Adam
```

The project is segmented into three primary sections. The initial phase involves a preprocessing task where various datasets are presented. Different transformations, such as horizontal flips or random crops, are set. Subsequently, the second part comprises conducting experiments to justify the selection of parameters and models for the subsequent phase. In the final part, an attempt is made to replicate figures 4 and 6 from the specified paper.

## Conda environment
A conda environment, named projectFOIL, with all Python packages that you might need for running the project. You can install it with the following command : 

```
conda create --name projectFOIL --file requirements.txt
```

Once installed, to activate the environment, please use `conda activate projectFOIL`. 


## Contributors
This project has been done by Vincent Roduit and Fabio Palmisano as a mandatory part of the course "EE-411: Fundamentals of Inference and Learning" given at Ecole Polytechnique Fédérale de Lausanne during the Fall semester of 2023.

## Results
