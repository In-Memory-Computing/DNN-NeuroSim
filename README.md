# DNN+NeuroSim
This repository contains code and steps to simulate the performance of an in-memory computing hardware accelerator using DNN+NeuroSim framework.

**What is DNN+NeuroSim?**\
DNN+NeuroSim is an integrated framework, which is developed in C++ and wrapped by Python to emulate the deep neural networks (DNN) performance, both for on-chip training and inference on the hardware accelerator based on in-memory computing architectures.

## System Requirements
The tool requires a Linux environment with system dependencies installed. Below are few tested environments that work,

1. Red Hat 5.11 (Tikanga), gcc v4.7.2, glibc 2.5,  (or)
2. Red Hat 7.3 (Maipo), gcc v4.8.5, glibc v2.1.7,  (or)
3. Ubuntu 16.04, gcc v5.4.0, glibc v2.23

On other linux environments tool requires gcc >= 4.5 for error free compilation.

To run the python wrapper, tool also requires a conda or python environment with an installation of PyTorch.
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
or
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

## Installation
Get the tool from github.
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```
The tool can also be cloned from original repository.
```
git clone https://github.com/neurosim/DNN_NeuroSim_V2.1.git
```
