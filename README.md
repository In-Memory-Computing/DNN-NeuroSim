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
Get the tool from github
```
git clone https://github.com/In-Memory-Computing/DNN_NeuroSim.git
```
The tool can also be cloned from original repository
```
git clone https://github.com/neurosim/DNN_NeuroSim_V2.1.git
```

## Execution Steps
1. Set up hardware parameters in NeuroSim Core (*./NeuroSIM/Param.cpp*) and compile the Code.
   ```
   make
   ```
2. Define the network structure in Network.csv file. (*./NeuroSIM/Network.csv*). Example: VGG8(Default)
   
3. Set up hardware constraints in Python wrapper (train.py)   

4. Run Pytorch wrapper integrated with NeuroSim (for online-training performance)
   ```
   python train.py
   ```
   For inference, set the 'model_path' variable inside inference.py file to the most recent trained weights file and run the wrapper.
   ```
   python inference.py
   ```

## Results
- The recent model weights (from training) can be found in 'latest.pth' file under log directory (*./log/default/ADCprecision=5/.../latest.pth*)
- Hardware performance for each epoch is captured under *./NeuroSim_Results_Each_Epoch* folder
  ![Results each epoch](./sample_outputs/Results_EachEpoch.png)
- Also the layer-wise cumulative training performance is displayed on the screen as well as under the log folder (*./log/default/ADCprecision=5/.../ \*.log*)
- Similarly the inference performance can be found under the log directory((*./log/default/ADCprecision=6/.../ \*.log*)

## Hardware parameters
User defined hardware parameters from 'Param.cpp' file.

-

## References
- X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, " DNN+NeuroSim: An end-to-end benchmarking framework for compute-in-memory accelerators with versatile device technologies," IEEE International Electron Devices Meeting (IEDM), 2019.
- X. Peng, S. Huang, H. Jiang, A. Lu and S. Yu, “DNN+ NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for Training,” arXiv, 2020. 
- S. Wu, et al. "Training and inference with integers in deep neural networks," arXiv: 1802.04680, 2018. 
- A. Lu, X. Peng, W. Li, H. Jiang, A. Lu and S. Yu, “NeuroSim Simulator for Compute-in-Memory Hardware Accelerator: Validation and Benchmark,” IEEE, 2021. 
- Repository:
  https://github.com/neurosim/DNN_NeuroSim_V2.1
- Installation Tutorial:
  https://www.youtube.com/watch?v=pdT9NCn1L44&t=735s
- User Manual:
  https://github.com/neurosim/DNN_NeuroSim_V2.1/blob/master/Documents/DNN%20NeuroSim%20V2.1%20User%20Manual.pdf

