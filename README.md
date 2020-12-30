StaggaredFDTD
=============
Samuel Pichardo, Ph.D  
Assistant Professor  
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute  
Cumming School of Medicine,  
University of Calgary   
samuel.pichardo@ucalgary.ca  
www.neurofus.ca

**Software library for FDTD of viscoelastic equation using an staggered grid arrangement with  CPU- or GPU-based backends (OpenMP, CUDA and OpenCL)**

This tool solves in time domain the viscoelastic equation for wave propagation using an staggered grid solution. While the underlying equations and methods were  developed primarily for seismic simulation, the StaggaredFDTD library was developed for biomedical applications to study the ultrasound transmission through bone material.

Users are invited to read first the paper associated to this work to review specifics of the solution (including details on boundary matching layers and averaging operators for heterogenous media):

  Pichardo, S., Moreno-Hernández, C., Drainville, R.A., Sin, V., Curiel, L. and Hynynen, K., 2017. *A viscoelastic model for the prediction of transcranial ultrasound propagation: application for the estimation of shear acoustic properties in the human skull*. Physics in Medicine & Biology, 62(17), p.6938. https://doi.org/10.1088/1361-6560/aa7ccc

If you find this software useful for your research, please consider adding a citation to the above reference in your publications and presentations.

The software implementation supports multiple front-ends (Matlab and Python), OS environments (Windows, Linux, MacOS) and CPU/GPU execution. While the implementation supports CPU-based execution, a modern NVIDIA-based GPU or AMD in MacOS is highly recommended.

Please note that Python and Linux are the preferred frontend and OS. MacOS support has degraded significantly as since several MacOS versions the support for NVIDIA cards is practically inexistent and OpenCL is officially being out of support beyond Big Slur. Nevertheless, OpenCL in MacOS still gives excellent performance. Back in 2017,  an OpenCL implementation with a AMD Vega56 outperformed a NVIDIA Titan XP via CUDA (31 s vs 47s, for a simulation with a domain size of 158$\times$154$\times$111 and 5800 time steps). In 2020, a RTX 2080 SUPER via CUDA can run a given simulation in around 5s for a domain 118\times$118\times$211 and 817 time steps, while a much simpler Radeon Pro 560 (used in MacBook Pro 2017) takes 18 s, and an i7-9700 (4 cores x 2 with hyperthreading) via OpenMP takes 35s.

*Just to say it is a pity HPC computing has been pretty much abandoned by MacOS*.

# Requirements
## Python 3.6 and up - x64
The code should work with Python 3.6 and up, including Anaconda, EDM Canopy (both supported in all 3 main OSs) or Linux-based distribution Python installation; same for brew-based installation for MacOS.

Use of virtual environments is highly recommended

Dependencies:
* numpy>=1.15.1
* scipy>=1.1.0
* h5py>=2.9.0
* pydicom>=1.3.0


## CUDA
The code has been verified to work from CUDA 9 to CUDA 11. Highly likely older versions of CUDA (7, 8) should work without a problem. Be sure of installing the CUDA samples and take note of the location where they were installed.

Please note CUDA is not supported anymore in MacOS as NVIDA and Apple seem to be in a feud since years.
## CMAKE
CMAKE version 3.16.3 and up should be good
## Linux
Overall, any LTS-type distribution is recommended to be sure CUDA compiler supports your default GCC installation. If your installation can run the default examples of CUDA, then you should be good.
## Linux in Windows through WSL2
Starting in 2020, support for CUDA execution directly in WSL2 became possible. I highly recommended you give it a try as I have had good experiences with it. Just follow the official instructions from NVIDIA (https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
## Windows native
Just be sure of having a VStudio installation that is compatible with your CUDA version (for example, CUDA 11.2 supports officially VS 2015 to 2019), and be sure of installing the CUDA samples and take note of the location where they were installed.

## MacOS
Any recent version of MacOS and XCode should be enough. Please note that the CPU version in MacOS does not support OpenMP (still working in a definitive solution via brew llvm or brew gcc). However, the OpenCL version works without a problem in Intel-based integrated GPUs and AMD GPUs.

## Installation
If CUDA and supporting compiler are correctly installed, then it is straightforward to install using `pip install <directory>`. You need to specify the location where the CUDA samples are installed as those are required for the compilation.

Below a few examples for both Linux and Windows; the command must be run in the directory where FDTDStaggered3D_with_relaxation was cloned (i.e. /home/<user>/Github)
### Linux
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc pip3 install  FDTDStaggered3D_with_relaxation/
```
or
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc pip3 install --user FDTDStaggered3D_with_relaxation/
```
if you do not have write access to the global Python installation
### Windows
```
set "CUDA_SAMPLES_LOCATION=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2\common\inc" && pip install  FDTDStaggered3D_with_relaxation\
```

# Structure of code
The FDTD solution is accessed as a Python external function. The primary method to execute a simulation is via the class
`FDTDStaggered3D_with_relaxation.PropagationModel` and its main function `StaggeredFDTD_3D_with_relaxation`

After installation, the class can be instatiated as:
```
from FDTDStaggered3D_with_relaxation import PropagationModel

Model=PropagationModel()
```


## Multi-platform single code
The underlying extension code (start at `FDTDStaggered3D_with_relaxation_python.c`) uses extensively C macros to provide a full agnostic implementation that remains as efficient as possible regardless if using a CPU or GPU. It supports via macro definitions compilation for Matlab and Python extensions; support for CUDA, X86_64 and OpenCL architectures; single or double precision.

Consult `setup.py` and `CompileMatlab.m` to review how all the potential modalities are generated.

Please note that Matlab implementation is still missing an updated high-level equivalence to `FDTDStaggered3D_with_relaxation\PropagationModel.py`. Given most of my personal needs gravitate around Python, the Matlab frontend is a low-priority by the time being.

## How to use
After installation, you can consult the Jupyter Notebooks in `Example Notebooks` to learn how to run the simulation. If you are familiar with FDTD-type or similar numerical tools for acoustic simulation (such as k-Wave), then it should not be hard to start using this tool.
