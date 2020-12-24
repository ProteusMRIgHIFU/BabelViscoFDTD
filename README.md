StaggaredFDTD
=============
Samuel Pichardo, Ph.D  
Assistant Professor  
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute  
Cumming School of Medicine,  
University of Calgary   
samuel.pichardo@ucalgary.ca  
www.neurofus.ca

**Software library for FDTD of viscoelastic equation using an staggered grid arrangement**

This tool solves in time domain the viscoelastic equation for wave propagation using an staggered grid solution.

Users are invited to read first the paper associated to this work to review specifics of the solution (including details on boundary matching layers and averaging operators for heterogenous media):

  Pichardo, S., Moreno-Hernández, C., Drainville, R.A., Sin, V., Curiel, L. and Hynynen, K., 2017. *A viscoelastic model for the prediction of transcranial ultrasound propagation: application for the estimation of shear acoustic properties in the human skull*. Physics in Medicine & Biology, 62(17), p.6938. https://doi.org/10.1088/1361-6560/aa7ccc

If you find this software useful for your research, please consider adding a citation to the above reference in your publications and presentations.

The software implementation supports multiple front-ends (Matlab and Python), OS environments (Windows, Linux, MacOS*) and CPU/GPU execution. While the implementation supports CPU-based execution, a modern NVIDIA-based GPU is highly recommended.

Please note that Python and Linux are the preferred frontend and OS. MacOS support has degraded significantly as since several MacOS versions the support for NVIDIA cards is practically inexistent and OpenCL is officially being out of support beyond Big Slur. As a side note, back in 2017, both external NVIDIA GPUs and OpenCL were better supported, with the OpenCL implementation with a Vega64 showing some nice head-to-head performance with a GTX 970 via external enclosure. Just to say it is a pity HPC computing has been pretty much abandoned by MacOS.

# Requirements
## Python 3.6 and up, x64
The code should work with Python 3.6 and up, including Anaconda, EDM Canopy or Linux-based distribution Python installation

Use of virtual environments is highly recommended

Dependencies:
* numpy>=1.15.1
* scipy>=1.1.0
* h5py>=2.9.0
* pydicom>=1.3.0


## CUDA
The code has been verified to work from CUDA 9 to CUDA 11. Highly likely older versions of CUDA (7, 8) should work without a problem. Be sure of installing the CUDA samples and take note of the location where they were installed.
## CMAKE
CMAKE version 3.16.3 and up should be good
## Linux
Overall, any LTS-type distribution is recommended to be sure CUDA compiler supports your default GCC installation. If your installation can run the default examples of CUDA, then you should be good.
## Linux in Windows through WSL2
Starting in 2020, support for CUDA execution directly in WSL2 became possible. I highly recommended you give it a try as I have had good experiences with it. Just follow the official instructions from NVIDIA (https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
## Windows native
Just be sure of having a VStudio installation that is compatible with your CUDA version (for example, CUDA 11.2 supports officially VS 2015 to 2019), and be sure of installing the CUDA samples and take note of the location where they were installed.

## Installation
If CUDA and supporting compiler are correctly installed, then it is straightforward to install. You need to specify the location where the CUDA samples are installed as those are required for the compilation.

Below a few examples for both Linux and Windows
### Linux
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc python3 setup.py  install
```
or
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc python3 setup.py  install --user
```
if you do not have write access to the global Python installation
### Windows
```
set "CUDA_SAMPLES_LOCATION=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2\common\inc" && python setup.py install
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
