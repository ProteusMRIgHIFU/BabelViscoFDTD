SPPVirieuxFDTD
=============
Samuel Pichardo, Ph.D  
Assistant Professor  
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute  
Cumming School of Medicine,  
University of Calgary   
samuel.pichardo@ucalgary.ca  
www.neurofus.ca

**Software library for FDTD of viscoelastic equation using an staggered grid arrangement and including the superposition method, with  CPU- or GPU-based backends (OpenMP, CUDA and OpenCL)**

This tool solves in time domain the viscoelastic equation for wave propagation using an staggered grid solution. While the underlying equations and methods were  developed primarily for seismic simulation, the SPPVirieuxFDTD library was developed for biomedical applications to study the ultrasound transmission through bone material, with skull bone as primary target for study.

The name of the library comes from **S**uper**P**osition, **Virieux** (who is the author from the 60s who implemented for the first time the staggered grid solution for the viscoelastic equation) and **FDTD** for Finite-Difference Time-Difference.

Users are invited to read first the paper associated to this work to review specifics of equations being solved in this software library, including details on boundary matching layers and averaging operators for heterogenous media.

    Pichardo, S., Moreno-Hernández, C., Drainville, R.A., Sin, V., Curiel, L. and Hynynen, K., 2017. *A viscoelastic model for the prediction of transcranial ultrasound propagation: application for the estimation of shear acoustic properties in the human skull*. Physics in Medicine & Biology, 62(17), p.6938. https://doi.org/10.1088/1361-6560/aa7ccc

For the superposition method, users can consult the corresponding paper

    Drainville, R.A., Curiel, L. and Pichardo, S., 2019. Superposition method for modelling boundaries between media in viscoelastic finite difference time domain simulations. The Journal of the Acoustical Society of America, 146(6), pp.4382-4401. https://asa.scitation.org/doi/abs/10.1121/1.5139221

If you find this software useful for your research, please consider adding a citation to the above references in your publications and presentations.

The software implementation supports multiple front-ends (Matlab and Python), OS environments (Windows, Linux, MacOS) and CPU/GPU execution. While the implementation supports CPU-based execution, a modern NVIDIA-based GPU or AMD in MacOS is highly recommended.

Please note that Python and Linux are the preferred frontend and OS. Some of the advanced tutorial notebooks need libraries that are primarily available on Linux.

MacOS support for HPC has shifted significantly as since several MacOS versions the support for NVIDIA cards is practically inexistent and OpenCL is officially being out of support beyond Big Slur. Nevertheless, OpenCL in MacOS still gives excellent performance. Back in 2017,  an OpenCL implementation with an AMD Vega56 outperformed an NVIDIA Titan XP via CUDA (31 s vs 47s, for a simulation with a domain size of 158$\times$154$\times$111 and 5800 time steps). In 2020, a RTX 2080 SUPER via CUDA can run a given simulation in around 5s for a domain 118\times$118\times$211 and 817 time steps, while a much simpler Radeon Pro 560 (available in a MacBook Pro 2017) takes 18 s, and an i7-9700 (4 cores x 2 with hyperthreading) via OpenMP takes 35s.

Future improvements to the library will include support to Metal, as it shares a lot of similitudes with OpenCL and the underlying code of SPPVirieuxFDTD should be easily adapted to  Metal.


# Requirements
## Python 3.6 and up - x64
The code should work with Python 3.6 and up, including Anaconda, EDM Canopy (both supported in all 3 main OSs) or Linux-based distribution Python installation; same for brew-based installation for MacOS.

Use of virtual environments is highly recommended

### Basic dependencies:
latest version of `pip`
* numpy>=1.15.1
* scipy>=1.1.0
* h5py>=2.9.0
* pydicom>=1.3.0
* setuptools >=51.0.0

### Extra dependencies required in some of the tutorials
* scikit-image >= 0.17
* pyvista >= 0.27.0
* pyvistaqt >= 0.2.0
* mayavi >= 4.7.0
* itkwidgets >= 0.32.0
* jupyter >= 1.0.0
* ipywidgets >= 1.0.0
* PySide2 >= 5.14.0
* pymp-pypi >= 0.4.3

All those packages are installable via `pip`


## CUDA
The code has been verified to work from CUDA 9 to CUDA 11. Highly likely older versions of CUDA (7, 8) should work without a problem. Be sure of installing the CUDA samples and take note of the location where they were installed.

Please note CUDA is not supported anymore in MacOS as NVIDA and Apple seem to be in a feud since years.
## CMAKE
CMAKE version 3.16.3 and up should be good
## Linux
Overall, any LTS-type distribution is recommended to be sure CUDA compiler supports your default GCC installation. If your installation can run the default examples of CUDA, then you should be good.

Ubuntu 20.04 LTS is an excellent candidate to start with.

## Linux in Windows through WSL2
Starting in 2020, support for CUDA execution directly in WSL2 became possible. I highly recommended you give it a try as I have had excellent experiences with it. Just follow the official instructions from NVIDIA (https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
## Windows native
You will need a VStudio installation that is compatible with your CUDA version (for example, CUDA 11.2 supports officially VS 2015 to 2019), and be sure of installing the CUDA samples and take note of the location where they were installed.

When installing CMAKE, be sure it is accessible in the Windows path.

## MacOS
Any recent version of MacOS and XCode should be enough. Please note that the CPU version in MacOS does not support OpenMP (still working in a definitive solution via brew llvm or brew gcc). However, the OpenCL version works without a problem in Intel-based integrated GPUs and AMD GPUs.

## Installation
If CUDA and supporting compiler are correctly installed, then it is straightforward to install using `pip install <directory>` or `pip3 install <directory>` depending on your installation. You need to specify the location where the CUDA samples are installed as those are required for the compilation.

Below a few examples for both Linux and Windows; the command must be run in the directory where SPPVirieuxFDTD was cloned (i.e. /home/<user>/Github)

Not every backend will be installed depending on your OS. For example, both Windows and Linux will install the CPU (OpenMP enabled) and CUDA backends, while MacOS will install the CPU and OpenCL backends.

### Linux
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc pip3 install  SPPVirieuxFDTD/
```
or
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc pip3 install --user SPPVirieuxFDTD/
```
if you do not have write access to the global Python installation
### Windows
```
set "CUDA_SAMPLES_LOCATION=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2\common\inc" && pip install  SPPVirieuxFDTD\
```

### MacOS
Since CUDA is not supported anymore in MacOS, just install with:
```
pip install  SPPVirieuxFDTD\
```
#### OpenCL backend
The OpenCL backend is a bit convoluted as the library needs to compile on-the-flight the GPU code. This cannot be avoided as the OpenCL accelerated code is driver-specific and cannot be generated as in CUDA in advance for all the possible hardware variants that OpenCL supports. *It is not as bad as it sounds*, but you need to compile manually a little supplementary program and copy it to the location where your simulation is being executed. The mentioned code is at `SPPVirieuxFDTD/pi_ocl`. Just open a terminal at that location and compile the program with `make`. It will generate a small program called `pi_ocl`. Copy that program to the location where your simulation will be run. For example, if you want to run the examples in a MacOS system using OpenCL, just copy `pi_ocl` to the `Example Notebooks` directory.


# Structure of code
The FDTD solution is accessed as a Python external function. The primary method to execute a simulation is via the class
`SPPVirieuxFDTD.PropagationModel` and its main function `StaggeredFDTD_3D_with_relaxation`

After installation, the class can be instatiated as:
```
from SPPVirieuxFDTD import PropagationModel

Model=PropagationModel()
```


## Multi-platform single code
The underlying extension code (start at `SPVirieuxFDTD_python.c`) uses extensively C macros to provide a fully agnostic implementation that remains as efficient as possible regardless if using a CPU or GPU. It supports via macro definitions compilation for Matlab and Python extensions; support for CUDA, X86_64 and OpenCL architectures; single or double precision.

Consult `setup.py` and `CompileMatlab.m` to review how all the potential modalities are generated.

Please note that the Matlab implementation is still missing an updated high-level equivalence to `SPPVirieuxFDTD\PropagationModel.py`. Given most of my personal computing platform moved years ago to Python, the Matlab frontend is a low-priority by the time being. However, the compilation for Matlab frontend is still operational.

# How to use
After installation, you can consult the Jupyter Notebooks in `Tutorial Notebooks` to learn how to run the simulation. The notebooks are ordered from basics of operation to more complex simulation scenarios, including simulation using the superposition method. If you are familiar with FDTD-type or similar numerical tools for acoustic simulation (such as k-Wave or Simsonic), then it should  be straightforward to start using this tool.
