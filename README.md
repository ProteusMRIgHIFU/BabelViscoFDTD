BabelViscoFDTD
=============
Samuel Pichardo, Ph.D  
Assistant Professor  
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute  
Cumming School of Medicine,  
University of Calgary   
samuel.pichardo@ucalgary.ca  
www.neurofus.ca

**Software library for FDTD of viscoelastic equation using an staggered grid arrangement and including the superposition method, with  multiple CPU- or GPU-based backends (OpenMP, CUDA, OpenCL and Metal)**

This tool solves in time domain the viscoelastic equation for wave propagation using an staggered grid solution. The solution is primarily based on a paper from Virieux (Virieux, J., 1986. P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. Geophysics, 51(4), pp.889-901.), who implemented for the first time the staggered grid solution for the viscoelastic equation.
 While the underlying equations and methods were  developed primarily for seismic simulation, the BabelViscoFDTD library was developed for biomedical applications to study the ultrasound transmission through bone material, with skull bone as primary target for study.

The name of the library comes from the fact this software may be one of the very few libraries that support all modern backends for high-performance computing including CUDA, OpenCL and Metal.

Users are invited to read first the paper associated to this work to review specifics of equations being solved in this software library, including details on boundary matching layers and averaging operators for heterogenous media.

    Pichardo, S., Moreno-Hernández, C., Drainville, R.A., Sin, V., Curiel, L. and Hynynen, K., 2017. *A viscoelastic model for the prediction of transcranial ultrasound propagation: application for the estimation of shear acoustic properties in the human skull*. Physics in Medicine & Biology, 62(17), p.6938. https://doi.org/10.1088/1361-6560/aa7ccc

For the superposition method, users can consult the corresponding paper

    Drainville, R.A., Curiel, L. and Pichardo, S., 2019. Superposition method for modelling boundaries between media in viscoelastic finite difference time domain simulations. The Journal of the Acoustical Society of America, 146(6), pp.4382-4401. https://asa.scitation.org/doi/abs/10.1121/1.5139221

If you find this software useful for your research, please consider adding a citation to the above references in your publications and presentations.

The software implementation supports multiple front-ends (Matlab and Python), OS environments (Windows, Linux, MacOS) and CPU/GPU execution. While the implementation supports CPU-based execution, a modern NVIDIA-based GPU or AMD in MacOS is highly recommended.

## Supported platforms

Please note that Python and Linux are the preferred frontend and OS. Some of the advanced tutorial notebooks need libraries that are primarily available on Linux. Below there is a table with the backends supported by each OS

| OS \ Feature | CPU single | CPU double | CUDA single | CUDA double | OpenCL single | OpenCL double | Metal single |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| Windows | Y (OpenMP) | Y (OpenMP) | Y | Y |Y | Y | N |
|Linux |  Y (OpenMP) | Y (OpenMP) | Y | Y | Y\* | Y\* | N |
| MacOS | Y (single thread) | Y (single thread)  | N | N | Y\+ | Y\+ | Y |

*Y\** Feature is enabled, but not yet fully tested. *Y\+* Feature may be limited to only 32-bits addressable memory independently of the GPU memory available.

OpenCL for Windows is operational via `pyopencl`. In Linux and MacOS, you can install pyopencl with `pip install pyopencl`. In Windows, use one of the precompiled wheels in https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl. In MacOS, an standalone OpenCL compiler (`pi_ocl`) is also included in BabelViscoFDTD. The FDTD kernels code is OpenCL >= 1.2 compliant.

### MacOS limitations
MacOS support for HPC has shifted significantly in recent years. In modern MacOS versions the support for NVIDIA cards is inexistent and OpenCL is officially being out of support beyond Big Slur. For MacOS, Metal backend is recommended. Also, OpenCL in MacOS has other limitations such as the underlying driver may only support for 32 bits memory access, even if the card has more than 4 GB of RAM. If you need to access more than 4 GB of space for your simulation, only Metal can ensure it will support it.

### Performance comparison
You can anticipate similar performance between modern AMD and NVIDIA GPUs, at least if using Metal backend. A simulation for a domain of  [237, 237, 469] grid size and over 2340 temporal steps shows the following computing times with different backends:

* Nvidia GTX A6000 (48 GB RAM, 10752 CUDA Cores, theoretical 38.7 SP TFLOP , memory bandwidth 768 GB/s) - CUDA : 90 s
* AMD Radeon Pro W6800 (32 GB RAM, 3840  stream processors, theoretical 17.83 SP TFLOP , memory bandwidth 512 GB/s)  - Metal: 56s
* M1 Max Pro  (64 GB RAM, 32 Cores, 4096 execution units (which PR material says translates into a theoretical 98304 simultaneous threads), theoretical 10.4 SP TFLOP , memory bandwidth 400 GB/s)  - Metal: 221s

The number of computing units is becoming a bit useless to compare. Anyway, there are few interesting bits:
* The ratio of performance between M1 Max Pro and A6000 is about 220% slower, so more or less following the theoretical SP throughput ratios
* The surprise was the W6800 with Metal, after using the same block thread arrangements as in the M1 it got a significant increase in performance, leaving the A6000 eating dust. The CUDA code is (in principle) optimized for maximal occupancy, but I know this may require a little more investigation to understand why the big difference.

### Metal support
Overall, Metal requires a bit more coding to prepare the pipelines for compute execution.  A challenge is that Metal for scientific computing lacks serious examples. Nevertheless, the support for Metal is desirable for Apple silicon. Once all toolchains including native Python becomes available, it will be interesting to see how well their devices stand compared to Nvidia or AMD based systems, which are still leading in performance by a significant margin. Also, there are other limitations such as maximal number of kernel parameters (32) and that each GPU buffer memory is limited to 3.5 GB RAM in AMD GPUs. But this is a limitation manageable by packing multiple logical arrays across multiple buffers. In the current release of BabelViscoFDTD, it is completely stable to run large domains with AMD GPUs and M1-based processros with 32 or more GB of RAM.

While Metal offers better performance overall over OpenCL, some issues remains. Extensive testing has indicated that the Python process frozens after running a few tens of thousands of kernel calls. For many applications, this won't be an issue, but if running very long extensive parametric studies, be aware you may need to split your exeuction in chunks that can be called in seperate `python <Myprogram.py>` calls. I suspect some driver issue limiting the number of consecutive kernels calls in a single process; I haven't yet found a mechanism to unblock/avoid this. 

### Supported platforms for Rayleigh integral
Since v0.9.2 Rayleigh integral was added a tool (see tutorial `Tutorial Notebooks\Tools -1 - Rayleigh Integral.ipynb`). This will be useful to combine models that include large volumes of water as Rayleigh integral benefits considerably of a GPU and the model is hyperparallel. The tool has support for 3 GPU backends: CUDA for Windows and Linux, and Metal and OpenCL for MacOS. 

Given the simplicity of the kernel, for the Rayleigh integral we use `pycuda` and `pyopencl` to compile the kernel directly in the Python library. For Metal, a wrapper written in Swift language is compiled during the installation. 

# Requirements
## Python 3.8 and up - x64
Use of virtual environments is highly recommended. Anaconda Python is a great choice as main environment in any OS, but overall any Python distribution should do the work. The only limitation in Windows is that wheels for latest versions of pyopencl are available for Python >=3.7 

Please note that the most advanced tutorial showing the Superposition method requires a library mainly available in Linux X64 and for Python 3.5 to 3.7. This library (`pymesh`) is constructive solid geometry (CSG) processing and is required to prepare the simulation domain. By saying this, any good CSG library that can perform intersection between meshes should do the job. If you know a more universal library that can run in any OS, please let me know via a new Github issue submission.


### Basic dependencies:
latest version of `pip`
* numpy>=1.15.1
* scipy>=1.1.0
* h5py>=2.9.0
* pydicom>=1.3.0
* setuptools >=51.0.0
* pyopencl>=2020
* pycuda>=2020 (only in Linux and Windows)

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
* pymesh == 0.3.0

All those packages (excepting `pymesh`) are installable via `pip`.  For `pymesh`, see tutorial 6 for more details.


## CUDA
The code has been verified to work from CUDA 9 to CUDA 11. Highly likely older versions of CUDA (7, 8) should work without a problem. Be sure of installing the CUDA samples and take note of the location where they were installed.


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
Any recent version of MacOS and XCode should be enough. Please note that the CPU version in MacOS does not support OpenMP (still working in a definitive solution via brew llvm or brew gcc). However, the OpenCL version works without a problem in Intel-based integrated GPUs and AMD GPUs. Metal has only been tested in AMD-based systems.

## Installation
If CUDA and supporting compiler are correctly installed, then it is straightforward to install using `pip install <directory>` or `pip3 install <directory>` depending on your installation. You need to specify the location where the CUDA samples are installed as those are required for the compilation.

Below a few examples for both Linux and Windows; the command must be run in the directory where BabelViscoFDTD was cloned (i.e. /home/<user>/Github)

Not every backend will be installed depending on your OS. For example, both Windows and Linux will install the CPU (OpenMP enabled) and CUDA backends, while MacOS will install the CPU and OpenCL backends.

### Linux
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc pip3 install  BabelViscoFDTD/
```
or
```
CUDA_SAMPLES_LOCATION=/usr/local/cuda/samples/common/inc pip3 install --user BabelViscoFDTD/
```
if you do not have write access to the global Python installation
### Windows
```
set "CUDA_SAMPLES_LOCATION=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2\common\inc" && pip install  BabelViscoFDTD\
```

### MacOS
Since CUDA is not supported anymore in MacOS, just install with:
```
pip install  BabelViscoFDTD/
```

# How to use
After installation, you can consult the Jupyter Notebooks in `Tutorial Notebooks` to learn how to run the simulation. The notebooks are ordered from basics of operation to more complex simulation scenarios, including simulation using the superposition method. If you are familiar with FDTD-type or similar numerical tools for acoustic simulation (such as k-Wave or Simsonic), then it should  be straightforward to start using this tool.

# Structure of code
The FDTD solution is accessed as a Python external function. The primary method to execute a simulation is via the class
`BabelViscoFDTD.PropagationModel` and its main function `StaggeredFDTD_3D_with_relaxation`

After installation, the class can be instatiated as:
```
from BabelViscoFDTD import PropagationModel

Model=PropagationModel()
```


## Multi-platform single code
The underlying extension code (start at `FDTDStaggered3D_with_relaxation_python.c`) uses extensively C macros to provide a fully agnostic implementation that remains as efficient as possible regardless if using a CPU or GPU backend. It supports via macro definitions compilation for Matlab and Python extensions; support for CUDA, X86_64 and OpenCL architectures; single or double precision.

Regardless if using CUDA, OpenCL or Metal, conceptually the workflow is very similar. However, the are a few implementation details that need to be handled, and the macros help a lot to reduce the coding.

Consult `setup.py` to review how all the potential modalities are generated.


# Release notes
* 0.9.4  Jan 18, 2021.
    * Cleaning some minor bugs and add BHTE code using pycuda and pyopencl.
* 0.9.3  Sep 29, 2021.
    * Improved support for both Metal and OpenCL. For Metal, stable operation is now feasible for large domains using all available memory in modern high-end GPUs. OpenCL is now supported in all OSs.
* 0.9.2  June 13, 2021.
    * Add Rayleigh integral support in homoneous medium with CUDA, OpenCL and Metal backends.
    * Support for stress sources
    * Able to select device in multiple GPU systems with CUDA
* 0.9.1  Feb 17, 2021. Pressure calculation added in low-level function.
* 0.9.0  Dec 2020. First porting from private repository and big reorganization to make it more user friendly
