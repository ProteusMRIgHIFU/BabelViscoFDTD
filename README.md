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

The software implementation supports Python as the main high-end interface, OS environments (Windows, Linux, MacOS) and CPU/GPU execution. While the implementation supports CPU-based execution, a modern NVIDIA-based GPU, or Apple-Silicon/AMD in MacOS is highly recommended.

## Supported platforms

Please note that not every backend is available in a given combination of OS+Python distribution; for example, Metal is not available under Windows, and CUDA is not available under MacOS. Some of the advanced tutorial notebooks need libraries that are primarily available on Linux. Below there is a table with the backends supported by each OS

| OS \ Feature | CPU single | CPU double | CUDA single | CUDA double | OpenCL single | OpenCL double | Metal single |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| Windows | Y (OpenMP) | Y (OpenMP) | Y | Y |Y | Y | N |
|Linux |  Y (OpenMP) | Y (OpenMP) | Y | Y | Y\* | Y\* | N |
| MacOS | Y (single thread) | Y (single thread)  | N | N | Y\+ | Y\+ | Y |

*Y\** Feature is enabled, but not yet fully tested. *Y\+* Feature may be limited to only 32-bits addressable memory independently of the GPU memory available.


# Requirements
## Python 3.8 and up 
Use of virtual environments is highly recommended. Anaconda Python and Enthought EDM are great choices as main environment in any OS, but overall any Python distribution should do the work. The only limitation in Windows is that wheels for latest versions of pyopencl are available for Python >=3.8. For Apple Silicon systems, it is highly recommended to use a native Python for arm64 (see below details).

## CUDA (For Windows and Linux)
The code has been verified to work from CUDA 9 to CUDA 11. Highly likely older versions of CUDA (7, 8) should work without a problem. Be sure of installing the CUDA samples and take note of the location where they were installed.

## CMAKE
CMAKE version 3.16.3 and up should be good. In Windows, be sure CMAKE is accessible in the Windows path.

## OpenCL
OpenCL for Windows and Apple Silicon systems is operational via `pyopencl`. In MacOS, you can install pyopencl with `pip install pyopencl`. In Windows, use one of the precompiled wheels in https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl. In MacOS for X86-64 systems, an standalone OpenCL compiler (`pi_ocl`) is also included in BabelViscoFDTD. The FDTD kernels code is OpenCL >= 1.2 compliant.

### Basic Python dependencies:
latest version of `pip`
* numpy>=1.15.1 (have this already installed if starting from a clean environment)
* scipy>=1.1.0 (have this already installed if starting from a clean environment)
* h5py>=2.9.0 (in native arm64 Python for Apple Silicon, install this via `conda install h5py`)
* hdf5plugin>=3.2.0
* pydicom>=1.3.0
* setuptools >=51.0.0
* pyopencl>=2020 (if in Windows, install manually a wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl)
* pycuda>=2020 (only in Linux and Windows)
* mkl (obligatory for OpenMP support in X86-64 Apple systems; mkl is not available for arm6

### Extra dependencies required in some of the tutorials
* matplotlib
* jupyter
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

## Linux
Overall, any LTS-type distribution is recommended to be sure CUDA compiler supports your default GCC installation. If your installation can run the default examples of CUDA, then you should be good.

Ubuntu 20.04 LTS is an excellent candidate to start with.

You will also need to install OpenCL headers and libraries such as `opencl-headers`, `ocl-icd-opencl-dev`, `intel-opencl-icd` and other libraries required by you GPU manufacturer to support OpenCL. You can verify you have a healthy OpenCL installation with the tool `clinfo`.

### Linux in Windows through WSL2
Starting in 2020, support for CUDA execution directly in WSL2 became possible. I highly recommended you give it a try as I have had excellent experiences with it. Just follow the official instructions from NVIDIA (https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Please note that OpenCL is not supported under WSL2. You will still need to install the packages  `opencl-headers` and `ocl-icd-opencl-dev` to ensure all compilation is completed.

## Windows native
You will need a VStudio installation that is compatible with your CUDA version; for example, CUDA 11.2 supports officially VS 2015 to 2019.

## MacOS
Any recent version of MacOS and XCode with the command-line tools should be enough. Most tests have been done in Big Slur and Monterey. The CPU version in MacOS supports OpenMP both in X86-64 and M1 processors. The OpenCL and Metal backed have been tested in Intel-based integrated GPUs, AMD GPUs and  M1-based systems. There is, however, some limitations of AMD GPUs with OpenCL (see below MacOS notes for more details).

Best scenario for M1-based systems is to use a fully native Python distribution. You can see details how to do this using homebrew; follow step 2 at https://towardsdatascience.com/how-to-easily-set-up-python-on-any-m1-mac-5ea885b73fab 

For X86-64, `libomp` and `mkl` must be installed first, otherwise compilation for OpenMP support will fail.
* Install `libomp` with `brew install libomp`  
* Install `mkl` with `pip install mkl` or `conda install mkl`

Metal backend will be available for both X86-64 and Apple Silicon systems.  

# Installation
BabelViscoFDTD is available via `pip`
```
 pip install BabelViscoFDTD
```

If you prefer trying experimental versions, you can clone https://github.com/ProteusMRIgHIFU/BabelViscoFDTD.git and install with:

```
pip install BabelViscoFDTD/
```
run in the parent directory where you cloned the repository.
# How to use
After installation, please consult the Jupyter Notebooks in `Tutorial Notebooks` in the repository https://github.com/ProteusMRIgHIFU/BabelViscoFDTD to learn how to run the simulation and get familiar with the input parameters The notebooks are ordered from basics of operation to more complex simulation scenarios, including simulation using the superposition method. If you are familiar with FDTD-type or similar numerical tools for acoustic simulation (such as k-Wave or Simsonic), then it should  be straightforward to start using this tool.

# Structure of code
The FDTD solution is accessed as a Python external function. The primary method to execute a simulation is via the class
`BabelViscoFDTD.PropagationModel` and its main function `StaggeredFDTD_3D_with_relaxation`

After installation, the class can be instatiated as:
```
from BabelViscoFDTD import PropagationModel

Model=PropagationModel()
```

### Multi-platform single code
The underlying extension code (start at `FDTDStaggered3D_with_relaxation_python.c`) uses extensively C macros to provide a fully agnostic implementation that remains as efficient as possible regardless if using a CPU or GPU backend. It supports via macro definitions compilation for native CPU (X86, arm64), CUDA, OpenCL and Metal architectures; single or double precision.

Regardless if using CUDA, OpenCL or Metal, conceptually the workflow is very similar. However, there are a few implementation details that need to be handled, and the macros help a lot to reduce the coding.

Consult `setup.py` to review how all the potential modalities are generated.

# Important information specific to the different environments for use
### MacOS notes
MacOS support for HPC has shifted in recent years significantly. In modern MacOS versions the support for NVIDIA cards is inexistent and OpenCL *was supposed to be* officially out of support beyond Big Slur (*it is still running quite well in Monterey*). For MacOS, Metal backend is recommended. Also, OpenCL in MacOS may have other limitations such as the underlying driver may only support 32 bits memory access, even if the card has more than 4 GB of RAM. However, this limitation seems to be case by case. For example, for M1 processors, OpenCL drivers can support 64 bits addressing, but for an AMD W6800 (32 GB RAM) it only supports 32 bits. Using `clinfo` tool (available with homebrew) can provide details if your current GPU and drivers supports 64 bits addressing. If you need to access more than 4 GB of space for your simulation, only Metal can support it without problems. 

**Important**: The OpenCL implementation with M1 processors seems to work only with the native Python arm64 installation mentioned above. 

### Metal support
Overall, Metal requires a bit more coding to prepare the pipelines for compute execution.  A challenge is that Metal for scientific computing still lacks serious examples. Nevertheless, the support for Metal is desirable for Apple Silicon systems. As toolchains, including native Python in arm64, are becoming available, it is interesting to see how well their devices stand compared to Nvidia or AMD based systems. Also, there are other limitations such as maximal number of kernel parameters (32) and that each GPU buffer memory is limited to 3.5 GB RAM in AMD GPUs. But this is a limitation manageable by packing multiple logical arrays across multiple buffers. In the current release of BabelViscoFDTD, it is completely stable to run large domains with AMD GPUs and M1-based processors with 32 or more GB of RAM.

While Metal offers better performance overall over OpenCL in AMD processors, some issues remains. Extensive testing has indicated that the Python process freezes after running a few tens of thousands of kernel calls. For many applications, this won't be an issue, but if running very long extensive parametric studies, be aware you may need to split your execution in chunks that can be called in separate `python <Myprogram.py>` calls. I suspect some driver issue limiting the number of consecutive kernels calls in a single process.

## Single precision performance comparison
Performance between modern AMD, NVIDIA and Apple Silicon GPUs can show important differences, especially when comparing Metal and OpenCL backends. A simulation for a domain of  [249,249,426] grid size and over 2262 temporal steps was used to benchmark multiple backends and systems.

* Nvidia RTX A6000 (48 GB RAM, 10752 CUDA Cores, theoretical 38.7 SP TFLOP , memory bandwidth 768 GB/s)
* AMD Radeon Pro W6800 (32 GB RAM, 3840  stream processors, theoretical 17.83 SP TFLOP , memory bandwidth 512 GB/s) 
* AMD Vega 56 (8 GB RAM, 3584  stream processors, theoretical 10.5 SP TFLOP , memory bandwidth  410 GB/s) 
* M1 Max  (64 GB RAM, 10 CPU cores, 32 GPU Cores, 4096 execution units (which PR material says translates into a theoretical 98304 simultaneous threads), theoretical 10.4 SP TFLOP , memory bandwidth 400 GB/s)

RTX A6000 test was done in 128 GB Xeon W-2125 CPU (4x2 cores) @ 4.00GHz Dell system. AMD Vega 64 and AMD Radeon Pro W6800 were tested in an 128 GB iMac Pro system (10x2 Core Intel Xeon W). The Vega 64 GPU is part of the iMac system, while the Pro W6800 is connected via a Thunderbolt 3 external enclosure. Please note that GPU connectivity should not have an important effect given memory transfers between GPU and CPU are minimal. The M1 Max was configured with 64 GB and installed in a 2021 MacBook Pro system. The Dell system, iMac Pro and MacBook Pro were also used for OpenMP benchmarks. Python 3.9 was used in the Dell and MacBook Pro systems, while Python 3.8 was used in the iMac Pro. CUDA code was compiled with CUDA 11.2 and VStudio 2019 under Windows 11. MacOS Monterey 12.1 with XCode 13.1 were used for both iMac Pro and MacBook Pro. Pyopencl 2021.2 was used as the OpenCL wrapper for the tests for the A6000 and M1 Max. The mini wrapper `pi_ocl` mentioned above was used for the W6800 and Vega 64 OpenCL tests. 

Wall-time was measured from the moment preparations to run GPU code started (initiate device operation) to the end of the memory transfer of results, with no access to main drives involved. Memory transfer between CPU and GPU occurred only at the beginning and end of GPU execution. Numerical difference among difference backends were in the order of single precision resolution.

### Summary of wall-time results for each device
| Device |  CUDA single | OpenCL single |  Metal single | OpenMP single|
| --- | --- |  --- |  --- |  ---  |
| AMD W6800 | - | 45 s | 37 s | - |
| AMD Vega 56 | - | 90 s | 83 s | - |
| NVidia A6000 | 53 s| 77 s | -| - |
| M1 Max | - |  57 s |152 s| 790 s (10 threads) |
| Xeon W-2125 | - | - | - | 2202 s (8 threads)|
| iMac Pro (Xeon W) | - | - | - | 1649 s  (20 threads)|

#### Discussion of results
The number of computing units is becoming a bit useless to compare. There are few interesting bits:
* The ratio of performance between M1 Max and A6000 (CUDA vs. Metal) is about 230% slower, which is close to the theoretical difference of raw SP TFLOPS performance of 180%.
* *BUT*, the OpenCL performance of the M1 Max is dramatically better than Metal, matching the A6000 performance, way far from the theoretical difference of raw SP TFLOPS that was expected. 
* Tests with larger domains indicated that the M1 Max performs even better. For example, for a domain size of [315, 315, 523] and 2808 temporal steps, the M1 Max showed a wall-time of 131 s while the A6000 took 139 s. 
* Multiple tryouts on the CUDA code to adjust grid and block sizes didn't improve performance in the A6000. On the contrary, wall-time was increased, indicating that the recommended method by NVidia to calculate maximal occupancy used by default in BabelViscoFDTD provided the best performance with the A6000.
* The other surprise was the W6800 with Metal and OpenCL that outperformed by a significant margin the A6000. 
* Contrary to the W6800, the Vega 56 GPU showed worse performance when using Metal, similar as for the M1 Max.
* The fact that Metal shows better performance than OpenCL in the W6800 compared to Apple Silicon is also surprising.
* The OpenMP performance of the M1 Max is simply excellent, showing a dramatic speedup compared to the Dell Xeon and iMac Pro systems. Highly likely the tight integration of the CPU to the memory bank in the M1 system may play a significant role.

## Possibility of manual adjustments to improve performance
All three GPU backends have analogous control to split the calculations in the GPU multiprocessors. BabelViscoFDTD uses the methods that are recommended for each backend to ensure maximal GPU occupancy. However, manual adjustments can provide improvement to the performance. You can specify manually the grid and thread block dimensions with the optional parameters `ManualGroupSize` and `ManualLocalSize`, respectively. 
 
 Please consult guidelines of each backend (CUDA, OpenCL and Metal) on how to calculate this correctly, otherwise there is a risk of specifying a too large or too short grid and thread size dimensions. For example, for both CUDA and Metal, the multiplication of `ManualGroupSize` and `ManualLocalSize` must be equal or larger than the domain size ([N1,N2,N3]) to ensure all the domain is covered; for example for a domain of size [231,220,450], `ManualGroupSize=[60,60,115]` with `ManualLocalSize=[4,4,4]` will ensure covering the domain size. For `OpenCL` each entry in `ManualGroupSize` must be equal or larger than [N1,N2,N3] and each entry must be a multiple of its corresponding entry in `ManualLocalSize`; for example for a domain of size [231,220,450], `ManualGroupSize=[240,240,460]` with `ManualLocalSize=[4,4,4]`. Be sure of specifying these parameters as an np.array of type np.int32, such as `ManualLocalSize=np.array([4,4,4]).astype(np.int32)`. 

# Supported platforms for Rayleigh integral
Since v0.9.2 Rayleigh-Somerfeld integral was added a tool (see tutorial `Tutorial Notebooks\Tools -1 - Rayleigh Integral.ipynb`). This will be useful to combine models that include large volumes of water as Rayleigh integral benefits considerably of a GPU and the model is hyperparallel. The tool has support for 3 GPU backends: CUDA for Windows and Linux, and Metal and OpenCL for MacOS. 

Given the simplicity of the kernel, for the Rayleigh integral we use `pycuda` and `pyopencl` to compile the kernel directly in the Python library. For Metal, a wrapper written in Swift language is compiled during the installation. 


# Release notes
* 0.9.6  Feb 5, 2022.
    * Improved performance in Metal-based devices by using mini-kernels.
    * Minor fixes for Rayleigh-Somerfeld in Metal
* 0.9.5  Jan 18, 2022.
    * Very first pip-registered version
    * Possibility of user-specified dimensions of blocks for computations for fine-tuning of performance
    * Cleaning some minor bugs and add BHTE code using pycuda and pyopencl.
    
* 0.9.3  Sep 29, 2021.
    * Improved support for both Metal and OpenCL. For Metal, stable operation is now feasible for large domains using all available memory in modern high-end GPUs. OpenCL is now supported in all OSs.
* 0.9.2  June 13, 2021.
    * Add Rayleigh integral support in homoneous medium with CUDA, OpenCL and Metal backends.
    * Support for stress sources
    * Able to select device in multiple GPU systems with CUDA
* 0.9.1  Feb 17, 2021. Pressure calculation added in low-level function.
* 0.9.0  Dec 2020. First porting from private repository and big reorganization to make it more user friendly
