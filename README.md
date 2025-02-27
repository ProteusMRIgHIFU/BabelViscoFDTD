BabelViscoFDTD
=============
Samuel Pichardo, Ph.D
Associate Professor
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute
Cumming School of Medicine,
University of Calgary
samuel.pichardo@ucalgary.ca
www.neurofus.ca

**Software library for FDTD of the viscoelastic equation using a staggered grid arrangement and including the superposition method, with multiple CPU- and GPU-based backends (OpenMP, CUDA, OpenCL and Metal)**

This tool solves in the time domain the viscoelastic equation for wave propagation using a staggered grid solution. The solution is primarily based on a paper from Virieux (Virieux, J., 1986. P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. Geophysics, 51(4), pp.889-901.), who implemented for the first time the staggered grid solution for the viscoelastic equation.
 While the underlying equations and methods were developed primarily for seismic simulation, the BabelViscoFDTD library was developed for biomedical applications to study the ultrasound transmission through bone material, with skull bone as the primary target for study.

The name of the library comes from the fact this software may be one of the very few libraries that support all modern backends for high-performance computing including CUDA, OpenCL and Metal.

Users are invited to read first the paper associated with this work to review specifics of equations being solved in this software library, including details on boundary matching layers and averaging operators for heterogeneous media.

    Pichardo, S., Moreno-Hernández, C., Drainville, R.A., Sin, V., Curiel, L. and Hynynen, K., 2017. *A viscoelastic model for the prediction of transcranial ultrasound propagation: application for the estimation of shear acoustic properties in the human skull*. Physics in Medicine & Biology, 62(17), p.6938. https://doi.org/10.1088/1361-6560/aa7ccc

For the superposition method, users can consult the corresponding paper

    Drainville, R.A., Curiel, L. and Pichardo, S., 2019. Superposition method for modelling boundaries between media in viscoelastic finite difference time domain simulations. The Journal of the Acoustical Society of America, 146(6), pp.4382-4401. https://asa.scitation.org/doi/abs/10.1121/1.5139221

If you find this software useful for your research, please consider adding a citation to the above references in your publications and presentations.

The software implementation supports Python as the main high-end interface, OS environments (Windows, Linux, macOS) and CPU/GPU execution. While the implementation supports CPU-based execution, a modern NVIDIA-based GPU, or Apple-Silicon/AMD in macOS is highly recommended.

## Supported platforms

Please note that not every backend is available in a given combination of OS+Python distribution; for example, Metal is not available under Windows, and CUDA is not available under macOS. Some of the advanced tutorial notebooks need libraries that are primarily available on Linux. Below there is a table with the backends supported by each OS

| OS \ Feature | CPU single | CPU double | CUDA single | CUDA double | OpenCL single | OpenCL double | Metal single |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| Windows | Y (OpenMP) | Y (OpenMP) | Y | Y |Y | Y | N |
|Linux |  Y (OpenMP) | Y (OpenMP) | Y | Y | Y\* | Y\* | N |
| macOS | Y (OpenMP\*\*) | Y (OpenMP\*\*)  | N | N | Y\+ | Y\+ | Y |

*Y\** Feature is enabled, but not yet fully tested. *Y\+* In macOS in X64 Intel systems this feature may be limited to only 32-bits addressable memory independently of the GPU memory available.\*\* OpenMP is enabled by default in Apple ARM64 processors. In macOS X64 systems the installation defaults to single thread, OpenMP can be enabled manually as an experimental feature (read below).


# Requirements
## Python 3.8 and up
The use of virtual environments is recommended. Anaconda Python and Enthought EDM are great choices as the main environment in any OS, but overall any Python distribution should do the work. The only limitation in Windows is that wheels for the latest versions of pyopencl are available for Python >=3.8. For Apple Silicon systems, it is recommended to use a native Python for arm64 (see below details).

## CUDA (For Windows and Linux)
The code has been verified to work from CUDA 9 to CUDA 11. Highly likely older versions of CUDA (7, 8) should work without a problem. Be sure of installing the CUDA samples and take note of the location where they were installed.

## CMAKE
CMAKE version>= 3.16.3.

## OpenCL
OpenCL for Windows and Apple Silicon systems is operational via `pyopencl`. In macOS, you can install pyopencl with `pip install pyopencl`. In Windows, use one of the precompiled wheels in https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl. The FDTD kernels code is OpenCL >= 1.2 compliant.

### Basic Python dependencies:
latest version of `pip`
* numpy>=1.15.1 (have this already installed if starting from a clean environment)
* scipy>=1.1.0 (have this already installed if starting from a clean environment)
* h5py>=2.9.0 (in native arm64 Python for Apple Silicon, install this via `conda install h5py`)
* hdf5plugin>=3.2.0
* pydicom>=1.3.0
* setuptools >=51.0.0
* pyopencl>=2020 (if in Windows, install manually a wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl)
* cupy-cuda11x in Linux (via pip). In Windows install with conda in windows with `conda install -c conda-forge cupy`

`h5py`, `hdf5plugin`, `pydicom` and `pyopencl` are installed automatically as requirements if they are no present in the Python enviroment..

### macOS systems: Manual installation of modified `metalcompute`
As noted in the release notes below for v0.9.9.20, we use a modified version of `py-metal-compute`. To avoid confusing with the original library, the modified version needs to be installed manually with

`pip install  git+https://github.com/ProteusMRIgHIFU/py-metal-compute.git`

This modified version will be installed with a different library name (metalcomputebabel) that is different from the original (metalcompute) to avoid conflicts.

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

# Installation
BabelViscoFDTD is available via `pip`
```
 pip install BabelViscoFDTD
```

If you prefer trying experimental versions, you can clone https://github.com/ProteusMRIgHIFU/BabelViscoFDTD.git and install within the cloned directory:

```
pip install .
```
run in the parent directory where you cloned the repository.

## Specific OS notes
### Linux
Overall, any LTS-type distribution is recommended to be sure CUDA compiler supports your default GCC installation. If your installation can run the default examples of CUDA, then you should be good.

You may also need to install OpenCL headers and libraries such as `opencl-headers`, `ocl-icd-opencl-dev`, `intel-opencl-icd` and other libraries required by you GPU manufacturer to support OpenCL. You can verify you have a healthy OpenCL installation with the tool `clinfo`.

#### Linux in Windows through WSL2
Starting in 2020, support for CUDA execution directly in WSL2 became possible. We have had excellent experiences with it. Just follow the official instructions from NVIDIA (https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Please note that OpenCL is not supported under WSL2.

### Windows native
You will need a VStudio installation that is compatible with your CUDA version; for example, CUDA 11.2 supports officially VS 2015 to 2019.

### macOS
Any recent version of macOS and XCode with the command-line tools should be enough. Most tests have been done in Big Slur and Monterey. The CPU version in macOS supports OpenMP in ARM64 processors (M1, M1 Max, M2 ultra, M2). In X86-64, the OpenMP feature is now turned as experimental; by default, it will run only single-thread. See below for details on how to enable it. For ARM64 version will have OpenMP fully enabled by default.

The OpenCL and Metal backed have been tested in Intel-based integrated GPUs, AMD GPUs and  ARM64-based systems. There are, however, some limitations of AMD GPUs with OpenCL (see below macOS notes for more details). Metal backend is available for both X86-64 and Apple Silicon systems. Overall, Metal is the recommended backend for M1 processors and AMD GPUs.

Best scenario for both X64 and ARM64-based systems is to use fully native Python distributions using homebrew:
1. Install  [homebrew](https://brew.sh/)
`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
1. Install miniforge
`brew install miniforge`
1. Create and activate a Python environment (for ARM64, Python 3.9 is recommended)
`conda create —-name Babel python=3.9 scipy numpy`
`conda activate Babel`
1. Install BabelViscoFDTD
`pip install BabelViscoFDTD`

#### macOS experimental OpenMP in X86-64

Thanks to the people who have reported the mitigated success with OpenMP. In my current systems, I didn't have problems, but later some users reported having issues with OpenMP. Then, we turned this feature as **experimental**, the time we can figure out a more stable approach.
For Intel or Apple Silicon CPU, `libomp` and `mkl` must be installed first, otherwise compilation and execution for OpenMP will fail.
Install `mkl` with `pip install mkl` or `conda install mkl`

Using Homebrew, install OpenMP and hint location to CMake:

```sh
brew install libomp

export OpenMP_ROOT=/opt/homebrew/opt/libomp
```


To enable the OpenMP version for macOS install BabelViscoFDTD with:
```
BABEL_MAC_OPENMP_X64=1 pip install BabelViscoFDTD
```
or if you cloned the repository
```
BABEL_MAC_OPENMP_X64=1 pip install BabelViscoFDTD/
```

# How to use
After installation, please consult the Jupyter Notebooks in `Tutorial Notebooks` in the repository https://github.com/ProteusMRIgHIFU/BabelViscoFDTD to learn how to run the simulation and get familiar with the input parameters The notebooks are ordered from basics of operation to more complex simulation scenarios, including simulation using the superposition method. If you are familiar with FDTD-type or similar numerical tools for acoustic simulation (such as k-Wave or Simsonic), then it should be straightforward to start using this tool.

# Structure of code
The FDTD solution is accessed as a Python external function. The primary method to execute a simulation is via the class
`BabelViscoFDTD.PropagationModel` and its main function `StaggeredFDTD_3D_with_relaxation`

After installation, the class can be initiated as:
```
from BabelViscoFDTD import PropagationModel

Model=PropagationModel()
```

### Multi-platform single code
The underlying GPU code (start at `StaggeredFDTD_3D_With_Relaxation_<xx>.py` files) uses extensively C macros to provide a fully agnostic implementation that remains as efficient as possible regardless of using a CPU or GPU backend. It supports via macro definitions compilation for native CPU (X86, arm64), CUDA, OpenCL and Metal architectures; single or double precision.

Regardless if using CUDA, OpenCL or Metal, conceptually the workflow is very similar. However, there are a few implementation details that need to be handled. cupy, pyopencl and metalcompute libraries help to minimize the amount the coding while still providing best performance possible with each backend.

# Important information specific to the different environments for use
### macOS notes
macOS support for HPC has shifted in recent years significantly. In modern macOS versions, the support for NVIDIA cards is inexistent and OpenCL *was supposed to be* officially out of support beyond Big Slur (*it is still running quite well in Monterey*). For macOS, Metal backend is recommended for AMD processors. OpenCL in macOS X86_64 may have other limitations such as the underlying driver may only support 32 bits memory access, even if the card has more than 4 GB of RAM. However, this limitation seems to be case by case. For ARM64 processors, OpenCL drivers can support 64 bit addressing. For an AMD W6800 (32 GB RAM) it only supports 32 bits. The `clinfo` tool (available with homebrew) can provide details if your current GPU and drivers support 64 bits addressing. The OpenCL implementation with ARM64 processors works only with a native Python arm64 installation.

### Metal support
Overall, Metal requires a bit more coding to prepare the pipelines for compute execution.  A challenge is that Metal for scientific computing still lacks serious examples. Nevertheless, the support for Metal is desirable for Apple Silicon systems. As toolchains, including native Python in arm64, are now becoming widespread available, it is interesting to see how well these devices stand quite well compared to Nvidia or AMD-based systems. There are some limitations such as the maximal number of kernel parameters (32) and that each GPU buffer memory is limited to 3.5 GB RAM for Metal-supported GPUs. But this is a limitation manageable by packing multiple logical arrays across multiple buffers. We explored the use of Metal Argument Buffers, but it ended in poorer performance than packing multiple arrays logically. In the current release of BabelViscoFDTD, it is completely stable to run large domains with AMD GPUs and ARM64-based processors with 32 or more GB of RAM.

While Metal offers better performance overall over OpenCL in both Apple and AMD processors, some issues remain. Extensive testing has indicated that the Python process freezes after running a few tens of thousands of kernel calls. For most of the applications, this won't be an issue. If running very long extensive parametric studies under a single execution, be aware you may need to split your execution into chunks that can be called in separate `python <Myprogram.py>` calls. I suspect some driver issue limiting the number of consecutive kernels calls in a single process.

## Single precision performance comparison
Performance between modern AMD, NVIDIA and Apple Silicon GPUs can show important differences. A simulation for a domain of  [414, 219 , 375] grid size and over 2841 temporal steps was used to benchmark multiple backends and systems.

* Nvidia RTX A6000 (48 GB RAM, 10752 CUDA Cores, theoretical 38.7 SP TFLOP , memory bandwidth 768 GB/s)
* AMD Radeon Pro W6800 (32 GB RAM, 3840  stream processors, theoretical 17.83 SP TFLOP, memory bandwidth 512 GB/s)
* AMD Vega 56 (8 GB RAM, 3584  stream processors, theoretical 10.5 SP TFLOP, memory bandwidth  410 GB/s)
* M1 Max  (64 GB RAM, 10 CPU cores, 32 GPU Cores, 4096 execution units (which PR material says translates into a theoretical 98304 simultaneous threads), theoretical 10.4 SP TFLOP, memory bandwidth 400 GB/s)

RTX A6000 test was done in 128 GB Xeon W-2125 CPU (4x2 cores) @ 4.00GHz Dell system. AMD Vega 64 and AMD Radeon Pro W6800 were tested in an 128 GB iMac Pro system (10x2 Core Intel Xeon W). The Vega 64 GPU is part of the iMac system, while the Pro W6800 is connected via a Thunderbolt 3 external enclosure. Please note that GPU connectivity should not have an important effect given memory transfers between GPU and CPU are minimal. The M1 Max was configured with 64 GB and installed in a 2021 MacBook Pro system. The Dell system, iMac Pro and MacBook Pro were also used for OpenMP benchmarks. Python 3.9 was used in all systems. The Dell system test used CUDA 11.4 and Visual Studio 2019 Community edition. The latest versions of pycuda and pyopencl at the time of testing (Sep 18, 2022) were used. macOS Monterey 12.5.1 with XCode 14 were used for both iMac Pro and MacBook Pro.

Wall-time was measured from the moment preparations to run GPU code started (initiate device operation) to the end of the memory transfer of results, with no access to the main drives involved. Memory transfer between CPU and GPU occurred only at the beginning and end of GPU execution. The numerical difference among different backends was in the order of single precision resolution.

### Summary of wall-time results for each device
| Device |  CUDA single | OpenCL single |  Metal single | OpenMP single|
| --- | --- |  --- |  --- |  ---  |
| AMD W6800 | - | - |  68 s | - |
| AMD Vega 56 | - |- | 127 s | - |
| NVidia A6000 | 109 s| 104 s | -| - |
| M1 Max | - |  94 s |92 s| 1546 s (10 threads) |
| Xeon W-2125 | - | - | - | 5163 s (8 threads)|
| iMac Pro (Xeon W) | - | - | - | 2982 s  (20 threads)|

#### Discussion of results
The number of computing units is becoming a bit useless to compare. There are a few interesting aspects worth mentioning:
* The ratio of performance between M1 Max and A6000 (CUDA vs. Metal) is not even close to the theoretical difference of raw SP TFLOPS%.
* Metal and OpenCL performances of the M1 Max are pretty much equal, outmatching the A6000 performance.
* Multiple tryouts on the CUDA code to adjust grid and block sizes didn't improve performance in the A6000. On the contrary, wall-time was increased, indicating that the recommended method by Nvidia to calculate maximal occupancy used by default in BabelViscoFDTD provided the best performance with the A6000.
* The other surprise was the W6800 with Metal and OpenCL outperforming by a significant margin the A6000.
* The OpenMP performance of the M1 Max is also excellent, showing a dramatic speedup compared to the Dell Xeon and iMac Pro systems.

## Possibility of manual adjustments to improve performance
All three GPU backends have analogous control to split the calculations in the GPU multiprocessors. BabelViscoFDTD uses the methods that are recommended for each backend to ensure maximal GPU occupancy. However, manual adjustments can provide improvement to the performance. You can specify manually the grid and thread block dimensions with the optional parameters `ManualGroupSize` and `ManualLocalSize`, respectively. Please consult the guidelines of each backend (CUDA, OpenCL and Metal) on how to calculate this correctly, otherwise there is a risk of specifying a too large or too short grid and thread size dimensions. For example, for both CUDA and Metal, the multiplication of `ManualGroupSize` and `ManualLocalSize` must be equal or larger than the domain size ([N1,N2,N3]) to ensure all the domain is covered; for example for a domain of size [231,220,450], `ManualGroupSize=[60,60,115]` with `ManualLocalSize=[4,4,4]` will ensure covering the domain size. For `OpenCL` each entry in `ManualGroupSize` must be equal or larger than [N1,N2,N3] and each entry must be a multiple of its corresponding entry in `ManualLocalSize`; for example for a domain of size [231,220,450], `ManualGroupSize=[240,240,460]` with `ManualLocalSize=[4,4,4]`. Be sure of specifying these parameters as an np.array of type np.int32, such as `ManualLocalSize=np.array([4,4,4]).astype(np.int32)`.

# Supported platforms for Rayleigh's integral
Since v0.9.2 Rayleigh-Sommerfeld's integral was added as a tool (see tutorial `Tutorial Notebooks\Tools -1 - Rayleigh Integral.ipynb`). This will be useful to combine models that include large volumes of water as the Rayleigh integral benefits considerably a GPU as the Rayleigh-Sommerfeld integral is hyper-parallel. The tool has support for 3 GPU backends: CUDA and OpenCL for Windows and Linux, and Metal and OpenCL for macOS.

# Release notes
* 1.09 - Feb 26, 2025
    * Fix error in perfusion conversion formula from ml/min/s to kg/m^3/s
    * Fix similar bug for 1.08 but for grouped sonications
* 1.08 - Dec 3, 2024
    * Fix bug on BHTE Calculations when using a very short duration time
* 1.07 - June 25, 2024
    * Fix bug on Metal for 2D simulations where PML was producing reflections
    * Simplification of 2D kernels for faster calculations 
* 1.06 - March 5, 2024
    * Change compilation of Rayleigh module in ARM64 Metal to support macOS Monterey and up 
* 1.05 - Feb 26, 2024
    * Fix an issue of reflections on one of the sides of the domain when using Metal backend
    * Improve Swift Metal for Rayleigh calculations, passing scalar parameters now using an structure rather than buffers.
    * Add the possibility to limit Rayleigh calculations to only certain distance, useful when forward propagating between Rayleigh and domains that are very close to the Rayleigh source. The Rayleigh functions now accept an optional MaxDistance parameter, which is >0, limits Rayleigh calculations to be less or equal to that distance. Use this with caution.

* 1.0.2 - Oct 14, 2023
    * Small fix to replace remaining use of np.int
* 1.0.1 - Sep 23, 2023
    * Add 2D FDTD operations
    * Significant improvement in performance of BHTE calculations for external AMD GPUs in Apple x64 systems via metal.
* 1.0.0-2 - To fix buffer creation in Metal version. Sep 22, 2023
* 1.0.0-1 - To fix pip version that had an incorrect file for OpenCL BHTE
* 1.0.0 - That is it! After thousands of simulations for a manuscript preparation to introduce the [BabelBrain](https://github.com/ProteusMRIgHIFU/BabelBrain) planning suite (Now public), this is ready for an official 1.0.0 release. 
    * cupy replaces PyCUDA for all CUDA operations. PyCUDA needs a Visual Studio compiler on the path to compile kernels in Windows. BabelBrain uses a lot cupy; switching to it helps to keep using a single interface while benefitting from the availability of Numpy-like methods.
    * Add a new correction parameter for attenuation. One of the findings while preparing the use with BabelBrain was that mapping procedures in the literature linking CT Hounsfield Units (HU) to the speed of sound and, especially, attenuation needs this sort of correction.
* 0.9.10-1 Nov 16, 2022
    * Add functions to list devices supported by computing backends
* 0.9.10 Oct 28, 2022
    * Fix the issue with mapping of unique values when attenuation is used, it could cause some divisions by zero
* 0.9.9.20 Sep 17, 2022
    * A lot of important improvements to make the final line
        - Metal is (finally) running as fast (sometimes slightly faster) than OpenCL in Apple processors. It took a lot of testing and fine tuning.
        - Use of a modified version of the excellent `py-metal-compute` library (https://github.com/baldand/py-metal-compute) that allows having a similar approach as with pyopencl, cupy and pycuda. Modified library is at https://github.com/ProteusMRIgHIFU/py-metal-compute. Because of this new approach, the old Swift interface to the FDTD code was removed.
        - Add Metal backend for BHTE tool. This version runs way faster than OpenCL in Apple processors (like a 10x factor, need to investigate more if we can replicate such gain)
        - Benchmark metrics above were refreshed
        - Moving forward, OpenCL is not recommended for macOS in X64 systems. Because of the limitation of the underlying 32-bit addressing, pyopencl does not catch easily when a simulation goes beyond 4GB. However, Metal for AMD in macOS runs quite well, so no need to stick with OpenCL

* 0.9.9  Sep 1, 2022
    * A lot of simplifications allowed having a much more straightforward code. Thanks to Andrew Xie (@IAmAndrewX) for a very productive summer trimming down code, replacing the old MTLPP with Swift, and making a new class arrangement for the different GPU backends. Now BabelViscoFDTD is based completely on pyopencl and pycuda for the FDTD viscoelastic solver. For Metal, the Swift-based wrapper does the interfacing. The old C extension is still around just for the OpenMP backend.
* 0.9.7  July 7, 2022
    * The MTLPP C++ library is now replaced by a Swift interface to access the Metal implementation for the viscoelastic FDTD solution. This will ensure using a more standard Apple development language for the future, as MTLPP is not maintained anymore. While there is a new Apple-based C++ wrapper for Metal, using Swift is still preferred as we created now a C-linking compatible library that in the future can be also used directly in Python. In the long term, we aim to eliminate the C code extension and use only Python code in tandem with pyopencl, pycuda and  Metal
* 0.9.6-post-10  June 27, 2022
    * A fix for OpenCL in X64 Mac system that was missing the new kernel names
    * OpenMP for X64 in Mac is being turned back as experimental feature as some systems are unable to run with it correctly and there is not a clear path on how to ensure this will be stable. The feature will remain accessible if installing the library with the BABEL_MAC_OPENMP_X64 option enabled.
* 0.9.6  Feb 5, 2022.
    * Improved performance in Metal-based devices by using mini-kernels.
    * Minor fixes for Rayleigh-Sommerfeld in Metal
* 0.9.5  Jan 18, 2022.
    * Very first pip-registered version
    * Possibility of user-specified dimensions of blocks for computations for fine-tuning performance
    * Cleaning some minor bugs and adding BHTE code using pycuda and pyopencl.

* 0.9.3  Sep 29, 2021.
    * Improved support for both Metal and OpenCL. For Metal, stable operation is now feasible for large domains using all available memory in modern high-end GPUs. OpenCL is now supported in all OSs.
* 0.9.2  June 13, 2021.
    * Add Rayleigh integral support in a homogenous medium with CUDA, OpenCL and Metal backends.
    * Support for stress sources
    * Able to select devices in multiple GPU systems with CUDA
* 0.9.1  Feb 17, 2021. Pressure calculation added in low-level function.
* 0.9.0  Dec 2020. First porting from the private repository and big reorganization to make it more user friendly
