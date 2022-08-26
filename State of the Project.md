State of the Project
====================
Andrew Xie  
Visiting Summer Research Student  
Department of Bioengineering  
McGill University  
andrew.xie@mail.mcgill.ca  
August 26, 2022 

**The purpose of this file is to discuss the state of the OOP migration to Python, as of August 26, 2022, for ease of future development on this project.**

## OpenCL and BASE Implementation
The PyOpenCL implementation was successfully manipulated to split it into a base class (`StaggeredFDTD_3D_With_Relaxation_BASE.py`) that unifies the common logic, and it's own child class to fill in the gaps. 

### BASE Specific Quirks
Currently, the child class passes an `extra_params` dictionary to the parent class. This is able to define certain things easily, such as which backend is being called on, to both address certain quirks (i.e. Metal only supporting single precision, OpenCL using an `SCode` list).

### The Future of the OpenCL Implementation on x86-64 Mac Devices
Currently, the OpenCL implementation on x86-64 Mac Devices involves the usage of the C backend, as PyOpenCL performs extremely poorly on these devices. However, one of the main goals of this project is to eliminate the C backend. Therefore, it is likely that OpenCL on x86-64 Mac Devices will no longer be supported as of project completion, unless we find that PyOpenCL can perform well on these devices. Therefore, in the future, the line in `BASE` that involves:
`if extra_params['BACKEND'] == 'OPENCL' and platform.system() == 'Darwin' and 'arm64' not in platform.platform():`
should throw an error.

## Metal
Although the Python version of the Metal backend is able to run, it has been giving us a bad output. I've spent a good amount of time on this, however, I still haven't been able to find the source of this bad output. It is likely caused by a faulty data transfer, or somewhere in the code I'm missing a 1:1 translation from C to Python.

One of the things I found was that the library `ctypes` (used to transmit data from Python to `MetalSwift.swift`) may send some weird data when trying to do certain things, such as translating an `np.uint64` to a `ctypes.c_uint32`. This may be a source of the error, or it may not be; I've fixed any data transfers that look weird due to the ctypes translation, however there may be one I missed.

Another source may be that I translated one of the constants incorrectly from `Indexing.h` or other files in the C implementation. Although I have checked multiple times, this may be the source of the error as well. All these constants are mostly in the `__init__` of the child class.

Most references and calls to variables relating to the scantly used `Snapshots` function has mostly been removed. To ensure that the code doesn't break upstream, `ArrayResCPU['Snapshots']` is defined to be a large array of zeros, with size as the upstream code would expect. This mostly just serves as a reminder for when the `Snapshots` function does eventually get totally phased out, to remove the very few remaining bits regarding the `Snapshots` array.

## CUDA
The beginnings of a CUDA implementation has been put into place, using PyCUDA. Since the implementation of CUDA is mostly 1:1 with the OpenCL implementation, this part of the project should not take much time. I will likely work on this on my offtime, just so that progress on this project isn't stalled so that future improvements to BabelViscoFDTD will not be stalled.

A future consideration is the new NVIDIA Python Library `CUDA Python`. Although it doesn't seem like there is any benefits to overhead performance, it may be worth looking into, especially if PyCUDA does eventually phase itself out. This development will likely be done in another branch after this one is integrated into the main repo.

## Overall Cleanup
Once all three implementations are completed, a cleanup of `StaggeredFDTD_3D_With_Relaxation_BASE.py`, `StaggeredFDTD_3D_With_Relaxation_CUDA.py`, `StaggeredFDTD_3D_With_Relaxation_METAL.py`, and `StaggeredFDTD_3D_With_Relaxation_OPENCL.py`, should be initiated in order to ensure the future readability of the code.

As well, the removal of several C files, including `FDTD3D_GPU_VERSION.h`, as well as the OpenCL Implementation on x86-64 Mac Devices (see above) should be removed.

## Conclusion
All in all, this project is fairly close to completion. With the CUDA implementation being nearly the same as the OpenCL implementation (which is already finished), and just minor bugfixing in the Metal implementation, the project is nearly at completion.

If you are needing any type of support for this project (i.e. you're a future summer student in Dr. Pichardo's lab), don't hesitate to reach out to me by my contact email!
