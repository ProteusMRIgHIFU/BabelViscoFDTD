import numpy as np;
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler

import _FDTDStaggered3D_with_relaxation_CUDA_double as FDTD_double;
import _FDTDStaggered3D_with_relaxation_CUDA_single as FDTD_single;

import time

from .StaggeredFDTD_3D_With_Relaxation_BASE import StaggeredFDTD_3D_With_Relaxation_BASE

def StaggeredFDTD_3D_With_Relaxation_CUDA(StaggeredFDTD_3D_With_Relaxation_BASE):
    def __init__(self):
        print("Stuff here.")

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _InitSymbolArray(self, IP,_NameVar,td,SCode=[]):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _ownGpuCalloc(self, Name,ctx,td,dims,ArraysGPUOp,flags):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _PrepParamsForKernel(self, arguments):
        raise NotImplementedError("This block must be implemented in a child class")

    def _Execution(self, arguments):
        raise NotImplementedError("This block must be implemented in a child class")

def StaggeredFDTD_3D_CUDA(arguments):
    if (type(arguments)!=dict):
        raise TypeError( "The input parameter must be a dictionary")

    for key in arguments.keys():
        if type(arguments[key])==np.ndarray:
            if np.isfortran(arguments[key])==False:
                #print "StaggeredFDTD_3D: Converting ndarray " + key + " to Fortran convention";
                arguments[key] = np.asfortranarray(arguments[key]);
        elif type(arguments[key])!=str:
            arguments[key]=np.array((arguments[key]))
    t0 = time.time()
    arguments['PI_OCL_PATH']='' #these are unused in CUDA but passed for completeness
    arguments['kernelfile']='' #these are unused in CUDA but passed for completeness
    arguments['kernbinfile']='' #these are unused in CUDA but passed for completeness
    if arguments['DT'].dtype==np.dtype('float32'):
        Results= FDTD_single.FDTDStaggered_3D(arguments)
    else:
        Results= FDTD_double.FDTDStaggered_3D(arguments)
    t0=time.time()-t0
    print ('Time to run low level FDTDStaggered_3D =', t0)
    return Results
