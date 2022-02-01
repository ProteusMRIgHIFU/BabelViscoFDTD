import numpy as np
import os
from pathlib import Path
import _FDTDStaggered3D_with_relaxation_METAL_single as FDTD_single;

import time
import tempfile
from shutil import copyfile

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc



def StaggeredFDTD_3D_METAL(arguments):
    IncludeDir=str(Path(__file__).parent.absolute())+os.sep

    filenames = [IncludeDir+'_indexing.h',IncludeDir+'_gpu_kernel.c']

    kernbinfile=IncludeDir+'tools'+os.sep+'Rayleigh.metallib'
    
    

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
    arguments['PI_OCL_PATH']='' #unused in METAL but needed in the low level function for completeness
    arguments['kernelfile']=''
    arguments['kernbinfile']=kernbinfile
    
    if arguments['DT'].dtype==np.dtype('float32'):
        Results= FDTD_single.FDTDStaggered_3D(arguments)
    else:
        raise SystemError("Metal backend only supports single precision")
    t0=time.time()-t0
    print ('Time to run low level FDTDStaggered_3D =', t0)
    return Results
