import numpy as np;
import os

import _FDTDStaggered3D_with_relaxation_OPENCL_single as FDTD_single;
import _FDTDStaggered3D_with_relaxation_OPENCL_double as FDTD_double;

import time
from shutil import copyfile

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc



def StaggeredFDTD_3D_OPENCL(arguments):
    IncludeDir=get_python_inc()+os.sep+'FDTDStaggered3D-with-relaxation'+os.sep
    print("Copying opencl files from "+IncludeDir +" to " +os.getcwd())
    copyfile(IncludeDir+'_opencl_kernel.c', os.getcwd()+os.sep+'_opencl_kernel.c')
    copyfile(IncludeDir+'_indexing.h', os.getcwd()+os.sep+'_indexing.h')

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
    if arguments['DT'].dtype==np.dtype('float32'):
        Results= FDTD_single.FDTDStaggered_3D(arguments)
    else:
        Results= FDTD_double.FDTDStaggered_3D(arguments)
    t0=time.time()-t0
    print ('Time to run low level FDTDStaggered_3D =', t0)
    return Results
