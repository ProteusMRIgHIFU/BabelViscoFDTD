import numpy as np
import _FDTDStaggered3D_with_relaxation_double as FDTD_double;
import _FDTDStaggered3D_with_relaxation_single as FDTD_single;
import time


def StaggeredFDTD_3D(arguments):
    if (type(arguments)!=dict):
        raise TypeError( "The input parameter must be a dictionary")

    for key in arguments.keys():
        if type(arguments[key])==np.ndarray:
            if np.isfortran(arguments[key])==False:
                #print "StaggeredFDTD_3D: Converting ndarray " + key + " to Fortran convention";
                arguments[key] = np.asfortranarray(arguments[key]);
        elif type(arguments[key])!=str:
            arguments[key]=np.array((arguments[key]))
    t0=time.time()
    arguments['PI_OCL_PATH']='' #unused in OpenMP but needed in the low level function for completeness
    arguments['kernelfile']='' #these are unused in OpenMP but passed for completeness
    arguments['kernbinfile']='' #these are unused in OpenMP but passed for completeness
    
    if arguments['DT'].dtype==np.dtype('float32'):
        Results= FDTD_single.FDTDStaggered_3D(arguments)
    else:
        Results = FDTD_double.FDTDStaggered_3D(arguments)
    t0=time.time()-t0
    print ('Time to run low level FDTDStaggered_3D =', t0)
    return Results
