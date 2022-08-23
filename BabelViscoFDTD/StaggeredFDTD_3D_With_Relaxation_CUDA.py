import numpy as np;
import pycuda.driver as cuda
import pycuda.compiler

import _FDTDStaggered3D_with_relaxation_CUDA_double as FDTD_double;
import _FDTDStaggered3D_with_relaxation_CUDA_single as FDTD_single;

import time

from .StaggeredFDTD_3D_With_Relaxation_BASE import StaggeredFDTD_3D_With_Relaxation_BASE

class StaggeredFDTD_3D_With_Relaxation_CUDA(StaggeredFDTD_3D_With_Relaxation_BASE):
    def __init__(self, arguments):
        extra_params = {"BACKEND":"CUDA"}
        super().__init__(arguments, extra_params)

    def _PostInitScript(self, arguments, extra_params):
        cuda.init()
        devCount = cuda.Device.count()
        print("Number of CUDA devices found:", devCount)
        if devCount == 0:
            raise SystemError("There are no CUDA devices.")
        
        devicelist = {}

        for deviceID in range(0, devCount):
            print("Found device", str(deviceID + 1) + "/" + str(devCount))
            device = cuda.Device(deviceID)
            print("Name:", device.name())
            print("Compute capability:", device.compute_capability())
            print("Total memory:", device.total_memory(), "bytes")
            print("Threads per block:", device.MAX_THREADS_PER_BLOCK)
            print("Max block dimensions:", device.MAX_BLOCK_DIM_X, "x", device.MAX_BLOCK_DIM_Y, "x", device.MAX_BLOCK_DIM_Z)
            print("Max grid size:", device.MAX_GRID_DIM_X, "x", device.MAX_GRID_DIM_Y, "x", device.MAX_GRID_DIM_Z)
            print("Shared memory per block:", device.MAX_SHARED_MEMORY_PER_BLOCK)
            print("Registers per block:", device.MAX_REGISTERS_PER_BLOCK)
            if device.COMPUTE_CAPABILITY_MAJOR >= 3 and arguments['DefaultGPUDeviceName'] in device.name():
                devicelist[deviceID] = device
    
        if len(devicelist) == 0:
            raise SystemError("There are no devices supporting CUDA or that matches selected device.")
        elif len(devicelist) < arguments['DefaultGPUDeviceNumber']:
            print("The requested device,", arguments['DefaultGPUDeviceNumber'], "(0-base index) is more than the number of devices available", len(devicelist))
            raise IndexError("Unable to select requested device.")

        print("Selecting device", arguments['DefaultGPUDeviceName'], "with number: ", arguments['DefaultGPUDeviceNumber'])

        self.device = devicelist[int(arguments['DefaultGPUDeviceNumber'])]
        print("Device Created?")
        self.context = device.make_context()
        print("Context Created!")
        self.context.set_cache_config(cuda.func_cache.PREFER_L1)

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):
        
        pass    
        
    def _InitSymbolArray(self, IP,_NameVar,td,SCode=[]):
        pass

    def _ownGpuCalloc(self, Name,ctx,td,dims,ArraysGPUOp,flags):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags):
        raise NotImplementedError("This block must be implemented in a child class")

    def _Execution(self, arguments):
        raise NotImplementedError("This block must be implemented in a child class")

def StaggeredFDTD_3D_CUDA(arguments):
    Instance = StaggeredFDTD_3D_With_Relaxation_CUDA(arguments)
    return Instance.Results
#    if (type(arguments)!=dict):
#        raise TypeError( "The input parameter must be a dictionary")

#    for key in arguments.keys():
#        if type(arguments[key])==np.ndarray:
#            if np.isfortran(arguments[key])==False:
#                #print "StaggeredFDTD_3D: Converting ndarray " + key + " to Fortran convention";
#                arguments[key] = np.asfortranarray(arguments[key]);
#        elif type(arguments[key])!=str:
#            arguments[key]=np.array((arguments[key]))
#    t0 = time.time()
#    arguments['PI_OCL_PATH']='' #these are unused in CUDA but passed for completeness
#    arguments['kernelfile']='' #these are unused in CUDA but passed for completeness
#    arguments['kernbinfile']='' #these are unused in CUDA but passed for completeness
#    if arguments['DT'].dtype==np.dtype('float32'):
#        Results= FDTD_single.FDTDStaggered_3D(arguments)
#    else:
#        Results= FDTD_double.FDTDStaggered_3D(arguments)
#    t0=time.time()-t0
#    print ('Time to run low level FDTDStaggered_3D =', t0)
#    return Results
