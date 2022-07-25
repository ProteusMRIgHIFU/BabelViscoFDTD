from multiprocessing.sharedctypes import Value
import numpy as np
import os
from pathlib import Path
import platform
import time
import tempfile
from shutil import copyfile

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc

# Do I need to put this in a try block?
import pyopencl as cl 
TotalAllocs=0

class StaggeredFDTD_3D_With_Relaxation_BASE():
    def __init__(self,arguments):
        global NumberSelRMSPeakMaps
        global NumberSelSensorMaps
        global TotalAllocs
        global AllC
        
        IncludeDir=str(Path(__file__).parent.absolute())+os.sep


        if (type(arguments)!=dict):
            raise TypeError( "The input parameter must be a dictionary")

        for key in arguments.keys():
            if type(arguments[key])==np.ndarray:
                if np.isfortran(arguments[key])==False:
                    #print "StaggeredFDTD_3D: Converting ndarray " + key + " to Fortran convention";
                    arguments[key] = np.asfortranarray(arguments[key])
            elif type(arguments[key])!=str:
                arguments[key]=np.array((arguments[key]))

        if arguments['DT'].dtype==np.dtype('float32'):
            dtype=np.float32
        else:
            dtype=np.float64

        
        gpu_kernelSrc=IncludeDir+'_gpu_kernel.c'
        index_src=IncludeDir+'_indexing.h'


        td = 'float'
        if dtype is np.float64:
            td='double'
        t0=time.time()
        
        NumberSelRMSPeakMaps=0
        NumberSelSensorMaps=0
        TotalAllocs=0
        
        outparams=self._PrepParamsForKernel(arguments)

        N1=arguments['N1']
        N2=arguments['N2']
        N3=arguments['N3']
        
        #we prepare the kernel code
        SCode =[]

        platforms = cl.get_platforms()
        devices=platforms[0].get_devices()
        bFound=False
        for device in devices:
            if arguments['DefaultGPUDeviceName'] in device.name:
                bFound=True
                break
        
        if bFound:
            print('Device ', arguments['DefaultGPUDeviceName'], ' Found!')
        else:
            raise ValueError('Device ' + arguments['DefaultGPUDeviceName'] + ' not found!')
        
        address_bits=device.get_info(cl.device_info.ADDRESS_BITS)

        print('Address bits', address_bits)

        if address_bits==32:
            SCode.append("#define _PT_32\n")
        SCode.append("#define mexType " + td +"\n")
        SCode.append("#define OPENCL\n")
        
        with open(index_src) as f:
            SCode+=f.readlines()

        LParamFloat = ['DT']
        LParamInt=["N1","N2", "N3", "Limit_I_low_PML", "Limit_J_low_PML", "Limit_K_low_PML", "Limit_I_up_PML","Limit_J_up_PML",\
                "Limit_K_up_PML","SizeCorrI","SizeCorrJ","SizeCorrK","PML_Thickness","NumberSources", "LengthSource","ZoneCount",\
                "SizePMLxp1","SizePMLyp1","SizePMLzp1","SizePML","SizePMLxp1yp1zp1","NumberSensors","TimeSteps","SelRMSorPeak",\
                "SelMapsRMSPeak", "IndexRMSPeak_ALLV","IndexRMSPeak_Vx","IndexRMSPeak_Vy", "IndexRMSPeak_Vz", "IndexRMSPeak_Sigmaxx",\
                "IndexRMSPeak_Sigmayy","IndexRMSPeak_Sigmazz","IndexRMSPeak_Sigmaxy","IndexRMSPeak_Sigmaxz","IndexRMSPeak_Sigmayz",\
                "IndexRMSPeak_Pressure","NumberSelRMSPeakMaps","SelMapsSensors","IndexSensor_ALLV","IndexSensor_Vx","IndexSensor_Vy",\
                "IndexSensor_Vz","IndexSensor_Sigmaxx","IndexSensor_Sigmayy","IndexSensor_Sigmazz","IndexSensor_Sigmaxy",\
                "IndexSensor_Sigmaxz","IndexSensor_Sigmayz","IndexSensor_Pressure","NumberSelSensorMaps","SensorSubSampling",
                "SensorStart"]
        LParamArray=['InvDXDTplus','DXDTminus','InvDXDTplushp','DXDTminushp']
        assert(len(outparams)==(len(LParamFloat)+len(LParamInt)+len(LParamArray)))
        for k in LParamFloat:
            self._InitSymbol(outparams,k,td,SCode)
        for k in LParamInt:
            self._InitSymbol(outparams,k,'unsigned int',SCode)
        for k in LParamArray:
            self._InitSymbolArray(outparams,k,td,SCode)

        with open(gpu_kernelSrc) as f:
            SCode+=f.readlines()
        AllC=''
        for l in SCode:
            AllC+=l
        
        if platform.system() != 'Windows':
            arguments['PI_OCL_PATH']=IncludeDir+'pi_ocl'
        
        t0 = time.time()
        
        Results = self._Execution(arguments)
            
        t0=time.time()-t0
        print ('Time to run low level FDTDStaggered_3D =', t0)
        AllC=''
        return Results

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):
        raise ValueError("This block shouldn't be reached! Something went wrong in _InitSymbol")
    
    def _InitSymbolArray(self, IP,_NameVar,td,SCode=[]):
        raise ValueError("This block shouldn't be reached! Something went wrong in _InitSymbolArray")
    
    def _ownGpuCalloc(self, Name,ctx,td,dims,ArraysGPUOp,flags=cl.mem_flags.READ_WRITE):
        raise ValueError("This block shouldn't be reached! Something went wrong in _ownGpuCalloc")
    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR):
        raise ValueError("This block shouldn't be reached! Something went wrong in _CreateAndCopyFromMXVarOnGPU")
    
    def _PrepParamsForKernel(self, arguments):
        raise ValueError("This block shouldn't be reached! Something went wrong in _PrepParamsForKernel")
    def _Execution(self, arguments):
        raise ValueError("This block shouldn't be reached! Something went wrong in _Execution")      
    