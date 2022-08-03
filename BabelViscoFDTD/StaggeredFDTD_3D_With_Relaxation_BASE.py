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

MASKID={}
MASKID['ALLV']=0x0000000001
MASKID['Vx']  =0x0000000002
MASKID['Vy']  =0x0000000004
MASKID['Vz']  =0x0000000008
MASKID['Sigmaxx'] =0x0000000010
MASKID['Sigmayy'] =0x0000000020
MASKID['Sigmazz'] =0x0000000040
MASKID['Sigmaxy'] =0x0000000080
MASKID['Sigmaxz'] =0x0000000100
MASKID['Sigmayz'] =0x0000000200
MASKID['Pressure']=0x0000000400
MASKID['SEL_RMS']=0x0000000001
MASKID['SEL_PEAK']=0x0000000002

class StaggeredFDTD_3D_With_Relaxation_BASE():
    def __init__(self,arguments, extra_params=[]):
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
        elif extra_params["BACKEND"] == "METAL":
            raise SystemError("Metal backend only supports single precision")
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
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _InitSymbolArray(self, IP,_NameVar,td,SCode=[]):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _ownGpuCalloc(self, Name,ctx,td,dims,ArraysGPUOp,flags):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _PrepParamsForKernel(self, arguments):
        global NumberSelRMSPeakMaps
        global NumberSelSensorMaps


        copyParams=['DT','N1','N2','N3','SensorSubSampling','SensorStart','LengthSource','TimeSteps',\
                'SelRMSorPeak','SelMapsRMSPeak','SelMapsSensors','InvDXDTplus','DXDTminus','InvDXDTplushp','DXDTminushp']
        outparams={}
        for c in copyParams:
            outparams[c]=arguments[c]
            
        outparams['PML_Thickness']=arguments['PMLThickness']
            
        outparams['NumberSources']=arguments['SourceFunctions'].shape[0]
        outparams['ZoneCount']=arguments['SPP_ZONES']
        outparams['NumberSensors']=arguments['IndexSensorMap'].size
        
        
        outparams['Limit_I_low_PML']=outparams['PML_Thickness']-1
        outparams['Limit_I_up_PML']=arguments['N1']-outparams['PML_Thickness']
        outparams['Limit_J_low_PML']=outparams['PML_Thickness']-1
        outparams['Limit_J_up_PML']=arguments['N2']-outparams['PML_Thickness']
        outparams['Limit_K_low_PML']=outparams['PML_Thickness']-1
        outparams['Limit_K_up_PML']=arguments['N3']-outparams['PML_Thickness']

        outparams['SizeCorrI']=arguments['N1']-2*outparams['PML_Thickness']
        outparams['SizeCorrJ']=arguments['N2']-2*outparams['PML_Thickness']
        outparams['SizeCorrK']=arguments['N3']-2*outparams['PML_Thickness']

        #//The size of the matrices where the PML is valid depends on the size of the PML barrier
        outparams['SizePML']= arguments['N1']*arguments['N2']*arguments['N3'] - outparams['SizeCorrI']*outparams['SizeCorrJ']*outparams['SizeCorrK']+1
        outparams['SizePMLxp1']= (arguments['N1']+1)*(arguments['N2'])*(arguments['N3']) - outparams['SizeCorrI']*outparams['SizeCorrJ']*outparams['SizeCorrK']+1
        outparams['SizePMLyp1']= arguments['N1']*(arguments['N2']+1)*arguments['N3'] - outparams['SizeCorrI']*outparams['SizeCorrJ']*outparams['SizeCorrK']+1
        outparams['SizePMLzp1']= arguments['N1']*(arguments['N2'])*(arguments['N3']+1) - outparams['SizeCorrI']*outparams['SizeCorrJ']*outparams['SizeCorrK']+1
        outparams['SizePMLxp1yp1zp1']= (arguments['N1']+1)*(arguments['N2']+1)*(arguments['N3']+1) - outparams['SizeCorrI']*outparams['SizeCorrJ']*outparams['SizeCorrK']+1

        for k in ['ALLV','Vx','Vy','Vz','Sigmaxx','Sigmayy','Sigmazz','Sigmaxy','Sigmaxz','Sigmayz','Pressure']:
            outparams['IndexRMSPeak_'+k]=0
            outparams['IndexSensor_'+k]=0
            if arguments['SelMapsRMSPeak'] & MASKID[k]:
                outparams['IndexRMSPeak_'+k]=NumberSelRMSPeakMaps
                NumberSelRMSPeakMaps+=1
            if arguments['SelMapsSensors'] & MASKID[k]:
                outparams['IndexSensor_'+k]=NumberSelSensorMaps
                NumberSelSensorMaps+=1
        outparams['NumberSelRMSPeakMaps']=NumberSelRMSPeakMaps
        outparams['NumberSelSensorMaps']=NumberSelSensorMaps
        return outparams

    def _Execution(self, arguments):
        raise NotImplementedError("This block must be implemented in a child class")
    