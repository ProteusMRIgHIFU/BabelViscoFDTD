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

TotalAllocs=0

MASKID={}
MASKID['Vx']=           0x0000000001
MASKID['Vy']  =         0x0000000002
MASKID['Sigmaxx']  =    0x0000000004
MASKID['Sigmayy'] =     0x0000000008
MASKID['Sigmaxy'] =     0x0000000010
MASKID['Pressure'] =    0x0000000020
MASKID['Pressure_gx'] = 0x0000000040
MASKID['Pressure_gy'] = 0x0000000080
MASKID['SEL_RMS'] =0x0000000001
MASKID['SEL_PEAK']=0x0000000002
class StaggeredFDTD_2D_With_Relaxation_BASE():
    def __init__(self, arguments, extra_params={}):
        global NumberSelRMSPeakMaps # Is it necessary to keep these global anymore?
        global NumberSelSensorMaps
        global TotalAllocs
        
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

        gpu_kernelSrc=IncludeDir+'_gpu_kernel2D.c'
        index_src=IncludeDir+'_indexing2D.h'


        td = 'float'
        if dtype is np.float64:
            td='double'
        extra_params['td'] = td
    
        t0=time.time()
        
        NumberSelRMSPeakMaps=0
        NumberSelSensorMaps=0
        TotalAllocs=0
        
        outparams = self._PrepParamsForKernel(arguments)
        
        #we prepare the kernel code

        self._PostInitScript(arguments, extra_params)
        
        
        SCode = extra_params["SCode"]
        with open(index_src) as f:
            SCode+=f.readlines()
        
    
        LParamFloat = ['DT']
        LParamInt=["N1","N2", "Limit_I_low_PML", "Limit_J_low_PML", "Limit_I_up_PML","Limit_J_up_PML",\
                 "SizeCorrI","SizeCorrJ","PML_Thickness","NumberSources", "LengthSource","ZoneCount",\
                "SizePMLxp1","SizePMLyp1","SizePML","SizePMLxp1yp1","NumberSensors","TimeSteps","SelRMSorPeak",\
                "SelMapsRMSPeak","IndexRMSPeak_Vx","IndexRMSPeak_Vy", "IndexRMSPeak_Sigmaxx",\
                "IndexRMSPeak_Sigmayy","IndexRMSPeak_Sigmaxy",\
                "IndexRMSPeak_Pressure","IndexRMSPeak_Pressure_gx","IndexRMSPeak_Pressure_gy","NumberSelRMSPeakMaps","SelMapsSensors","IndexSensor_Vx","IndexSensor_Vy",\
                "IndexSensor_Sigmaxx","IndexSensor_Sigmayy","IndexSensor_Sigmaxy",\
                "IndexSensor_Pressure","IndexSensor_Pressure_gx","IndexSensor_Pressure_gy","NumberSelSensorMaps","SensorSubSampling",
                "SensorStart"]
        LParamArray=['InvDXDTplus','DXDTminus','InvDXDTplushp','DXDTminushp']
        tt =LParamFloat+LParamInt+LParamArray
        for k in tt:
            if k not in outparams:
                print(k)
        assert len(outparams)==(len(LParamFloat)+len(LParamInt)+len(LParamArray))
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
    
        N1=arguments['N1']
        N2=arguments['N2']

                    
        ArrayResCPU={}
        for k in ['Sigma_xx','Sigma_yy','Pressure']:
            ArrayResCPU[k]=np.zeros((N1,N2),dtype,order='F')
        for k in ['Sigma_xy']:
            ArrayResCPU[k]=np.zeros((N1+1,N2+1),dtype,order='F')
        ArrayResCPU['Vx']=np.zeros((N1+1,N2),dtype,order='F')
        ArrayResCPU['Vy']=np.zeros((N1,N2+1),dtype,order='F')
    
        if 	(arguments['SelRMSorPeak'] &  MASKID['SEL_PEAK']) and (arguments['SelRMSorPeak'] &  MASKID['SEL_RMS']):
            #both peak and RMS
            updims=2 
        else:
            updims=1

        ArrayResCPU['SqrAcc']=np.zeros((N1,N2,outparams['NumberSelRMSPeakMaps'],updims),dtype,order='F')
        Ns=1
        NumberSnapshots=arguments['SnapshotsPos'].size
        NumberSensors=arguments['IndexSensorMap'].size
        if NumberSnapshots>0:
            Ns=NumberSnapshots
        ArrayResCPU['Snapshots']=np.zeros((N1,N2,Ns),dtype,order='F')
        TimeSteps=arguments['TimeSteps']
        SensorSubSampling=arguments['SensorSubSampling']
        SensorStart=arguments['SensorStart']
        print("Number Selected Sensor Maps:", outparams['NumberSelRMSPeakMaps'])
        ArrayResCPU['SensorOutput']=np.zeros((NumberSensors,int(TimeSteps/SensorSubSampling)+1-SensorStart,outparams['NumberSelSensorMaps']),dtype,order='F')
        
        self._InitiateCommands(AllC)

        ArraysGPUOp={}
        if extra_params["BACKEND"] in ["OPENCL","CUDA"]:
            for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                        'Ox','Oy','SourceFunctions','IndexSensorMap','SourceMap','MaterialMap']:            
                self._CreateAndCopyFromMXVarOnGPU(k,ArraysGPUOp,arguments)
        for k in ['V_x_x','V_y_x']:
            self._ownGpuCalloc(k,td,outparams['SizePMLxp1']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['V_x_y','V_y_y']:
            self._ownGpuCalloc(k,td,outparams['SizePMLyp1']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Sigma_x_xx','Sigma_y_xx','Sigma_x_yy','Sigma_y_yy']:
            self._ownGpuCalloc(k,td,outparams['SizePML']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Sigma_x_xy','Sigma_y_xy']:
            self._ownGpuCalloc(k,td,outparams['SizePMLxp1yp1']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Rxx','Ryy']:
            self._ownGpuCalloc(k,td,ArrayResCPU['Sigma_xx'].size*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Rxy']:
            self._ownGpuCalloc(k,td,ArrayResCPU['Sigma_xy'].size*outparams['ZoneCount'],ArraysGPUOp)
        if extra_params['BACKEND'] in ['OPENCL','CUDA']:
            for k in ['Vx','Vy','Sigma_xx','Sigma_yy','Pressure','Sigma_xy','Snapshots']:
                self._ownGpuCalloc(k,td,ArrayResCPU[k].size*outparams['ZoneCount'],ArraysGPUOp)
            for k in ['SensorOutput','SqrAcc']:
                self._CreateAndCopyFromMXVarOnGPU(k, ArraysGPUOp, ArrayResCPU)
            
        else: # ORDER DOES MATTER FOR METAL, AS IT INVOLVES MANIPULATING AND READING _c_uint_type OR _c_mex_type******
            for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                        'Ox','Oy','SourceFunctions','IndexSensorMap','SourceMap','MaterialMap']:            
                self._CreateAndCopyFromMXVarOnGPU(k,ArraysGPUOp,arguments)
            
            for k in ['Vx','Vy','Sigma_xx','Sigma_yy','Sigma_xy','Pressure','Snapshots']:
                self._ownGpuCalloc(k,td,ArrayResCPU[k].size*outparams['ZoneCount'],ArraysGPUOp)

            for k in ['SqrAcc','SensorOutput']:
                self._CreateAndCopyFromMXVarOnGPU(k, ArraysGPUOp, ArrayResCPU)

        self._PreExecuteScript(arguments, ArraysGPUOp, outparams)

        self._Execution(arguments, ArrayResCPU, ArraysGPUOp)
            
        t0=time.time()-t0
        print ('Time to run low level FDTDStaggered_3D =', t0)

        AllC=''
        
        self.CreateResults(ArrayResCPU)

        return

    def _PostInitScript(self, arguments):
        raise NotImplementedError("This block must be implemented in a child class!")

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):
        raise NotImplementedError("This block must be implemented in a child class!")
    
    def _InitSymbolArray(self, IP,_NameVar,td,SCode=[]):
        raise NotImplementedError("This block must be implemented in a child class!")
    
    def _ownGpuCalloc(self, Name,td,dims,ArraysGPUOp,flags):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags):
        raise NotImplementedError("This block must be implemented in a child class")
    
    def _PrepParamsForKernel(self, arguments):
        global NumberSelRMSPeakMaps
        global NumberSelSensorMaps


        copyParams=['DT','N1','N2','SensorSubSampling','SensorStart','LengthSource','TimeSteps',\
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

        outparams['SizeCorrI']=arguments['N1']-2*outparams['PML_Thickness']
        outparams['SizeCorrJ']=arguments['N2']-2*outparams['PML_Thickness']

        #//The size of the matrices where the PML is valid depends on the size of the PML barrier
        outparams['SizePML']= arguments['N1']*arguments['N2'] - outparams['SizeCorrI']*outparams['SizeCorrJ']+1
        outparams['SizePMLxp1']= (arguments['N1']+1)*(arguments['N2']) - outparams['SizeCorrI']*outparams['SizeCorrJ']+1
        outparams['SizePMLyp1']= arguments['N1']*(arguments['N2']+1) - outparams['SizeCorrI']*outparams['SizeCorrJ']+1
        outparams['SizePMLxp1yp1']= (arguments['N1']+1)*(arguments['N2']+1) - outparams['SizeCorrI']*outparams['SizeCorrJ']+1

        for k in ['Vx','Vy','Sigmaxx','Sigmayy','Sigmaxy','Pressure','Pressure_gx','Pressure_gy']:
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
    
    def CreateResults(self, ArrayResCPU):
        self.Results = ArrayResCPU['SensorOutput'],\
                {'Vx':ArrayResCPU['Vx'],\
                'Vy':ArrayResCPU['Vy'],\
                'Sigma_xx':ArrayResCPU['Sigma_xx'],\
                'Sigma_yy':ArrayResCPU['Sigma_yy'],\
                'Sigma_xy':ArrayResCPU['Sigma_xy'],\
                'Pressure':ArrayResCPU['Pressure']},\
                ArrayResCPU['SqrAcc'],ArrayResCPU['Snapshots'] 