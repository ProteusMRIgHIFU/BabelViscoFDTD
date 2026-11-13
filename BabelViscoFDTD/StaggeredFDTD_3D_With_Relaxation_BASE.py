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
MASKID['Vx']=0x0000000001
MASKID['Vy']  =0x0000000002
MASKID['Vz']  =0x0000000004
MASKID['Sigmaxx']  =0x0000000008
MASKID['Sigmayy'] =0x0000000010
MASKID['Sigmazz'] =0x0000000020
MASKID['Sigmaxy'] =0x0000000040
MASKID['Sigmaxz'] =0x0000000080
MASKID['Sigmayz'] =0x0000000100
MASKID['Pressure'] =0x0000000200
MASKID['SEL_RMS'] =0x0000000001
MASKID['SEL_PEAK']=0x0000000002

class StaggeredFDTD_3D_With_Relaxation_BASE():
    def __init__(self, arguments, extra_params={}):
        global NumberSelRMSPeakMaps # Is it necessary to keep these global anymore?
        global NumberSelSensorMaps
        global TotalAllocs
        
        IncludeDir=str(Path(__file__).parent.absolute())+os.sep


        if (type(arguments)!=dict):
            raise TypeError( "The input parameter must be a dictionary")

        for key in arguments.keys():
            if type(arguments[key])==np.ndarray:
                # the kernels use C order (k is the fastest index); C-contiguous arrays are used without a copy
                arguments[key] = np.ascontiguousarray(arguments[key])
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
        LParamInt=["N1","N2", "N3", "Limit_I_low_PML", "Limit_J_low_PML", "Limit_K_low_PML", "Limit_I_up_PML","Limit_J_up_PML",\
                "Limit_K_up_PML","SizeCorrI","SizeCorrJ","SizeCorrK","PML_Thickness","NumberSources", "LengthSource","ZoneCount",\
                "SizePMLxp1","SizePMLyp1","SizePMLzp1","SizePML","SizePMLxp1yp1zp1","NumberSensors","TimeSteps","SelRMSorPeak",\
                "SelMapsRMSPeak","IndexRMSPeak_Vx","IndexRMSPeak_Vy", "IndexRMSPeak_Vz", "IndexRMSPeak_Sigmaxx",\
                "IndexRMSPeak_Sigmayy","IndexRMSPeak_Sigmazz","IndexRMSPeak_Sigmaxy","IndexRMSPeak_Sigmaxz","IndexRMSPeak_Sigmayz",\
                "IndexRMSPeak_Pressure","NumberSelRMSPeakMaps","SelMapsSensors","IndexSensor_Vx","IndexSensor_Vy",\
                "IndexSensor_Vz","IndexSensor_Sigmaxx","IndexSensor_Sigmayy","IndexSensor_Sigmazz","IndexSensor_Sigmaxy",\
                "IndexSensor_Sigmaxz","IndexSensor_Sigmayz","IndexSensor_Pressure","NumberSelSensorMaps","SensorSubSampling",
                "SensorStart"]
        LParamArray=['InvDXDTplus','DXDTminus','InvDXDTplushp','DXDTminushp']
        assert len(outparams)==(len(LParamFloat)+len(LParamInt)+len(LParamArray))
        for k in LParamFloat:
            self._InitSymbol(outparams,k,td,SCode)
        for k in LParamInt:
            self._InitSymbol(outparams,k,'unsigned int',SCode)
        for k in LParamArray:
            self._InitSymbolArray(outparams,k,td,SCode)

        with open(gpu_kernelSrc) as f:
            SCode+=f.readlines()


        N1=arguments['N1']
        N2=arguments['N2']
        N3=arguments['N3']
                    
        # All the arrays are C-contiguous, with the layout used by the kernels. The maps of SqrAcc, the
        # snapshots and the sensor maps and time steps are their slowest dimensions; CreateResults returns
        # them as views with the dimension order of the results
        ArrayResCPU={}
        for k in ['Sigma_xx','Sigma_yy','Sigma_zz','Pressure']:
            ArrayResCPU[k]=np.zeros((N1,N2,N3),dtype)
        for k in ['Sigma_xy','Sigma_xz','Sigma_yz']:
            ArrayResCPU[k]=np.zeros((N1+1,N2+1,N3+1),dtype)
        ArrayResCPU['Vx']=np.zeros((N1+1,N2,N3),dtype)
        ArrayResCPU['Vy']=np.zeros((N1,N2+1,N3),dtype)
        ArrayResCPU['Vz']=np.zeros((N1,N2,N3+1),dtype)

        if 	(arguments['SelRMSorPeak'] &  MASKID['SEL_PEAK']) and (arguments['SelRMSorPeak'] &  MASKID['SEL_RMS']):
            #both peak and RMS
            updims=2 
        else:
            updims=1

        ArrayResCPU['SqrAcc']=np.zeros((updims,outparams['NumberSelRMSPeakMaps'],N1,N2,N3),dtype)
        Ns=1
        NumberSnapshots=arguments['SnapshotsPos'].size
        NumberSensors=arguments['IndexSensorMap'].size
        if NumberSnapshots>0:
            Ns=NumberSnapshots
        ArrayResCPU['Snapshots']=np.zeros((Ns,N1,N2),dtype)
        TimeSteps=arguments['TimeSteps']
        SensorSubSampling=arguments['SensorSubSampling']
        SensorStart=arguments['SensorStart']
        print("Number Selected Sensor Maps:", outparams['NumberSelRMSPeakMaps'])
        ArrayResCPU['SensorOutput']=np.zeros((outparams['NumberSelSensorMaps'],int(TimeSteps/SensorSubSampling)+1-SensorStart,NumberSensors),dtype)
        
        self._InitiateCommands(SCode)

        ArraysGPUOp={}
        if extra_params["BACKEND"] in ["OPENCL","CUDA"]:
            for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                        'Ox','Oy','Oz','SourceFunctions','IndexSensorMap','SourceMap','MaterialMap','ReflectorMask']:            
                self._CreateAndCopyFromMXVarOnGPU(k,ArraysGPUOp,arguments)
        for k in ['V_x_x','V_y_x','V_z_x']:
            self._ownGpuCalloc(k,td,outparams['SizePMLxp1']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['V_x_y','V_y_y','V_z_y']:
            self._ownGpuCalloc(k,td,outparams['SizePMLyp1']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['V_x_z','V_y_z','V_z_z']:
            self._ownGpuCalloc(k,td,outparams['SizePMLzp1']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Sigma_x_xx','Sigma_y_xx','Sigma_z_xx','Sigma_x_yy','Sigma_y_yy','Sigma_z_yy','Sigma_x_zz','Sigma_y_zz','Sigma_z_zz']:
            self._ownGpuCalloc(k,td,outparams['SizePML']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Sigma_x_xy','Sigma_y_xy','Sigma_x_xz','Sigma_z_xz','Sigma_y_yz','Sigma_z_yz']:
            self._ownGpuCalloc(k,td,outparams['SizePMLxp1yp1zp1']*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Rxx','Ryy','Rzz']:
            self._ownGpuCalloc(k,td,ArrayResCPU['Sigma_xx'].size*outparams['ZoneCount'],ArraysGPUOp)
        for k in ['Rxy','Rxz','Ryz']:
            self._ownGpuCalloc(k,td,ArrayResCPU['Sigma_xy'].size*outparams['ZoneCount'],ArraysGPUOp)
        if extra_params['BACKEND'] in ['OPENCL','CUDA']:
            for k in ['Vx','Vy','Vz','Sigma_xx','Sigma_yy','Sigma_zz','Pressure','Sigma_xy','Sigma_xz','Sigma_yz','Snapshots']:
                self._ownGpuCalloc(k,td,ArrayResCPU[k].size*outparams['ZoneCount'],ArraysGPUOp)
            for k in ['SensorOutput','SqrAcc']:
                self._CreateAndCopyFromMXVarOnGPU(k, ArraysGPUOp, ArrayResCPU)
            
        else: # ORDER DOES MATTER FOR METAL, AS IT INVOLVES MANIPULATING AND READING _c_uint_type OR _c_mex_type******
            for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                        'Ox','Oy','Oz','SourceFunctions','IndexSensorMap','SourceMap','MaterialMap','ReflectorMask']:            
                self._CreateAndCopyFromMXVarOnGPU(k,ArraysGPUOp,arguments)
            
            for k in ['Vx','Vy','Vz','Sigma_xx','Sigma_yy','Sigma_zz','Sigma_xy','Sigma_xz','Sigma_yz','Pressure','Snapshots']:
                self._ownGpuCalloc(k,td,ArrayResCPU[k].size*outparams['ZoneCount'],ArraysGPUOp)

            for k in ['SensorOutput','SqrAcc']:
                self._CreateAndCopyFromMXVarOnGPU(k, ArraysGPUOp, ArrayResCPU)

        self._PreExecuteScript(arguments, ArraysGPUOp, outparams)

        self._Execution(arguments, ArrayResCPU, ArraysGPUOp)
            
        t0=time.time()-t0
        print ('Time to run low level FDTDStaggered_3D =', t0)

        AllC=''
        
        self.CreateResults(ArrayResCPU)

        return

    # ---- Host helpers shared by the Apple GPU backends (Metal and MLX) ----

    def _RequiresIndex64(self, arguments):
        '''
        Kernels use 32-bit indexing unless an array has 2^32 or more entries (the indices inside
        each array are 32-bit; the offsets of arrays inside the shared Metal buffers remain 64-bit).
        The bounds below are conservative upper limits of the largest arrays.
        '''
        N1=int(arguments['N1'])
        N2=int(arguments['N2'])
        N3=int(arguments['N3'])
        ZoneCount=int(arguments['SPP_ZONES'])
        NumberSteps=int(arguments['TimeSteps'])//int(arguments['SensorSubSampling'])+1
        LargestArray=max((N1+1)*(N2+1)*(N3+1)*ZoneCount, # stresses, velocities, memory variables
                         N1*N2*N3*10*2, # SqrAcc, up to 10 maps for both RMS and peak
                         int(arguments['IndexSensorMap'].size)*NumberSteps*10, # SensorOutput
                         int(arguments['SourceFunctions'].size))
        return LargestArray >= 2**32

    def _FoldReflectorMask(self, arguments):
        '''
        Reflector cells are marked in the material map instead of being read from ReflectorMask in every step:
        the material property tables are duplicated and reflector cells use the copy (MaterialID + NumberMaterials).
        Material properties (including those used when averaging with neighbour cells) are unchanged, and the
        kernels detect a reflector with MaterialID >= REFLECTOR_MATERIAL_START, saving one memory read per cell.
        The arrays of the caller are not modified; folded copies are used when uploading to the GPU.
        '''
        Tables = ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH']
        NumberMaterials = int(arguments['MiuMatOverH'].size)
        assert all(arguments[k].size == NumberMaterials for k in Tables)
        self.ReflectorMaterialStart = NumberMaterials
        self._HostOverride = {}
        Reflector = np.asarray(arguments['ReflectorMask']) == 1
        if not np.any(Reflector):
            return
        MaterialMap = np.array(arguments['MaterialMap'])
        assert int(MaterialMap.max()) < NumberMaterials
        for k in Tables:
            Values = arguments[k].ravel()
            self._HostOverride[k] = np.concatenate([Values, Values])
        N1, N2, N3 = Reflector.shape
        # first dimension of the material map is the SPP zone (if present); all zones share the reflector mask
        Zones = MaterialMap.reshape((-1,) + MaterialMap.shape[-3:])
        Zones[:, :N1, :N2, :N3][:, Reflector] += NumberMaterials
        self._HostOverride['MaterialMap'] = MaterialMap

    def _HostArray(self, Name, Arrays):
        return self._HostOverride.get(Name, Arrays[Name])

    def _ReflectorDefinitions(self):
        return ["#define METAL_REFLECTOR_IN_MATERIALS\n",
                "#define REFLECTOR_MATERIAL_START %iU\n" % self.ReflectorMaterialStart]

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

        for k in ['Vx','Vy','Vz','Sigmaxx','Sigmayy','Sigmazz','Sigmaxy','Sigmaxz','Sigmayz','Pressure']:
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
        # views (no copies) with the dimension order of the results:
        # SensorOutput (sensor, time, map), SqrAcc (N1, N2, N3, map, RMS/peak) and Snapshots (N1, N2, snapshot)
        self.Results = ArrayResCPU['SensorOutput'].transpose(2,1,0),\
                {'Vx':ArrayResCPU['Vx'],\
                'Vy':ArrayResCPU['Vy'],\
                'Vz':ArrayResCPU['Vz'],\
                'Sigma_xx':ArrayResCPU['Sigma_xx'],\
                'Sigma_yy':ArrayResCPU['Sigma_yy'],\
                'Sigma_zz':ArrayResCPU['Sigma_zz'],\
                'Sigma_xy':ArrayResCPU['Sigma_xy'],\
                'Sigma_yz':ArrayResCPU['Sigma_yz'],\
                'Pressure':ArrayResCPU['Pressure']},\
                ArrayResCPU['SqrAcc'].transpose(2,3,4,1,0),ArrayResCPU['Snapshots'].transpose(1,2,0) 