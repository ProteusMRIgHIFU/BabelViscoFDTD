import numpy as np
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

import time

from .StaggeredFDTD_3D_With_Relaxation_BASE import StaggeredFDTD_3D_With_Relaxation_BASE

import struct
import pycuda.driver as cuda
TotalAllocs=0
class StaggeredFDTD_3D_With_Relaxation_CUDA(StaggeredFDTD_3D_With_Relaxation_BASE):
    def __init__(self, arguments):
        extra_params = {"BACKEND":"CUDA"}
        super().__init__(arguments, extra_params)

    def _PostInitScript(self, arguments, extra_params):
        global TotalAllocs
        TotalAllocs=0
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
        # self.context.set_cache_config(cuda.func_cache.PREFER_L1)
        SCode=[]
        SCode.append("#define mexType " + extra_params['td'] +"\n")
        SCode.append("#define CUDA\n")
        extra_params['SCode'] = SCode

    def _InitiateCommands(self, AllC):
        self._prgcuda  = SourceModule(AllC)#,include_dirs=[npyInc+os.sep+'numpy',info['include']])

        PartsStress=['MAIN_1']
        self.AllStressKernels={}
        for k in PartsStress:
            self.AllStressKernels[k]=self._prgcuda.get_function(k+"_StressKernel")

        PartsParticle=['MAIN_1']
        self.AllParticleKernels={}
        for k in PartsParticle:
            self.AllParticleKernels[k]=self._prgcuda.get_function(k+"_ParticleKernel")
        
        self.SensorsKernel=self._prgcuda.get_function("SensorsKernel")

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):

        if td in ['float','double']:
            res = '__constant__ ' + td  + ' ' + _NameVar + ' = %0.9g;\n' %(IP[_NameVar])
        else:
            lType =' _PT '
            res = '__constant__ '+ lType  + _NameVar + ' = %i;\n' %(IP[_NameVar])
        SCode.append(res)    
        
    def _InitSymbolArray(self, IP,_NameVar,td,SCode=[]):
        res =  "__constant__ "+ td + " gpu" + _NameVar + "pr[%i] ={\n" % (IP[_NameVar].size)
        for n in range(IP[_NameVar].size):
            if td in ['float','double']:
                res+="%.9g" % (IP[_NameVar][n])
            else:
                res+="%i" % (IP[_NameVar][n])
            if n<IP[_NameVar].size-1:
                res+=',\n'
            else:
                res+='};\n'
        SCode.append(res)   

    def _ownGpuCalloc(self, Name,td,dims,ArraysGPUOp,flags=None):
        global TotalAllocs
        if td in ['float','unsigned int']:
            f=4
        else: # double
            f=8
        print('Allocating for',Name,dims,'elements')
        ArraysGPUOp[Name]=cuda.mem_alloc(int(dims*f))
        TotalAllocs+=1   
    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ArraysGPUOp,ArrayResCPU,flags=None):
        global TotalAllocs
        print('Allocating for',Name,ArrayResCPU[Name].size,'elements')
        ArraysGPUOp[Name]=cuda.mem_alloc(ArrayResCPU[Name].nbytes)
        cuda.memcpy_htod(ArraysGPUOp[Name], ArrayResCPU[Name])
        TotalAllocs+=1

    def _Execution(self, arguments, ArrayResCPU, ArraysGPUOp):
        TimeSteps=arguments['TimeSteps']
        NumberSensors=arguments['IndexSensorMap'].size

        for nStep in range(TimeSteps):
            for AllKrnl in [self.AllStressKernels,self.AllParticleKernels]:
                for k in AllKrnl:
                    AllKrnl[k](ArraysGPUOp["V_x_x"],
                        ArraysGPUOp["V_y_x"],
                        ArraysGPUOp["V_z_x"],
                        ArraysGPUOp["V_x_y"],
                        ArraysGPUOp["V_y_y"],
                        ArraysGPUOp["V_z_y"],
                        ArraysGPUOp["V_x_z"],
                        ArraysGPUOp["V_y_z"],
                        ArraysGPUOp["V_z_z"],
                        ArraysGPUOp["Vx"],
                        ArraysGPUOp["Vy"],
                        ArraysGPUOp["Vz"],
                        ArraysGPUOp["Rxx"],
                        ArraysGPUOp["Ryy"],
                        ArraysGPUOp["Rzz"],
                        ArraysGPUOp["Rxy"],
                        ArraysGPUOp["Rxz"],
                        ArraysGPUOp["Ryz"],
                        ArraysGPUOp["Sigma_x_xx"],
                        ArraysGPUOp["Sigma_y_xx"],
                        ArraysGPUOp["Sigma_z_xx"],
                        ArraysGPUOp["Sigma_x_yy"],
                        ArraysGPUOp["Sigma_y_yy"],
                        ArraysGPUOp["Sigma_z_yy"],
                        ArraysGPUOp["Sigma_x_zz"],
                        ArraysGPUOp["Sigma_y_zz"],
                        ArraysGPUOp["Sigma_z_zz"],
                        ArraysGPUOp["Sigma_x_xy"],
                        ArraysGPUOp["Sigma_y_xy"],
                        ArraysGPUOp["Sigma_x_xz"],
                        ArraysGPUOp["Sigma_z_xz"],
                        ArraysGPUOp["Sigma_y_yz"],
                        ArraysGPUOp["Sigma_z_yz"],
                        ArraysGPUOp["Sigma_xy"],
                        ArraysGPUOp["Sigma_xz"],
                        ArraysGPUOp["Sigma_yz"],
                        ArraysGPUOp["Sigma_xx"],
                        ArraysGPUOp["Sigma_yy"],
                        ArraysGPUOp["Sigma_zz"],
                        ArraysGPUOp["SourceFunctions"],
                        ArraysGPUOp["LambdaMiuMatOverH"],
                        ArraysGPUOp["LambdaMatOverH"],
                        ArraysGPUOp["MiuMatOverH"],
                        ArraysGPUOp["TauLong"],
                        ArraysGPUOp["OneOverTauSigma"],
                        ArraysGPUOp["TauShear"],
                        ArraysGPUOp["InvRhoMatH"],
                        ArraysGPUOp["SqrAcc"],
                        ArraysGPUOp["MaterialMap"],
                        ArraysGPUOp["SourceMap"],
                        ArraysGPUOp["Ox"],
                        ArraysGPUOp["Oy"],
                        ArraysGPUOp["Oz"],
                        ArraysGPUOp["Pressure"],
                        np.uint32(nStep),
                        arguments['TypeSource'],
                        block=self.AllBlockSizes[k],
                        grid=self.AllGridSizes[k])
                    self.context.synchronize()

            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):
                self.SensorsKernel(ArraysGPUOp["V_x_x"],
                        ArraysGPUOp["V_y_x"],
                        ArraysGPUOp["V_z_x"],
                        ArraysGPUOp["V_x_y"],
                        ArraysGPUOp["V_y_y"],
                        ArraysGPUOp["V_z_y"],
                        ArraysGPUOp["V_x_z"],
                        ArraysGPUOp["V_y_z"],
                        ArraysGPUOp["V_z_z"],
                        ArraysGPUOp["Vx"],
                        ArraysGPUOp["Vy"],
                        ArraysGPUOp["Vz"],
                        ArraysGPUOp["Rxx"],
                        ArraysGPUOp["Ryy"],
                        ArraysGPUOp["Rzz"],
                        ArraysGPUOp["Rxy"],
                        ArraysGPUOp["Rxz"],
                        ArraysGPUOp["Ryz"],
                        ArraysGPUOp["Sigma_x_xx"],
                        ArraysGPUOp["Sigma_y_xx"],
                        ArraysGPUOp["Sigma_z_xx"],
                        ArraysGPUOp["Sigma_x_yy"],
                        ArraysGPUOp["Sigma_y_yy"],
                        ArraysGPUOp["Sigma_z_yy"],
                        ArraysGPUOp["Sigma_x_zz"],
                        ArraysGPUOp["Sigma_y_zz"],
                        ArraysGPUOp["Sigma_z_zz"],
                        ArraysGPUOp["Sigma_x_xy"],
                        ArraysGPUOp["Sigma_y_xy"],
                        ArraysGPUOp["Sigma_x_xz"],
                        ArraysGPUOp["Sigma_z_xz"],
                        ArraysGPUOp["Sigma_y_yz"],
                        ArraysGPUOp["Sigma_z_yz"],
                        ArraysGPUOp["Sigma_xy"],
                        ArraysGPUOp["Sigma_xz"],
                        ArraysGPUOp["Sigma_yz"],
                        ArraysGPUOp["Sigma_xx"],
                        ArraysGPUOp["Sigma_yy"],
                        ArraysGPUOp["Sigma_zz"],
                        ArraysGPUOp["SourceFunctions"],
                        ArraysGPUOp["LambdaMiuMatOverH"],
                        ArraysGPUOp["LambdaMatOverH"],
                        ArraysGPUOp["MiuMatOverH"],
                        ArraysGPUOp["TauLong"],
                        ArraysGPUOp["OneOverTauSigma"],
                        ArraysGPUOp["TauShear"],
                        ArraysGPUOp["InvRhoMatH"],
                        ArraysGPUOp["SqrAcc"],
                        ArraysGPUOp["MaterialMap"],
                        ArraysGPUOp["SourceMap"],
                        ArraysGPUOp["Ox"],
                        ArraysGPUOp["Oy"],
                        ArraysGPUOp["Oz"],
                        ArraysGPUOp["Pressure"],
                        ArraysGPUOp['SensorOutput'],
                        ArraysGPUOp['IndexSensorMap'],
                        np.uint32(nStep),
                        block=self.BlockSensors,
                        grid=self.GridSensors)
                self.context.synchronize()

        bFirstCopy=True
        events=[]
        for k in ['Vx','Vy','Vz','Sigma_xx','Sigma_yy','Sigma_zz',
            'Sigma_xy','Sigma_xz','Sigma_yz','Pressure']:
            sz=ArrayResCPU[k].shape
            tempArray=np.zeros((sz[0],sz[1],sz[2],arguments['SPP_ZONES']),dtype=ArrayResCPU[k].dtype,order='F')
            cuda.memcpy_dtoh( tempArray, ArraysGPUOp[k])
            ArrayResCPU[k][:,:,:]=tempArray.sum(axis=3)/arguments['SPP_ZONES']

        for k in ['SqrAcc','Snapshots','SensorOutput']:
            cuda.memcpy_dtoh( ArrayResCPU[k], ArraysGPUOp[k])
            
        self.context.synchronize()
        
        self.context.pop()
        
        for k in ArraysGPUOp:
            ArraysGPUOp[k].free()


    def _PreExecuteScript(self, arguments, ArraysGPUOp, dummy):
        self.TimeSteps = arguments['TimeSteps']
        N1=arguments['N1']
        N2=arguments['N2']
        N3=arguments['N3']
        
        if arguments['ManualLocalSize'][0]!=-1:
            AllBlockSizes=(arguments['ManualLocalSize'][0],
                           arguments['ManualLocalSize'][1],
                           arguments['ManualLocalSize'][2])
        else:
            AllBlockSizes=(8,8,2)

        self.AllBlockSizes={}
        self.AllBlockSizes['MAIN_1']=AllBlockSizes
        self.AllGridSizes={}

        if arguments['ManualGroupSize'][0]!=-1:
            self.AllGridSizes['MAIN_1']=(arguments['ManualGroupSize'][0],
                            arguments['ManualGroupSize'][1],
                            arguments['ManualGroupSize'][2])
        else:
            self.AllGridSizes['MAIN_1']=(int(N1//self.AllBlockSizes['MAIN_1'][0]+1),
                              int(N2//self.AllBlockSizes['MAIN_1'][1]+1),
                              int(N3//self.AllBlockSizes['MAIN_1'][2]+1))

        self.BlockSensors=(64,1,1)
        self.GridSensors=(int(arguments['IndexSensorMap'].size//self.BlockSensors[0]+1),1,1)


def StaggeredFDTD_3D_CUDA(arguments):
    Instance = StaggeredFDTD_3D_With_Relaxation_CUDA(arguments)
    return Instance.Results

