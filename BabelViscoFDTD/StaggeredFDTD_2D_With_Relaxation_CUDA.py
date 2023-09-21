import numpy as np
import cupy as cp

import time

from .StaggeredFDTD_2D_With_Relaxation_BASE import StaggeredFDTD_2D_With_Relaxation_BASE

import struct
TotalAllocs=0

def ListDevices():
    devCount = cp.cuda.runtime.getDeviceCount()
    devicesIDs=[]
    for deviceID in range(0, devCount):
        d=cp.cuda.runtime.getDeviceProperties(deviceID)
        devicesIDs.append(d['name'].decode('UTF-8'))
    return devicesIDs

class StaggeredFDTD_2D_With_Relaxation_CUDA(StaggeredFDTD_2D_With_Relaxation_BASE):
    def __init__(self, arguments):
        extra_params = {"BACKEND":"CUDA"}
        super().__init__(arguments, extra_params)

    def _PostInitScript(self, arguments, extra_params):
        global TotalAllocs
        global _bCudaInit
        TotalAllocs=0
        devCount = cp.cuda.runtime.getDeviceCount()
        print("Number of CUDA devices found:", devCount)
        if devCount == 0:
            raise SystemError("There are no CUDA devices.")
        
        devicelist = {}

        for deviceID in range(0, devCount):
            d=cp.cuda.runtime.getDeviceProperties(deviceID)
            print("Found device", str(deviceID + 1) + "/" + str(devCount),d['name'].decode('UTF-8'))
            device = cp.cuda.Device(deviceID)
            if arguments['DefaultGPUDeviceName'] in d['name'].decode('UTF-8'):
                devicelist[deviceID] = device
    
        if len(devicelist) == 0:
            raise SystemError("There are no devices supporting CUDA or that matches selected device.")
        elif len(devicelist) < arguments['DefaultGPUDeviceNumber']:
            print("The requested device,", arguments['DefaultGPUDeviceNumber'], "(0-base index) is more than the number of devices available", len(devicelist))
            raise IndexError("Unable to select requested device.")

        print("Selecting device", arguments['DefaultGPUDeviceName'], "with number: ", arguments['DefaultGPUDeviceNumber'])

        self.device = devicelist[int(arguments['DefaultGPUDeviceNumber'])]
        self.context = cp.cuda.Device(self.device)
        print("Context Created!")
        self.context.use()
        SCode=[]
        SCode.append("#define mexType " + extra_params['td'] +"\n")
        SCode.append("#define CUDA\n")
        extra_params['SCode'] = SCode
        

    def _InitiateCommands(self, AllC):
        self._prgcuda  = cp.RawModule(code=AllC)

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

        if td == 'float':
            dt=cp.float32
        elif td =='unsigned int':
            dt=cp.uint32
        else: # double
            dt=cp.float64
        print('Allocating for',Name,dims,'elements')
        ArraysGPUOp[Name]=cp.zeros((dims),dt)
        TotalAllocs+=1   
    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ArraysGPUOp,ArrayResCPU,flags=None):
        global TotalAllocs
        print('Allocating for',Name,ArrayResCPU[Name].size,'elements')
        ArraysGPUOp[Name]=cp.asarray(ArrayResCPU[Name])
        TotalAllocs+=1

    def _Execution(self, arguments, ArrayResCPU, ArraysGPUOp):
        TimeSteps=arguments['TimeSteps']
        NumberSensors=arguments['IndexSensorMap'].size

        for nStep in range(TimeSteps):
            for AllKrnl in [self.AllStressKernels,self.AllParticleKernels]:
                for k in AllKrnl:
                    AllKrnl[k](self.AllGridSizes[k],
                              self.AllBlockSizes[k],
                        (ArraysGPUOp["V_x_x"],
                        ArraysGPUOp["V_y_x"],
                        ArraysGPUOp["V_x_y"],
                        ArraysGPUOp["V_y_y"],
                        ArraysGPUOp["Vx"],
                        ArraysGPUOp["Vy"],
                        ArraysGPUOp["Rxx"],
                        ArraysGPUOp["Ryy"],
                        ArraysGPUOp["Rxy"],
                        ArraysGPUOp["Sigma_x_xx"],
                        ArraysGPUOp["Sigma_y_xx"],
                        ArraysGPUOp["Sigma_x_yy"],
                        ArraysGPUOp["Sigma_y_yy"],
                        ArraysGPUOp["Sigma_x_xy"],
                        ArraysGPUOp["Sigma_y_xy"],
                        ArraysGPUOp["Sigma_xy"],
                        ArraysGPUOp["Sigma_xx"],
                        ArraysGPUOp["Sigma_yy"],
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
                        ArraysGPUOp["Pressure"],
                        nStep,
                        arguments['TypeSource'])
                        )
                    cp.cuda.Stream.null.synchronize() 

            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):
                self.SensorsKernel(self.GridSensors,
                        self.BlockSensors,
                        (ArraysGPUOp["V_x_x"],
                        ArraysGPUOp["V_y_x"],
                        ArraysGPUOp["V_x_y"],
                        ArraysGPUOp["V_y_y"],
                        ArraysGPUOp["Vx"],
                        ArraysGPUOp["Vy"],
                        ArraysGPUOp["Rxx"],
                        ArraysGPUOp["Ryy"],
                        ArraysGPUOp["Rxy"],
                        ArraysGPUOp["Sigma_x_xx"],
                        ArraysGPUOp["Sigma_y_xx"],
                        ArraysGPUOp["Sigma_x_yy"],
                        ArraysGPUOp["Sigma_y_yy"],
                        ArraysGPUOp["Sigma_x_xy"],
                        ArraysGPUOp["Sigma_y_xy"],
                        ArraysGPUOp["Sigma_xy"],
                        ArraysGPUOp["Sigma_xx"],
                        ArraysGPUOp["Sigma_yy"],
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
                        ArraysGPUOp["Pressure"],
                        ArraysGPUOp['SensorOutput'],
                        ArraysGPUOp['IndexSensorMap'],
                        nStep))
                cp.cuda.Stream.null.synchronize()

        bFirstCopy=True
        events=[]
        for k in ['Vx','Vy','Sigma_xx','Sigma_yy',
            'Sigma_xy','Pressure']:
            sz=ArrayResCPU[k].shape
            tempArray=ArraysGPUOp[k].get().reshape((sz[0],sz[1],arguments['SPP_ZONES']), order='F')
            ArrayResCPU[k][:,:]=tempArray.sum(axis=2)/arguments['SPP_ZONES']

        for k in ['SqrAcc','Snapshots','SensorOutput']:
            ArrayResCPU[k]=ArraysGPUOp[k].get()
            
        cp.cuda.Stream.null.synchronize()
        
        for k in list(ArraysGPUOp.keys()):
            h= ArraysGPUOp.pop(k)
            del h


    def _PreExecuteScript(self, arguments, ArraysGPUOp, dummy):
        self.TimeSteps = arguments['TimeSteps']
        N1=arguments['N1']
        N2=arguments['N2']
        
        if arguments['ManualLocalSize'][0]!=-1:
            AllBlockSizes=(arguments['ManualLocalSize'][0],
                           arguments['ManualLocalSize'][1])
        else:
            AllBlockSizes=(8,2)

        self.AllBlockSizes={}
        self.AllBlockSizes['MAIN_1']=AllBlockSizes
        self.AllGridSizes={}

        if arguments['ManualGroupSize'][0]!=-1:
            self.AllGridSizes['MAIN_1']=(arguments['ManualGroupSize'][0],
                            arguments['ManualGroupSize'][1])
        else:
            self.AllGridSizes['MAIN_1']=(int(N1//self.AllBlockSizes['MAIN_1'][0]+1),
                              int(N2//self.AllBlockSizes['MAIN_1'][1]+1))

        self.BlockSensors=(64,1)
        self.GridSensors=(int(arguments['IndexSensorMap'].size//self.BlockSensors[0]+1),1)


def StaggeredFDTD_2D_CUDA(arguments):
    Instance = StaggeredFDTD_2D_With_Relaxation_CUDA(arguments)
    return Instance.Results

