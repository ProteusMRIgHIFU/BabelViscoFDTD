import numpy as np
import os
os.environ['GPU_FORCE_64BIT_PTR'] ="1"
from pathlib import Path

import pyopencl as cl
import time
from shutil import copyfile
import tempfile

from .StaggeredFDTD_3D_With_Relaxation_BASE import StaggeredFDTD_3D_With_Relaxation_BASE

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc
import platform

TotalAllocs=0
AllC=''

def ListDevices():
    devicesIDs=[]
    platforms = cl.get_platforms()
    devices=platforms[0].get_devices() 
    for device in devices:
        devicesIDs.append(device.name)
    return devicesIDs

class StaggeredFDTD_3D_With_Relaxation_OPENCL(StaggeredFDTD_3D_With_Relaxation_BASE):
    def __init__(self, arguments):
        extra_params = {"BACKEND":"OPENCL"}
        super().__init__(arguments, extra_params)

    def _PostInitScript(self, arguments, extra_params):
        global TotalAllocs
        TotalAllocs=0
        SCode = []
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
        SCode.append("#define mexType " + extra_params['td'] +"\n")
        SCode.append("#define OPENCL\n")
        extra_params['SCode'] = SCode

        self.ctx=cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)


    def _InitSymbol(self, IP,_NameVar,td,SCode):
        if td in ['float','double']:
            res = '__constant ' + td  + ' ' + _NameVar + ' = %0.9g;\n' %(IP[_NameVar])
        else:
            lType =' _PT '
            res = '__constant '+ lType  + _NameVar + ' = %i;\n' %(IP[_NameVar])
        SCode.append(res)
        
    def _InitSymbolArray(self, IP,_NameVar,td,SCode):
        res =  "__constant "+ td + " gpu" + _NameVar + "pr[%i] ={\n" % (IP[_NameVar].size)
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
        
    def _ownGpuCalloc(self, Name,td,dims,ArraysGPUOp,flags=cl.mem_flags.READ_WRITE):
        global TotalAllocs
        if td in ['float','unsigned int']:
            f=4
        else: # double
            f=8
        print('Allocating for',Name,dims,'elements')
        ArraysGPUOp[Name]=cl.Buffer(self.ctx, flags,size=dims*f)
        TotalAllocs+=1            

    def _CreateAndCopyFromMXVarOnGPU(self, Name,ArraysGPUOp,ArrayResCPU, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR):
        global TotalAllocs
        print('Allocating for',Name,ArrayResCPU[Name].size,'elements')
        ArraysGPUOp[Name]=cl.Buffer(self.ctx, flags,hostbuf=ArrayResCPU[Name])
        TotalAllocs+=1

    def _Execution(self, arguments, ArrayResCPU, ArraysGPUOp):
        TimeSteps=arguments['TimeSteps']
        NumberSensors=arguments['IndexSensorMap'].size

        for nStep in range(TimeSteps):
            for k in self.AllStressKernels:
                self.AllStressKernels[k].set_arg(54,np.uint32(nStep))
                self.AllStressKernels[k].set_arg(55,arguments['TypeSource'])
            for k in self.AllParticleKernels:
                self.AllParticleKernels[k].set_arg(54,np.uint32(nStep))
                self.AllParticleKernels[k].set_arg(55,arguments['TypeSource'])
            for k in self.AllStressKernels:
                ev = cl.enqueue_nd_range_kernel(self.queue, self.AllStressKernels[k], self.AllGroupSizes[k], self.LocalSize)
            # queue.finish()
            for k in self.AllParticleKernels:
                ev = cl.enqueue_nd_range_kernel(self.queue, self.AllParticleKernels[k], self.AllGroupSizes[k], self.LocalSize)
            # queue.finish()
            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):
                self.SensorsKernel.set_arg(56,np.uint32(nStep))
                ev = cl.enqueue_nd_range_kernel(self.queue, self.SensorsKernel, (NumberSensors,1), None)
            self.queue.finish()


        bFirstCopy=True
        events=[]
        for k in ['Vx','Vy','Vz','Sigma_xx','Sigma_yy','Sigma_zz',
            'Sigma_xy','Sigma_xz','Sigma_yz','Pressure']:
            sz=ArrayResCPU[k].shape
            tempArray=np.zeros((sz[0],sz[1],sz[2],arguments['SPP_ZONES']),dtype=ArrayResCPU[k].dtype,order='F')
            events.append(cl.enqueue_copy(self.queue, tempArray, ArraysGPUOp[k]))
            cl.wait_for_events(events)
            ArrayResCPU[k][:,:,:]=tempArray.sum(axis=3)/arguments['SPP_ZONES']
            
        for k in ['SqrAcc','Snapshots','SensorOutput']:
            events.append(cl.enqueue_copy(self.queue, ArrayResCPU[k], ArraysGPUOp[k]))
        cl.wait_for_events(events)
          
        self.queue.finish()
        
        
        for k in ArraysGPUOp:
            ArraysGPUOp[k].release()


    def _PreExecuteScript(self, arguments, ArraysGPUOp, dummy):
        self.TimeSteps = arguments['TimeSteps']
        N1=arguments['N1']
        N2=arguments['N2']
        N3=arguments['N3']
        _IndexDataKernel=["V_x_x",
            "V_y_x",
            "V_z_x",
            "V_x_y",
            "V_y_y",
            "V_z_y",
            "V_x_z",
            "V_y_z",
            "V_z_z",
            "Vx",
            "Vy",
            "Vz",
            "Rxx",
            "Ryy",
            "Rzz",
            "Rxy",
            "Rxz",
            "Ryz",
            "Sigma_x_xx",
            "Sigma_y_xx",
            "Sigma_z_xx",
            "Sigma_x_yy",
            "Sigma_y_yy",
            "Sigma_z_yy",
            "Sigma_x_zz",
            "Sigma_y_zz",
            "Sigma_z_zz",
            "Sigma_x_xy",
            "Sigma_y_xy",
            "Sigma_x_xz",
            "Sigma_z_xz",
            "Sigma_y_yz",
            "Sigma_z_yz",
            "Sigma_xy",
            "Sigma_xz",
            "Sigma_yz",
            "Sigma_xx",
            "Sigma_yy",
            "Sigma_zz",
            "SourceFunctions",
            "LambdaMiuMatOverH",
            "LambdaMatOverH",
            "MiuMatOverH",
            "TauLong",
            "OneOverTauSigma",
            "TauShear",
            "InvRhoMatH",
            "SqrAcc",
            "MaterialMap",
            "SourceMap",
            "Ox",
            "Oy",
            "Oz",
            "Pressure"]
        assert len(_IndexDataKernel)==54
        
        for n,k in enumerate(_IndexDataKernel):
            for k2 in self.AllStressKernels:
                self.AllStressKernels[k2].set_arg(n,ArraysGPUOp[k])
            for k2 in self.AllParticleKernels:
                self.AllParticleKernels[k2].set_arg(n,ArraysGPUOp[k])

            self.SensorsKernel.set_arg(n,ArraysGPUOp[k])
        self.SensorsKernel.set_arg(54,ArraysGPUOp['SensorOutput'])
        self.SensorsKernel.set_arg(55,ArraysGPUOp['IndexSensorMap'])

        if arguments['ManualGroupSize'][0]!=-1:
            GroupSize=(arguments['ManualGroupSize'][0],arguments['ManualGroupSize'][1],arguments['ManualGroupSize'][2])
        else:
            GroupSize=(N1,N2,N3)

        self.AllGroupSizes={}
        self.AllGroupSizes['MAIN_1']=GroupSize
        

        if arguments['ManualLocalSize'][0]!=-1:
            self.LocalSize=(arguments['ManualLocalSize'][0],arguments['ManualLocalSize'][1],arguments['ManualLocalSize'][2])
        else:
            self.LocalSize=None


    def _InitiateCommands(self, AllC):
        prg = cl.Program(self.ctx,AllC).build()
        PartsStress=['MAIN_1']
        self.AllStressKernels={}
        for k in PartsStress:
            self.AllStressKernels[k]=getattr(prg,k+"_StressKernel")

        PartsParticle=['MAIN_1']
        self.AllParticleKernels={}
        for k in PartsParticle:
            self.AllParticleKernels[k]=getattr(prg,k+"_ParticleKernel")
        
        self.SensorsKernel=prg.SensorsKernel
        
    def CreateResults(self, ArrayResCPU):
        self.Results = ArrayResCPU['SensorOutput'],\
                {'Vx':ArrayResCPU['Vx'],\
                'Vy':ArrayResCPU['Vy'],\
                'Vz':ArrayResCPU['Vz'],\
                'Sigma_xx':ArrayResCPU['Sigma_xx'],\
                'Sigma_yy':ArrayResCPU['Sigma_yy'],\
                'Sigma_zz':ArrayResCPU['Sigma_zz'],\
                'Sigma_xy':ArrayResCPU['Sigma_xy'],\
                'Sigma_yz':ArrayResCPU['Sigma_yz'],\
                'Pressure':ArrayResCPU['Pressure']},\
                ArrayResCPU['SqrAcc'],ArrayResCPU['Snapshots']  



def StaggeredFDTD_3D_OPENCL(arguments):
    Instance = StaggeredFDTD_3D_With_Relaxation_OPENCL(arguments)
    return Instance.Results
