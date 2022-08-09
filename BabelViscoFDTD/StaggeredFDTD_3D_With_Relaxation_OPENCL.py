from asyncio.windows_events import NULL
import numpy as np
import os
os.environ['GPU_FORCE_64BIT_PTR'] ="1"
from pathlib import Path

import pyopencl as cl
import time
from shutil import copyfile
import tempfile

import StaggeredFDTD_3D_With_Relaxation_BASE

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc
import platform

if platform.system() =='Darwin' and 'arm64' not in platform.platform():
    import _FDTDStaggered3D_with_relaxation_OPENCL_single as FDTD_single;
    import _FDTDStaggered3D_with_relaxation_OPENCL_double as FDTD_double;

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

NumberSelRMSPeakMaps=0
NumberSelSensorMaps=0
TotalAllocs=0
AllC=''

class StaggeredFDTD_3D_With_Relaxation_OPENCL(StaggeredFDTD_3D_With_Relaxation_BASE):
    def __init__(self, arguments):
        extra_params = {"BACKEND":"OPENCL"}
        super().__init__(arguments)

    def _PostInitScript(self, arguments):
        pass

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):
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
        
    def _ownGpuCalloc(self, Name,ctx,td,dims,ArraysGPUOp,flags=cl.mem_flags.READ_WRITE):
        global TotalAllocs
        if td in ['float','unsigned int']:
            f=4
        else: # double
            f=8
        print('Allocating for',Name,dims,'elements')
        ArraysGPUOp[Name]=cl.Buffer(ctx, flags,size=dims*f)
        TotalAllocs+=1            

    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR):
        global TotalAllocs
        print('Allocating for',Name,ArrayResCPU[Name].size,'elements')
        ArraysGPUOp[Name]=cl.Buffer(ctx, flags,hostbuf=ArrayResCPU[Name])
        TotalAllocs+=1

    def _OpenCL_86_64(self, arguments):
        handle, kernelfile = tempfile.mkstemp(suffix='.cu',dir=os.getcwd(), text=True)
        with os.fdopen(handle,'w') as ft:
            ft.write(AllC)
        handle, kernbinfile = tempfile.mkstemp(suffix='.BIN',dir=os.getcwd())
        os.close(handle)
        arguments['kernelfile']=kernelfile
        arguments['kernbinfile']=kernbinfile
        if arguments['DT'].dtype==np.dtype('float32'):
            Results= FDTD_single.FDTDStaggered_3D(arguments)
        else:
            Results= FDTD_double.FDTDStaggered_3D(arguments)
        os.remove(kernelfile) 
        if os.path.isfile(kernbinfile):
            os.remove(kernbinfile) 
        return Results

    def _Execution(self, arguments):
        Results=self._StaggeredFDTD_3D_OPENCL_pyopenCL(arguments)
        return Results

    def _StaggeredFDTD_3D_OPENCL_pyopenCL(self, arguments,dtype=np.float32):
        print('Running OpenCL version via pyopencl')
        global NumberSelRMSPeakMaps
        global NumberSelSensorMaps
        global TotalAllocs
        global AllC
        
        NumberSelRMSPeakMaps=0
        NumberSelSensorMaps=0
        TotalAllocs=0

        IncludeDir=str(Path(__file__).parent.absolute())+os.sep
        gpu_kernelSrc=IncludeDir+'_gpu_kernel.c'
        index_src=IncludeDir+'_indexing.h'


        td = 'float'
        if dtype is np.float64:
            td='double'
        t0=time.time()
        
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

        ctx=cl.Context([device])
        queue = cl.CommandQueue(ctx)
        
        outparams=self.PrepParamsForKernel(arguments)

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
        assert(len(_IndexDataKernel)==54)
        
        ArrayResCPU={}
        for k in ['Sigma_xx','Sigma_yy','Sigma_zz','Pressure']:
            ArrayResCPU[k]=np.zeros((N1,N2,N3),dtype,order='F')
        for k in ['Sigma_xy','Sigma_xz','Sigma_yz']:
            ArrayResCPU[k]=np.zeros((N1+1,N2+1,N3+1),dtype,order='F')
        ArrayResCPU['Vx']=np.zeros((N1+1,N2,N3),dtype,order='F')
        ArrayResCPU['Vy']=np.zeros((N1,N2+1,N3),dtype,order='F')
        ArrayResCPU['Vz']=np.zeros((N1,N2,N3+1),dtype,order='F')

        if 	(arguments['SelRMSorPeak'] &  MASKID['SEL_PEAK']) and (arguments['SelRMSorPeak'] &  MASKID['SEL_RMS']):
            #both peak and RMS
            updims=2 
        else:
            updims=1

        ArrayResCPU['SqrAcc']=np.zeros((N1,N2,N3,NumberSelRMSPeakMaps,updims),dtype,order='F')
        Ns=1
        NumberSnapshots=arguments['SnapshotsPos'].size
        NumberSensors=arguments['IndexSensorMap'].size
        if NumberSnapshots>0:
            Ns=NumberSnapshots
        ArrayResCPU['Snapshots']=np.zeros((N1,N2,Ns),dtype,order='F')
        TimeSteps=arguments['TimeSteps']
        SensorSubSampling=arguments['SensorSubSampling']
        SensorStart=arguments['SensorStart']
        ArrayResCPU['SensorOutput']=np.zeros((NumberSensors,int(TimeSteps/SensorSubSampling)+1-SensorStart,NumberSelSensorMaps),dtype,order='F')

        prg = cl.Program(ctx,AllC).build()
        PartsStress=['MAIN_1']
        AllStressKernels={}
        for k in PartsStress:
            AllStressKernels[k]=getattr(prg,k+"_StressKernel")

        PartsParticle=['MAIN_1']
        AllParticleKernels={}
        for k in PartsParticle:
            AllParticleKernels[k]=getattr(prg,k+"_ParticleKernel")
        
        SensorsKernel=prg.SensorsKernel
        
        ArraysGPUOp={}

        for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                'Ox','Oy','Oz','SourceFunctions','IndexSensorMap','SourceMap','MaterialMap']:
            self._CreateAndCopyFromMXVarOnGPU(k,ctx,ArraysGPUOp,arguments)
        for k in ['V_x_x','V_y_x','V_z_x']:
            self._ownGpuCalloc(k,ctx,td,outparams['SizePMLxp1'],ArraysGPUOp)
        for k in ['V_x_y','V_y_y','V_z_y']:
            self._ownGpuCalloc(k,ctx,td,outparams['SizePMLyp1'],ArraysGPUOp)
        for k in ['V_x_z','V_y_z','V_z_z']:
            self._ownGpuCalloc(k,ctx,td,outparams['SizePMLzp1'],ArraysGPUOp)
        for k in ['Sigma_x_xx','Sigma_y_xx','Sigma_z_xx','Sigma_x_yy','Sigma_y_yy','Sigma_z_yy','Sigma_x_zz','Sigma_y_zz','Sigma_z_zz']:
            self._ownGpuCalloc(k,ctx,td,outparams['SizePML'],ArraysGPUOp)
        for k in ['Sigma_x_xy','Sigma_y_xy','Sigma_x_xz','Sigma_z_xz','Sigma_y_yz','Sigma_z_yz']:
            self._ownGpuCalloc(k,ctx,td,outparams['SizePMLxp1yp1zp1'],ArraysGPUOp)
        for k in ['Rxx','Ryy','Rzz']:
            self._ownGpuCalloc(k,ctx,td,ArrayResCPU['Sigma_xx'].size,ArraysGPUOp)
        for k in ['Rxy','Rxz','Ryz']:
            self._ownGpuCalloc(k,ctx,td,ArrayResCPU['Sigma_xy'].size,ArraysGPUOp)
        for k in ['Vx','Vy','Vz','Sigma_xx','Sigma_yy','Sigma_zz','Pressure','Sigma_xy','Sigma_xz','Sigma_yz','Snapshots','SensorOutput','SqrAcc']:
            self._ownGpuCalloc(k,ctx,td,ArrayResCPU[k].size,ArraysGPUOp)
            
        for n,k in enumerate(_IndexDataKernel):
            for k2 in AllStressKernels:
                AllStressKernels[k2].set_arg(n,ArraysGPUOp[k])
            for k2 in AllParticleKernels:
                AllParticleKernels[k2].set_arg(n,ArraysGPUOp[k])

            SensorsKernel.set_arg(n,ArraysGPUOp[k])
        SensorsKernel.set_arg(54,ArraysGPUOp['SensorOutput'])
        SensorsKernel.set_arg(55,ArraysGPUOp['IndexSensorMap'])

        if arguments['ManualGroupSize'][0]!=-1:
            GroupSize=(arguments['ManualGroupSize'][0],arguments['ManualGroupSize'][1],arguments['ManualGroupSize'][2])
        else:
            GroupSize=(N1,N2,N3)

        AllGroupSizes={}
        AllGroupSizes['MAIN_1']=GroupSize
        

        if arguments['ManualLocalSize'][0]!=-1:
            LocalSize=(arguments['ManualLocalSize'][0],arguments['ManualLocalSize'][1],arguments['ManualLocalSize'][2])
        else:
            LocalSize=None

        for nStep in range(TimeSteps):
            for k in AllStressKernels:
                AllStressKernels[k].set_arg(54,np.uint32(nStep))
                AllStressKernels[k].set_arg(55,arguments['TypeSource'])
            for k in AllParticleKernels:
                AllParticleKernels[k].set_arg(54,np.uint32(nStep))
                AllParticleKernels[k].set_arg(55,arguments['TypeSource'])
            for k in AllStressKernels:
                ev = cl.enqueue_nd_range_kernel(queue, AllStressKernels[k], AllGroupSizes[k], LocalSize)
            # queue.finish()
            for k in AllParticleKernels:
                ev = cl.enqueue_nd_range_kernel(queue, AllParticleKernels[k], AllGroupSizes[k], LocalSize)
            # queue.finish()
            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):
                SensorsKernel.set_arg(56,np.uint32(nStep))
                ev = cl.enqueue_nd_range_kernel(queue, SensorsKernel, (NumberSensors,1), None)
            queue.finish()


        bFirstCopy=True
        events=[]
        for k in ['SqrAcc','Vx','Vy','Vz','Sigma_xx','Sigma_yy','Sigma_zz',
            'Sigma_xy','Sigma_xz','Sigma_yz','Pressure','Snapshots','SensorOutput']:
            events.append(cl.enqueue_copy(queue,ArrayResCPU[k] , ArraysGPUOp[k]))
            
        cl.wait_for_events(events)
        queue.finish()
        
        
        for k in ArraysGPUOp:
            ArraysGPUOp[k].release()
            
        t1=time.time()
        print('time to do low level calculations',t1-t0)
        
        return ArrayResCPU['SensorOutput'],\
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
    return StaggeredFDTD_3D_With_Relaxation_OPENCL(arguments)

