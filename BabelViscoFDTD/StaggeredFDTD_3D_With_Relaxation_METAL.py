from calendar import c
from multiprocessing.dummy import Array
from unicodedata import name
import numpy as np
import os
import re
from pathlib import Path
import platform
import time
import tempfile
from shutil import copyfile

from .StaggeredFDTD_3D_With_Relaxation_BASE import StaggeredFDTD_3D_With_Relaxation_BASE

from distutils.sysconfig import get_python_inc
from math import ceil
import ctypes
from ctypes import c_byte, c_int, c_uint32, c_float, c_wchar_p, c_uint64
import gc

mc = None

def ListDevices():
    import metalcomputebabel 
    devicesIDs=[]
    devices = metalcomputebabel.get_devices()
    for dev in devices:
        devicesIDs.append(dev.deviceName)
    return devicesIDs

class StaggeredFDTD_3D_With_Relaxation_METAL_MetalCompute(StaggeredFDTD_3D_With_Relaxation_BASE):
    '''
    This version is mainly for Mx processors and which is based in a modified+forked version of metalcompute. As X64 will be phased out, in the future the metalcompute version will take over
    '''
    def __init__(self, arguments):
        global mc
        import metalcomputebabel
        mc = metalcomputebabel
        if not hasattr(mc.Device,'arg_buffer'):
            raise ImportError('BabelViscoFDTD requires metalcomputebabel >= 0.2.6 (argument buffers) for the Metal backend')
        # Definition of some constants, etc
        self.MAX_SIZE_PML = 101
        self.ZoneCount = arguments['SPP_ZONES']
        # every array is a separate Metal buffer; the kernels receive them through the argument buffers
        # declared in kernelparamsMetal.h (struct name: access)
        self._ArgumentStructs = {'FDTD_ArgsIn':'r', 'FDTD_ArgsInOut':'rw', 'FDTD_ArgsOut':'w'}
        self._ArraySpecs = {} # name -> (number of entries, dtype, host data to copy or None)
        self.C_IND = {"nStep":0, "TypeSource":1}
        # self.LENGTH_CONST_MEX = 1+self.MAX_SIZE_PML*4

        extra_params = {"BACKEND":"METAL"}
        
        self.bUseSingleKernel = False
        # threads per threadgroup for each kernel (0 = largest size supported by the kernel)
        self.ThreadsPerGroup = {}
        super().__init__(arguments, extra_params)
        
    def _PostInitScript(self, arguments, extra_params):
        print("Attempting Metal Initiation...")
        devices = mc.get_devices()
        SelDevice=None
        for n,dev in enumerate(devices):
            if arguments['DefaultGPUDeviceName'] in dev.deviceName:
                SelDevice=dev
                break
        if SelDevice is None:
            raise SystemError("No Metal device containing name [%s]" %(arguments['DefaultGPUDeviceName']))
        else:
            print('Selecting device: ', dev.deviceName)
        SCode = []
        SCode.append("#define mexType " + extra_params['td'] +"\n")
        SCode.append("#define METAL\n")
        if self.bUseSingleKernel:
            SCode.append("#define METAL_SINGLE_KERNEL\n")
        SCode.append("#define METALCOMPUTE\n")
        if self._RequiresIndex64(arguments):
            print('Using 64-bit indexing in GPU kernels')
            SCode.append("#define METAL_INDEX64\n")
        SCode.append("#define MAX_SIZE_PML 101\n")
        extra_params['SCode'] = SCode
        self._FoldReflectorMask(arguments)
        self.ctx = mc.Device(n)
        print(self.ctx)
        if 'arm64' not in platform.platform():
            print('Setting Metal for External or AMD processor')
            self.ctx.set_external_gpu(1) 
    
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
        MaterialMap = np.array(arguments['MaterialMap'], order='F')
        assert int(MaterialMap.max()) < NumberMaterials
        for k in Tables:
            Values = arguments[k].flatten(order='F')
            self._HostOverride[k] = np.concatenate([Values, Values])
        N1, N2, N3 = Reflector.shape
        # last dimension of the material map is the SPP zone (if present); all zones share the reflector mask
        Zones = MaterialMap.reshape(MaterialMap.shape[0], MaterialMap.shape[1], MaterialMap.shape[2], -1, order='F')
        Zones[:N1, :N2, :N3, :][Reflector] += NumberMaterials
        self._HostOverride['MaterialMap'] = MaterialMap

    def _HostArray(self, Name, Arrays):
        return self._HostOverride.get(Name, Arrays[Name])

    def _InitSymbol(self, IP,_NameVar,td,SCode):
        if td in ['float','double']:
            res = 'constant ' + td  + ' ' + _NameVar + ' = %0.9g;\n' %(IP[_NameVar])
        else:
            lType =' _PT '
            res = 'constant '+ lType  + _NameVar + ' = %i;\n' %(IP[_NameVar])
        SCode.append(res)
        
    def _InitSymbolArray(self, IP,_NameVar,td,SCode):
        res =  "constant "+ td + " " + _NameVar + "_pr[%i] ={\n" % (IP[_NameVar].size)
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


    def _ownGpuCalloc(self, Name,td,dims,ArraysGPUOp):
        if Name == "Snapshots": # not used by the Metal kernels
            return
        print("Allocating for", Name, dims, "elements")
        self._ArraySpecs[Name] = (int(dims), np.float32 if td == "float" else np.uint32, None)

    def _CreateAndCopyFromMXVarOnGPU(self, Name,ArraysGPUOp,ArrayResCPU,flags=[]):
        Host = self._HostArray(Name, ArrayResCPU)
        print("Allocating for", Name, Host.size, "elements")
        if Name in ['IndexSensorMap','SourceMap','MaterialMap','ReflectorMask']:
            dtype = np.uint32
        else:
            dtype = np.float32
        if Name in ['SensorOutput','SqrAcc']: # outputs start at zero
            Host = None
        self._ArraySpecs[Name] = (int(ArrayResCPU[Name].size if Host is None else Host.size), dtype, Host)

    def _PreExecuteScript(self, arguments, ArraysGPUOp, outparams):
        self._outparams = outparams
        self.gpu_arrays = {}
        for Name,(Size,dtype,Host) in self._ArraySpecs.items():
            if Host is None:
                self.gpu_arrays[Name] = self.ctx.buffer(max(Size,1)*np.dtype(dtype).itemsize)
            else:
                self.gpu_arrays[Name] = self.ctx.buffer(np.ascontiguousarray(Host.flatten(order='F'),dtype=dtype))

        self.arg_buffers = []
        Bound = set()
        for Struct,Access in self._ArgumentStructs.items():
            Names = self._ArgumentOrder(Struct)
            self.arg_buffers.append(self.ctx.arg_buffer(tuple(self.gpu_arrays[k] for k in Names),Access))
            Bound.update(Names)
        assert Bound == set(self.gpu_arrays), 'Arrays not declared in the kernel argument structs: ' + str(set(self.gpu_arrays)-Bound)

        self.localSensor = [1, 1, 1]
        self.globalSensor = [ceil(arguments['IndexSensorMap'].size / self.localSensor[0]), 1, 1]

        self._CompileKernels()

    def _ArgumentOrder(self, Struct):
        '''
        Array names of an argument buffer struct of the kernel source, ordered by their [[id(n)]]
        '''
        Source = ''.join(self._KernelSource)
        Body = re.search(r'struct\s+%s\s*\{(.*?)\};' % Struct, Source, re.S)
        if Body is None:
            raise RuntimeError('Argument buffer struct %s not found in the kernel source' % Struct)
        Entries = sorted((int(Id),Name) for Name,Id in re.findall(r'(\w+)\s*\[\[\s*id\((\d+)\)\s*\]\]', Body.group(1)))
        assert [Id for Id,_ in Entries] == list(range(len(Entries))), 'ids of %s must be consecutive from 0' % Struct
        return [Name for _,Name in Entries]

    def _InitiateCommands(self, SCode):
        # Compilation is done by _CompileKernels at the end of _PreExecuteScript
        self._KernelSource = SCode

    def _ReflectorDefinitions(self):
        return ["#define METAL_REFLECTOR_IN_MATERIALS\n",
                "#define REFLECTOR_MATERIAL_START %iU\n" % self.ReflectorMaterialStart]

    def _CompileKernels(self):
        AllC=''.join(self._ReflectorDefinitions() + self._KernelSource)
        prg = self.ctx.kernel(AllC)
        if self.bUseSingleKernel:
            PartsStress=['MAIN_1']
        else:
            # PML_12 replaces PML_1 and PML_2 (both i-slabs in a single kernel, see GPU_KERNELS.h)
            PartsStress=['PML_12','PML_3','PML_4','PML_5','PML_6','MAIN_1']
        
        self.AllStressKernels={}
        for k in PartsStress:
            self.AllStressKernels[k]=prg.function(k+"_StressKernel")

        if self.bUseSingleKernel:
            PartsParticle =['MAIN_1']
        else:
            PartsParticle=['PML_12','PML_3','PML_4','PML_5','PML_6','MAIN_1']
        
        self.AllParticleKernels={}
        for k in PartsParticle:
            self.AllParticleKernels[k]=prg.function(k+"_ParticleKernel")
        
        self.SensorsKernel=prg.function('SensorsKernel')

    def _TimeLoopBatched(self, arguments, kernels, DimsKernel):
        '''
        Time loop encoding many steps per command buffer, so the CPU does not wait for the GPU after every step.
        nStep and TypeSource of every step are stored once in a buffer bound as buffer 0, with each dispatch
        selecting its step by the buffer offset. The CPU encodes the next batch while the GPU runs the current one.
        The batches use a concurrent encoder: the PML and main kernels of each phase update disjoint regions and can
        run simultaneously, with a memory barrier between dependent phases.
        '''
        TimeSteps = int(arguments['TimeSteps'])
        SensorSubSampling = int(arguments['SensorSubSampling'])
        SensorStart = int(arguments['SensorStart'])
        StepStride = 64 # uint32 entries per step (256 bytes), keeps buffer offsets aligned for all GPU families
        StepConstants = np.zeros((TimeSteps,StepStride),np.uint32)
        StepConstants[:,self.C_IND['nStep']] = np.arange(TimeSteps)
        StepConstants[:,self.C_IND['TypeSource']] = int(arguments['TypeSource'])
        self.step_constants = self.ctx.buffer(StepConstants)
        AllBuffers = (self.step_constants,) + tuple(self.arg_buffers)
        SizeKernel = {k:int(np.prod(DimsKernel[k])) for k in kernels}
        NumberSensors = int(np.prod(self.globalSensor))
        StepsPerBatch = 64
        ctx = self.ctx
        TPG = self.ThreadsPerGroup
        Begin = lambda : ctx.batch_begin(AllBuffers,True)
        Dispatch = lambda fn,count,offset,kernel : ctx.batch_dispatch(fn,count,offset,TPG.get(kernel,0))
        Barrier = ctx.batch_barrier

        Begin()
        try:
            for nStep in range(TimeSteps):
                Offset = nStep*StepStride*4
                for i in kernels:
                    Dispatch(self.AllStressKernels[i],SizeKernel[i],Offset,i)
                Barrier()
                for i in kernels:
                    Dispatch(self.AllParticleKernels[i],SizeKernel[i],Offset,i)
                Barrier()
                if (nStep % SensorSubSampling)==0  and (int(nStep/SensorSubSampling)>=SensorStart):
                    Dispatch(self.SensorsKernel,NumberSensors,Offset,'Sensors')
                    Barrier()
                if (nStep+1) % StepsPerBatch == 0 or nStep == TimeSteps-1:
                    self.ctx.batch_commit()
                    self.ctx.batch_wait(1) # at most one batch running while the next one is encoded
                    if nStep < TimeSteps-1:
                        Begin()
            self.ctx.batch_wait(0)
        except BaseException:
            # leave the device without open or running batches before reporting the error
            try:
                self.ctx.batch_commit()
            except mc.error:
                pass
            try:
                self.ctx.batch_wait(0)
            except mc.error:
                pass
            raise
        del self.step_constants

    def _Execution(self, arguments, ArrayResCPU, ArrayResOP):
        outparams=self._outparams
        DimsKernel={}
        DimsKernel['PML_12']=[outparams['PML_Thickness']*2,
                             outparams['N2'],
                             outparams['N3']]
        DimsKernel['PML_3']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness'],
                            outparams['N3']]
        DimsKernel['PML_4']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness'],
                            outparams['N3']]
        DimsKernel['PML_5']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness']]
        DimsKernel['PML_6']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness']]
        if self.bUseSingleKernel:
            DimsKernel['MAIN_1']=[outparams['N1'],
                            outparams['N2'],
                            outparams['N3']]
            kernels=["MAIN_1"]
            
        
        else:
            DimsKernel['MAIN_1']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['N3']-outparams['PML_Thickness']*2]
            kernels =["PML_12", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]
            

        self._TimeLoopBatched(arguments,kernels,DimsKernel)
        if 'arm64' not in platform.platform():
            self.ctx.sync_buffers(tuple(self.gpu_arrays.values()))
        for i in ['SqrAcc', 'SensorOutput']:
            Buffer=np.frombuffer(self.gpu_arrays[i],dtype=np.float32)[:ArrayResCPU[i].size]
            ArrayResCPU[i][:] = np.reshape(Buffer, ArrayResCPU[i].shape,order='F')

        for i in ['Vx', 'Vy', 'Vz', 'Sigma_xx', 'Sigma_yy', 'Sigma_zz', 'Sigma_xy', 'Sigma_xz', 'Sigma_yz', 'Pressure']:
            sz=ArrayResCPU[i].shape
            Buffer=np.frombuffer(self.gpu_arrays[i],dtype=np.float32)[:ArrayResCPU[i].size * self.ZoneCount]
            Buffer=Buffer.reshape((sz[0],sz[1],sz[2],self.ZoneCount),order='F')
            ArrayResCPU[i][:,:,:] = np.sum(Buffer,axis=3)/self.ZoneCount

        del self.arg_buffers
        del self.gpu_arrays
        gc.collect()

def StaggeredFDTD_3D_METAL(arguments):
    Instance = StaggeredFDTD_3D_With_Relaxation_METAL_MetalCompute(arguments)
    Results = Instance.Results
    return Results
