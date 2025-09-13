from calendar import c
from multiprocessing.dummy import Array
from unicodedata import name
import numpy as np
import os
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

mc = None

import mlx.core as mx

def ListDevices():
    import metalcomputebabel 
    devicesIDs=[]
    devices = metalcomputebabel.get_devices()
    for dev in devices:
        devicesIDs.append(dev.deviceName)
    return devicesIDs

class StaggeredFDTD_3D_With_Relaxation_MLX(StaggeredFDTD_3D_With_Relaxation_BASE):
    '''
    This version is mainly for Mx processors and which is based in a modified+forked version of metalcompute. As X64 will be phased out, in the future the metalcompute version will take over
    '''
    def __init__(self, arguments):
        global mc
        
        # Definition of some constants, etc
        self.MAX_SIZE_PML = 101
        self._c_mex_type = np.zeros(12, np.uint64)
        self._c_uint_type = np.uint64(0)
        self.HOST_INDEX_MEX = np.zeros((53, 2), np.uint64)
        self.HOST_INDEX_UINT = np.zeros((3, 2), np.uint64)
        self.LENGTH_INDEX_MEX = 53
        self.LENGTH_INDEX_UINT = 3
        self.ZoneCount = arguments['SPP_ZONES']
        self._IndexDataMetal = {
            "V_x_x":0, "V_y_x":0, "V_z_x":0, "V_x_y":0, "V_y_y":0, "V_z_y":0, "V_x_z":0, "V_y_z":0, "V_z_z":0,
            "Vx":1, "Vy":1, "Vz":1,
            "Rxx":2, "Ryy":2, "Rzz":2,
            "Rxy":3, "Rxz":3, "Ryz":3,
            "Sigma_x_xx":4, "Sigma_y_xx":4, "Sigma_z_xx":4, "Sigma_x_yy":4, "Sigma_y_yy":4, "Sigma_z_yy":4, "Sigma_x_zz":4, "Sigma_y_zz":4,
            "Sigma_z_zz":5, "Sigma_x_xy":5, "Sigma_y_xy":5, "Sigma_x_xz":5, "Sigma_z_xz":5, "Sigma_y_yz":5, "Sigma_z_yz":5,
            "Sigma_xy":6, "Sigma_xz":6, "Sigma_yz":6, 
            "Sigma_xx":7, "Sigma_yy":7, "Sigma_zz":7,
            "SourceFunctions":8,
            "LambdaMiuMatOverH":9, "LambdaMatOverH":9, "MiuMatOverH":9, "TauLong":9, "OneOverTauSigma":9, "TauShear":9, "InvRhoMatH":9, "Ox":9, "Oy":9, "Oz":9, "Pressure":9, 
            "SqrAcc":10,
            "SensorOutput":11
            }
        self.C_IND = {
            "IndexSensorMap":0, "SourceMap":1, "MaterialMap": 2,
            "nStep":0, "TypeSource":1, 
            "V_x_x":0, "V_y_x":1, "V_z_x":2, "V_x_y":3, "V_y_y":4, "V_z_y":5, "V_x_z":6, "V_y_z":7, "V_z_z":8, "Vx":9, "Vy":10, "Vz":11, "Rxx":12, "Ryy":13, "Rzz":14, "Rxy":15, "Rxz":16, "Ryz":17,
            "Sigma_x_xx":18, "Sigma_y_xx":19, "Sigma_z_xx":20, "Sigma_x_yy":21, "Sigma_y_yy":22, "Sigma_z_yy":23, "Sigma_x_zz":24, "Sigma_y_zz":25, "Sigma_z_zz":26, 
            "Sigma_x_xy":27, "Sigma_y_xy":28, "Sigma_x_xz":29, "Sigma_z_xz":30, "Sigma_y_yz":31, "Sigma_z_yz":32, "Sigma_xy":33, "Sigma_xz":34, "Sigma_yz":35, "Sigma_xx":36, "Sigma_yy":37, "Sigma_zz": 38,
            "SourceFunctions":39, "LambdaMiuMatOverH":40, "LambdaMatOverH":41, "MiuMatOverH":42, "TauLong":43, "OneOverTauSigma":44, "TauShear":45, "InvRhoMatH":46, "Ox":47, "Oy":48, "Oz":49,
            "Pressure":50, "SqrAcc":51, "SensorOutput":52, 
            }
        self.LENGTH_CONST_UINT = 2
        # self.LENGTH_CONST_MEX = 1+self.MAX_SIZE_PML*4

        extra_params = {"BACKEND":"METAL"}
        
        self.bUseSingleKernel = True

        commonIds=['PML_1','PML_2','PML_3','PML_4','PML_5','PML_6','MAIN_1']
        IdsToParse=[]
        for t in ['STRESS','PARTICLE']:
            for k in commonIds:
                IdsToParse.append(f"{k}_{t}")
        IdsToParse.append('SENSORS')
        IdsToParse.append('SNAPSHOT')
        self.IdsToParse = IdsToParse
        super().__init__(arguments, extra_params)
        
    def _PostInitScript(self, arguments, extra_params):
        print("Attempting MLX Initiation...")
        SCode = []
        SCode.append("#define mexType " + extra_params['td'] +"\n")
        SCode.append("#define METAL\n")
        if self.bUseSingleKernel:
            SCode.append("#define METAL_SINGLE_KERNEL\n")
        SCode.append("#define MLX\n")
        SCode.append("#define MAX_SIZE_PML 101\n")
        extra_params['SCode'] = SCode
        self.ctx=mx
        self.ConstantBufferUINT=np.zeros(self.LENGTH_CONST_UINT,np.uint32)
        
    
    def _InitSymbol(self, IP,_NameVar,td,SCode):
        SCode.append('//MLX_CONSTANT_START\n')
        if td in ['float','double']:
            res = 'constant ' + td  + ' ' + _NameVar + ' = %0.9g;\n' %(IP[_NameVar])
        else:
            lType =' int '
            res = 'constant '+ lType  + _NameVar + ' = %i;\n' %(IP[_NameVar])
        SCode.append(res)
        SCode.append('//MLX_CONSTANT_END\n')
        
    def _InitSymbolArray(self, IP,_NameVar,td,SCode):
        SCode.append('//MLX_CONSTANT_START\n')
        res =  "constant  "+ td + " " + _NameVar + "_pr[%i] ={\n" % (IP[_NameVar].size)
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
        SCode.append('//MLX_CONSTANT_END\n')


    def _UpdateSymbol(self, IP, _NameVar,td, SCode):
        if td == "unsigned int": 
            self.constant_buffer_uint[self.C_IND[_NameVar]] = IP[_NameVar]
        else:
            raise ValueError("Something was passed incorrectly in symbol initiation.")


    def _ownGpuCalloc(self, Name,td,dims,ArraysGPUOp):
        print("Allocating for", Name, dims, "elements")
        if Name == "Snapshots":
            pass
        elif td == "float":
            self.HOST_INDEX_MEX[self.C_IND[Name]][0] = np.int64(self._c_mex_type[self._IndexDataMetal[Name]])
            self.HOST_INDEX_MEX[self.C_IND[Name]][1] = np.int64(dims ) 
            self._c_mex_type[self._IndexDataMetal[Name]] += dims
        elif td == "unsigned int":
            self.HOST_INDEX_UINT[self.C_IND[Name]][0] = np.int64(self._c_uint_type)
            self.HOST_INDEX_UINT[self.C_IND[Name]][1] = np.int64(dims)
            self._c_uint_type += dims

    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ArraysGPUOp,ArrayResCPU,flags=[]):
        print("Allocating for", Name, ArrayResCPU[Name].size, "elements")
        SizeCopy = ArrayResCPU[Name].size
        if Name in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH', 'Ox','Oy','Oz', 'SourceFunctions', 'SensorOutput','SqrAcc']: # float
            self.HOST_INDEX_MEX[self.C_IND[Name]][0] = np.int64(self._c_mex_type[self._IndexDataMetal[Name]])
            self.HOST_INDEX_MEX[self.C_IND[Name]][1] = np.int64(SizeCopy)
            self._c_mex_type[self._IndexDataMetal[Name]] += SizeCopy
        elif Name in ['IndexSensorMap','SourceMap','MaterialMap',]: # unsigned int
            self.HOST_INDEX_UINT[self.C_IND[Name]][0] = np.int64(self._c_uint_type)
            self.HOST_INDEX_UINT[self.C_IND[Name]][1] = np.int64(SizeCopy)
            self._c_uint_type += SizeCopy
    
    def _PreExecuteScript(self, arguments, ArraysGPUOp, outparams):
        print("Float entries:", np.sum(self._c_mex_type), "int entries:", self._c_uint_type)
        self._outparams = outparams
        self.mex_buffer=[]
        for nSizes in self._c_mex_type:
            self.mex_buffer.append(self.ctx.zeros(int(nSizes*4)))
        self.uint_buffer=self.ctx.zeros(int(self._c_uint_type*4),self.ctx.uint32)
        self.constant_buffer_uint=self.ctx.array(self.ConstantBufferUINT)

        self._IndexManip()

        for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                    'Ox','Oy','Oz','SourceFunctions']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "float")

        for k in ['IndexSensorMap','SourceMap','MaterialMap']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "unsigned int")

        self.localSensor = [1, 1, 1]
        self.globalSensor = (ceil(arguments['IndexSensorMap'].size / self.localSensor[0]), 1, 1)

    def _IndexManip(self):
        index=np.zeros((self.LENGTH_INDEX_MEX,2),np.uint32)
        for i in range(self.LENGTH_INDEX_MEX):
            index[i,0] = np.uint32(np.int64(0xFFFFFFFF) & np.int64(self.HOST_INDEX_MEX[i][0])) # Not exactly sure if this works
            index[i,1] = np.uint32(np.int64([self.HOST_INDEX_MEX[i][0]])>>32)
        self.index_mex=self.ctx.array(index)
        
        index=np.zeros((self.LENGTH_INDEX_UINT,2),np.uint32)
       
        for i in range(self.LENGTH_INDEX_UINT):
            index[i,0] = np.uint32(0xFFFFFFFF) & np.uint32(self.HOST_INDEX_UINT[i][0]) # Not exactly sure if this works
            index[i,1] = np.uint32(np.int64([self.HOST_INDEX_MEX[i][0]])>>32)
        self.index_uint=self.ctx.array(index)


    def _CompleteCopyToGPU(self, Name, args, SizeCopy, td):
        if td == "float":
            l=int(self.HOST_INDEX_MEX[self.C_IND[Name]][0])
            self.mex_buffer[self._IndexDataMetal[Name]][l:l+SizeCopy] = args[Name].flatten(order='F')
        elif td == "unsigned int":
            l=int(self.HOST_INDEX_UINT[self.C_IND[Name]][0])
            self.uint_buffer[l:l+SizeCopy] = args[Name].flatten(order='F')
        else:
            raise RuntimeError("Something has gone horribly wrong.")
    
    def ParseAndSelectCode(self,OrigLines,Current):
        '''
        This helps to select only the lines of code that are relevant to the current MLX block.
        '''
        sCode=[f'#define MLX_{Current}\n']
        hCode=[]
        bAdd=True
        bHeader= False
        for l in OrigLines:
            if '//MLX_BLOCK' in l:
                if '_START' in l:
                    for k in self.IdsToParse:
                        if k in l:
                            if k != Current:
                                bAdd=False
                elif "_END" in l:
                    bAdd=True
            if '//MLX_CONSTANT_START' in l:
                bHeader=True
            if bAdd and not bHeader:
                sCode.append(l)
            if bHeader:
                hCode.append(l)
            if bHeader and '//MLX_CONSTANT_END' in l:
                bHeader=False
        return ''.join(hCode),''.join(sCode)

    def _InitiateCommands(self, AllC):
        if self.bUseSingleKernel:
            PartsStress=['MAIN_1']
        else:
            PartsStress=['PML_1','PML_2','PML_3','PML_4','PML_5','PML_6','MAIN_1']

        MLXInputNames=['p_CONSTANT_BUFFER_UINT',
                       'p_INDEX_MEX',
                       'p_INDEX_UINT',
                       'p_UINT_BUFFER',
                       'p_MEX_BUFFER_0',
                       'p_MEX_BUFFER_1',
                       'p_MEX_BUFFER_2',
                       'p_MEX_BUFFER_3',
                       'p_MEX_BUFFER_4',
                       'p_MEX_BUFFER_5',
                       'p_MEX_BUFFER_6',
                       'p_MEX_BUFFER_7',
                       'p_MEX_BUFFER_8',
                       'p_MEX_BUFFER_9',
                       'p_MEX_BUFFER_10',
                       'p_MEX_BUFFER_11']

        MLXReadWriteStatus=[] # only MEX buffers are read+write
        for e in MLXInputNames:
            if 'MEX' in e:
                MLXReadWriteStatus.append(True)
            else:
                MLXReadWriteStatus.append(False)

        self.AllStressKernels={}
        for k in PartsStress:
            header,source=self.ParseAndSelectCode(AllC,f"{k}_STRESS")
            kernel = self.ctx.fast.metal_kernel(
                    name=k+"_StressKernel",
                    input_names=MLXInputNames,
                    input_rw_status=MLXReadWriteStatus,
                    output_names=["dummy"],
                    source=source,
                    header=header)
            self.AllStressKernels[k]=kernel

        if self.bUseSingleKernel:
            PartsParticle =['MAIN_1']
        else:
            PartsParticle=['PML_1','PML_2','PML_3','PML_4','PML_5','PML_6','MAIN_1']
        
        self.AllParticleKernels={}
        for k in PartsParticle:
            header,source=self.ParseAndSelectCode(AllC,f"{k}_PARTICLE")
            kernel = self.ctx.fast.metal_kernel(
                    name=k+"_ParticleKernel",
                    input_names=MLXInputNames,
                    input_rw_status=MLXReadWriteStatus,
                    output_names=["dummy"],
                    source=source,
                    header=header)
            self.AllParticleKernels[k]=kernel

        header,source=self.ParseAndSelectCode(AllC,"SENSORS")
        self.SensorsKernel=self.ctx.fast.metal_kernel(
                name="SensorsKernel",
                input_names=MLXInputNames,
                input_rw_status=MLXReadWriteStatus,
                output_names=["dummy"],
                source=source,
                header=header)

    def _Execution(self, arguments, ArrayResCPU, ArrayResOP):
        TimeSteps = arguments['TimeSteps']
        InitDict = {'nStep':0, 'TypeSource':int(arguments['TypeSource'])}
        outparams=self._outparams
        DimsKernel={}
        DimsGroup={}
        DimsKernel['PML_1']=(outparams['PML_Thickness'],
                             outparams['N2'],
                             outparams['N3'])
        DimsGroup['PML_1']=(2, 8, 64)
        DimsKernel['PML_2']=(outparams['PML_Thickness'],
                             outparams['N2'],
                             outparams['N3'])
        DimsGroup['PML_2']=(2, 8, 64)
        DimsKernel['PML_3']=(outparams['N1']-outparams['PML_Thickness']*2,
                             outparams['PML_Thickness'],
                             outparams['N3'])
        DimsGroup['PML_3']=(8, 2, 64)
        DimsKernel['PML_4']=(outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness'],
                            outparams['N3'])
        DimsGroup['PML_4']=(8, 2, 64)
        DimsKernel['PML_5']=(outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness'])
        DimsGroup['PML_5']=(16, 16, 4)
        DimsKernel['PML_6']=(outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness'])
        DimsGroup['PML_6']=(16, 16, 4)
        if self.bUseSingleKernel:
            DimsKernel['MAIN_1']=(outparams['N1'],
                            outparams['N2'],
                            outparams['N3'])
            DimsGroup['MAIN_1']=(8,8, 16)
            kernels=["MAIN_1"]
            
        
        else:
            DimsKernel['MAIN_1']=(outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['N3']-outparams['PML_Thickness']*2)
            DimsGroup['MAIN_1']=(8,8, 16)
            kernels =["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]
            

        AllHandles=[]
            
        for nStep in range(TimeSteps):
            InitDict["nStep"] = nStep
            for i in ['nStep', 'TypeSource']:
                self._UpdateSymbol(InitDict, i, 'unsigned int', [])

            inputs=[self.constant_buffer_uint,
                    self.index_mex,
                    self.index_uint, 
                    self.uint_buffer,
                    self.mex_buffer[0],
                    self.mex_buffer[1],
                    self.mex_buffer[2],
                    self.mex_buffer[3],
                    self.mex_buffer[4],
                    self.mex_buffer[5],
                    self.mex_buffer[6],
                    self.mex_buffer[7],
                    self.mex_buffer[8],
                    self.mex_buffer[9],
                    self.mex_buffer[10],
                    self.mex_buffer[11]]
            for i in kernels:
                nSize=np.prod(DimsKernel[i])

                handle = self.AllStressKernels[i](
                                        inputs=inputs,
                                        grid=[nSize,1,1],
                                        threadgroup=DimsGroup[i],
                                        output_shapes=[[1,1,1]], # dummy output is just 1 float, as we never write to it
                                        output_dtypes=[self.ctx.float32],
                                        use_optimal_threadgroups=True
                                        )[0]

                


                AllHandles.append(handle)

            for i in kernels:
                handle = self.AllParticleKernels[i](
                                        inputs=inputs,
                                        grid=[nSize,1,1],
                                        threadgroup=DimsGroup[i],
                                        output_shapes=[[1,1,1]], # dummy output is just 1 float, as we never write to it
                                        output_dtypes=[self.ctx.float32],
                                        use_optimal_threadgroups=True
                                        )[0]

                AllHandles.append(handle)
            
            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):
                handle=self.SensorsKernel(
                                        inputs=inputs,
                                        grid=self.globalSensor,
                                        threadgroup=DimsGroup[i],
                                        output_shapes=[[1,1,1]], # dummy output is just 1 float, as we never write to it
                                        output_dtypes=[self.ctx.float32],
                                        use_optimal_threadgroups=True
                                        )[0]

                AllHandles.append(handle)

            while len(AllHandles)>0:
                self.ctx.eval(AllHandles.pop(0))

        for i in ['SqrAcc', 'SensorOutput']:
            SizeCopy = ArrayResCPU[i].size
            Shape = ArrayResCPU[i].shape
            Buffer=np.array(self.mex_buffer[self._IndexDataMetal[i]][int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)])
            ArrayResCPU[i][:,:,:] = np.reshape(Buffer, Shape,order='F')
     
        
        SizeBuffer = {1:0, 6:0, 7:0, 9:0}
        for i in ['Vx', 'Vy', 'Vz', 'Sigma_xx', 'Sigma_yy', 'Sigma_zz', 'Sigma_xy', 'Sigma_xz', 'Sigma_yz']:
            SizeBuffer[self._IndexDataMetal[i]] += ArrayResCPU[i].size* self.ZoneCount
        
        for i in ["LambdaMiuMatOverH", "LambdaMatOverH", "MiuMatOverH", "TauLong", "OneOverTauSigma", "TauShear", "InvRhoMatH", "Ox", "Oy", "Oz"]:
            SizeBuffer[9] += arguments[i].size
        
        SizeBuffer[9] += ArrayResCPU['Pressure'].size* self.ZoneCount

        for i in ['Vx', 'Vy', 'Vz', 'Sigma_xx', 'Sigma_yy', 'Sigma_zz', 'Sigma_xy', 'Sigma_xz', 'Sigma_yz', 'Pressure']:
            SizeCopy = ArrayResCPU[i].size * self.ZoneCount
            sz=ArrayResCPU[i].shape
            Shape = (sz[0],sz[1],sz[2],self.ZoneCount)
            Buffer=np.array(self.mex_buffer[self._IndexDataMetal[i]])[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
            Buffer=Buffer.reshape(Shape,order='F')
            ArrayResCPU[i][:,:,:] = np.sum(Buffer,axis=3)/self.ZoneCount
          
        del self.constant_buffer_uint
        del self.index_mex
        del self.index_uint 
        del self.uint_buffer
        while len(self.mex_buffer)>0:
            handle = self.mex_buffer.pop()
            del handle

def StaggeredFDTD_3D_MLX(arguments):
    Instance = StaggeredFDTD_3D_With_Relaxation_MLX(arguments)
    Results = Instance.Results
    return Results
