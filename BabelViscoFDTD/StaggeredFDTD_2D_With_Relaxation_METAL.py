from calendar import c
from multiprocessing.dummy import Array
from unicodedata import name
import numpy as np
import os
from pathlib import Path
import platform
import time
import sys
import gc
import tempfile
from shutil import copyfile

from .StaggeredFDTD_2D_With_Relaxation_BASE import StaggeredFDTD_2D_With_Relaxation_BASE

from distutils.sysconfig import get_python_inc
from math import ceil
import ctypes
from ctypes import c_byte, c_int, c_uint32, c_float, c_wchar_p, c_uint64

mc = None

def ListDevices():
    import metalcomputebabel 
    devicesIDs=[]
    devices = metalcomputebabel.get_devices()
    for dev in devices:
        devicesIDs.append(dev.deviceName)
    return devicesIDs

class StaggeredFDTD_2D_With_Relaxation_METAL_MetalCompute(StaggeredFDTD_2D_With_Relaxation_BASE):
    '''
    This version is mainly for Mx processors and which is based in a modified+forked version of metalcompute. As X64 will be phased out, in the future the metalcompute version will take over
    '''
    def __init__(self, arguments):
        global mc
        import metalcomputebabel
        mc = metalcomputebabel
        # Definition of some constants, etc
        self.MAX_SIZE_PML = 101
        self._c_mex_type = np.zeros(12, np.uint64)
        self._c_uint_type = np.uint64(0)
        self.HOST_INDEX_MEX = np.zeros((31, 2), np.uint64)
        self.HOST_INDEX_UINT = np.zeros((3, 2), np.uint64)
        self.LENGTH_INDEX_MEX = self.HOST_INDEX_MEX.shape[0]
        self.LENGTH_INDEX_UINT = self.HOST_INDEX_UINT.shape[0]
        self.ZoneCount = arguments['SPP_ZONES']
        self._IndexDataMetal = {
            "V_x_x":0, "V_y_x":0, "V_x_y":0, "V_y_y":0,
            "Vx":1, "Vy":1,
            "Rxx":2, "Ryy":2,
            "Rxy":3,
            "Sigma_x_xx":4, "Sigma_y_xx":4, "Sigma_x_yy":4, "Sigma_y_yy":4,
            "Sigma_x_xy":5, "Sigma_y_xy":5,
            "Sigma_xy":6, "Sigma_xx":6,
            "Sigma_yy":7, "Pressure":7,
            "SourceFunctions":8,
            "LambdaMiuMatOverH":9, "LambdaMatOverH":9, "MiuMatOverH":9,
            "TauLong":9, "OneOverTauSigma":9, "TauShear":9, "InvRhoMatH":9, "Ox":9, "Oy":9,  
            "SqrAcc":10,
            "SensorOutput":11
            }
        self.C_IND = {
            "IndexSensorMap":0, "SourceMap":1, "MaterialMap": 2,
            "nStep":0, "TypeSource":1, 
            "V_x_x":0, "V_y_x":1, "V_x_y":2, "V_y_y":3, "Vx":4, "Vy":5,
            "Rxx":6, "Ryy":7, "Rxy":8,
            "Sigma_x_xx":9, "Sigma_y_xx":10, "Sigma_x_yy":11, "Sigma_y_yy":12,
            "Sigma_x_xy":13, "Sigma_y_xy":14, "Sigma_xy":15, 
            "Sigma_xx":16, "Sigma_yy":17, 
            "SourceFunctions":18, "LambdaMiuMatOverH":19, 
            "LambdaMatOverH":20, "MiuMatOverH":21, 
            "TauLong":22, "OneOverTauSigma":23, "TauShear":24, 
            "InvRhoMatH":25, "Ox":26, "Oy":27,
            "Pressure":28, "SqrAcc":29, "SensorOutput":30, 
            }

        self.FUNCTION_LOCALS = {}
        for i in ['MAIN_1' ]:
            self.FUNCTION_LOCALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
        self.FUNCTION_GLOBALS = {}
        for i in ['MAIN_1']:
            self.FUNCTION_GLOBALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
                
        self.LENGTH_CONST_UINT = 2
        # self.LENGTH_CONST_MEX = 1+self.MAX_SIZE_PML*4

        extra_params = {"BACKEND":"METAL"}
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
        SCode.append("#define METALCOMPUTE\n")
        SCode.append("#define MAX_SIZE_PML 101\n")
        extra_params['SCode'] = SCode
        self.ctx = mc.Device(n)
        self.ConstantBufferUINT=np.zeros(self.LENGTH_CONST_UINT,np.uint32)
        # self.ConstantBufferMEX=np.zeros(self.LENGTH_CONST_MEX,np.float32)
        print(self.ctx)
        if 'arm64' not in platform.platform():
            print('Setting Metal for External or AMD processor')
            self.ctx.set_external_gpu(1) 
    
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


    def _UpdateSymbol(self, IP, _NameVar,td, SCode):
        if td == "float":
            self.constant_buffer_mex.modify(np.array([IP[_NameVar]],dtype=np.float32),int(self.C_IND[_NameVar]),1)
        elif td == "unsigned int": 
            self.constant_buffer_uint.modify(np.array(IP[_NameVar],dtype=np.uint32),int(self.C_IND[_NameVar]),1)
        else:
            raise ValueError("Something was passed incorrectly in symbol initiation.")


    def _ownGpuCalloc(self, Name,td,dims,ArraysGPUOp):
        print("Allocating for", Name, dims, "elements")
        if Name == "Snapshots":
            pass
        elif td == "float":
            self.HOST_INDEX_MEX[self.C_IND[Name]][0] = np.uint64(self._c_mex_type[self._IndexDataMetal[Name]])
            self.HOST_INDEX_MEX[self.C_IND[Name]][1] = np.uint64(dims ) 
            self._c_mex_type[self._IndexDataMetal[Name]] += dims
        elif td == "unsigned int":
            self.HOST_INDEX_UINT[self.C_IND[Name]][0] = np.uint64(self._c_uint_type)
            self.HOST_INDEX_UINT[self.C_IND[Name]][1] = np.uint64(dims)
            self._c_uint_type += dims

    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ArraysGPUOp,ArrayResCPU,flags=[]):
        print("Allocating for", Name, ArrayResCPU[Name].size, "elements")
        SizeCopy = ArrayResCPU[Name].size
        if Name in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH', 'Ox','Oy', 'SourceFunctions', 'SensorOutput','SqrAcc']: # float
            self.HOST_INDEX_MEX[self.C_IND[Name]][0] = np.uint64(self._c_mex_type[self._IndexDataMetal[Name]])
            self.HOST_INDEX_MEX[self.C_IND[Name]][1] = np.uint64(SizeCopy)
            self._c_mex_type[self._IndexDataMetal[Name]] += SizeCopy
        elif Name in ['IndexSensorMap','SourceMap','MaterialMap',]: # unsigned int
            self.HOST_INDEX_UINT[self.C_IND[Name]][0] = np.uint64(self._c_uint_type)
            self.HOST_INDEX_UINT[self.C_IND[Name]][1] = np.uint64(SizeCopy)
            self._c_uint_type += SizeCopy
    
    def _PreExecuteScript(self, arguments, ArraysGPUOp, outparams):
        print("Float entries:", np.sum(self._c_mex_type), "int entries:", self._c_uint_type)
        self.mex_buffer=[]
        for nSizes in self._c_mex_type:
            handle=self.ctx.buffer(nSizes*4)
            self.mex_buffer.append(handle)
        self.uint_buffer=self.ctx.buffer(self._c_uint_type*4)
        self.constant_buffer_uint=self.ctx.buffer(self.ConstantBufferUINT)
        # self.constant_buffer_mex=self.ctx.buffer(self.ConstantBufferMEX)

        self._IndexManip()

        for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                    'Ox','Oy','SourceFunctions']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "float")

        for k in ['IndexSensorMap','SourceMap','MaterialMap']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "unsigned int")

        if arguments['ManualLocalSize'][0]!=-1:
            self._SET_USER_LOCAL(arguments['ManualLocalSize'])
        else:
            self._CALC_USER_LOCAL("MAIN_1", "STRESS")
            self._CALC_USER_LOCAL("MAIN_1", "PARTICLE")
               
        if arguments['ManualGroupSize'][0] != -1:
            self._SET_USER_GLOBAL(arguments['ManualGroupSize'])
        else:
            self._CALC_USER_GROUP_MAIN(arguments, outparams)

        self.localSensor = [1, 1, 1]
        self.globalSensor = [ceil(arguments['IndexSensorMap'].size / self.localSensor[0]), 1, 1]

    def _IndexManip(self):
        index=np.zeros((self.LENGTH_INDEX_MEX,2),np.uint32)
        for i in range(self.LENGTH_INDEX_MEX):
            index[i,0] = np.uint32(np.uint64(0xFFFFFFFF) & np.uint64(self.HOST_INDEX_MEX[i][0])) # Not exactly sure if this works
            index[i,1] = np.uint32(np.uint64([self.HOST_INDEX_MEX[i][0]])>>32)
        self.index_mex=self.ctx.buffer(index)
        
        index=np.zeros((self.LENGTH_INDEX_UINT,2),np.uint32)
       
        for i in range(self.LENGTH_INDEX_UINT):
            index[i,0] = np.uint32(0xFFFFFFFF) & np.uint32(self.HOST_INDEX_UINT[i][0]) # Not exactly sure if this works
            index[i,1] = np.uint32(np.uint64([self.HOST_INDEX_MEX[i][0]])>>32)
        self.index_uint=self.ctx.buffer(index)


    def _CompleteCopyToGPU(self, Name, args, SizeCopy, td):
        if td == "float":
            self.mex_buffer[self._IndexDataMetal[Name]].modify(args[Name].flatten(order='F'),int(self.HOST_INDEX_MEX[self.C_IND[Name]][0]),int(SizeCopy))
        elif td == "unsigned int":
            self.uint_buffer.modify(args[Name].flatten(order='F'),int(self.HOST_INDEX_UINT[self.C_IND[Name]][0]),int(SizeCopy))
        else:
            raise RuntimeError("Something has gone horribly wrong.")
    

    def _CALC_USER_LOCAL(self, Name, Type):
        #this will be calculated by metalcompute library
        self.FUNCTION_LOCALS[Name][Type][0] = 1
        self.FUNCTION_LOCALS[Name][Type][1] = 1
        self.FUNCTION_LOCALS[Name][Type][2] = 1
        # print(Name, "local", Type + " = [" + str(w) + ", " + str(h) + ", " + str(z) + "]")
    
    def _SET_USER_LOCAL(self, ManualLocalSize):
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(2):
                self.FUNCTION_LOCALS['MAIN_1'][Type][index] = ManualLocalSize[index] # Can probably change this
    
    def _SET_USER_GLOBAL(self, ManualGlobalSize):
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(2):
               self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ManualGlobalSize[index] 

    def _CALC_USER_GROUP_MAIN(self, arguments, outparams):
        self._outparams = outparams
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(2):
                self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ceil((arguments[('N'+str(index + 1))]) / self.FUNCTION_LOCALS['MAIN_1'][Type][index])
            print("MAIN_1_global_" + Type, "=", str(self.FUNCTION_GLOBALS['MAIN_1'][Type]))
    

    def _InitiateCommands(self, AllC):
        prg = self.ctx.kernel(AllC)
        PartsStress=['MAIN_1']
        self.AllStressKernels={}
        for k in PartsStress:
            self.AllStressKernels[k]=prg.function(k+"_StressKernel")

        PartsParticle=['MAIN_1']
        self.AllParticleKernels={}
        for k in PartsParticle:
            self.AllParticleKernels[k]=prg.function(k+"_ParticleKernel")
        
        self.SensorsKernel=prg.function('SensorsKernel')

    def _Execution(self, arguments, ArrayResCPU, ArrayResOP):
        TimeSteps = arguments['TimeSteps']
        InitDict = {'nStep':0, 'TypeSource':int(arguments['TypeSource'])}
        outparams=self._outparams
        DimsKernel={}
        DimsKernel['MAIN_1']=[outparams['N1'],
                            outparams['N2']]
    
        nref=sys.getrefcount(self.mex_buffer[7])
        print('before exec nref',nref)
        for nStep in range(TimeSteps):
            InitDict["nStep"] = nStep
            for i in ['nStep', 'TypeSource']:
                self._UpdateSymbol(InitDict, i, 'unsigned int', [])

            self.ctx.init_command_buffer()
            AllHandles=[]
  
            for i in ["MAIN_1"]:
                nSize=np.prod(DimsKernel[i])
                handle=self.AllStressKernels[i](nSize,self.constant_buffer_uint,
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
                                               self.mex_buffer[11])
                AllHandles.append(handle)

            for i in ["MAIN_1"]:
                nSize=np.prod(DimsKernel[i])
                handle = self.AllParticleKernels[i](nSize,self.constant_buffer_uint,
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
                                               self.mex_buffer[11])

                AllHandles.append(handle)
            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):
                handle=self.SensorsKernel(np.prod(self.globalSensor),
                                self.constant_buffer_uint,
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
                                self.mex_buffer[11])
                AllHandles.append(handle)
            self.ctx.commit_command_buffer()
            self.ctx.wait_command_buffer()
            while len(AllHandles)>0:
                handle = AllHandles.pop(0) 
                del handle
        # nref=sys.getrefcount(self.mex_buffer[7])
        # print('after exec nref',nref)
        if 'arm64' not in platform.platform():
            self.ctx.sync_buffers((self.mex_buffer[0],
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
                                    self.mex_buffer[11]))
        for i in ['SqrAcc', 'SensorOutput']:
            SizeCopy = ArrayResCPU[i].size
            Shape = ArrayResCPU[i].shape
            print('getting ',i,self._IndexDataMetal[i])
            Buffer=np.frombuffer(self.mex_buffer[self._IndexDataMetal[i]],dtype=np.float32)[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
            ArrayResCPU[i][:,:,:] = Buffer.reshape(Shape,order='F').copy()
     
        
        SizeBuffer = {1:0, 6:0, 7:0, 9:0}
        for i in ['Vx', 'Vy',  'Sigma_xx', 'Sigma_yy', 'Sigma_xy']:
            SizeBuffer[self._IndexDataMetal[i]] += ArrayResCPU[i].size* self.ZoneCount
        
        for i in ["LambdaMiuMatOverH", "LambdaMatOverH", "MiuMatOverH", "TauLong", "OneOverTauSigma", "TauShear", "InvRhoMatH", "Ox", "Oy"]:
            SizeBuffer[9] += arguments[i].size
        
        SizeBuffer[9] += ArrayResCPU['Pressure'].size* self.ZoneCount

        for i in ['Vx', 'Vy',  'Sigma_xx', 'Sigma_yy', 'Sigma_xy']:
            SizeCopy = ArrayResCPU[i].size * self.ZoneCount
            sz=ArrayResCPU[i].shape
            Shape = (sz[0],sz[1],self.ZoneCount)
            Buffer=np.frombuffer(self.mex_buffer[self._IndexDataMetal[i]],dtype=np.float32)[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
            Buffer=Buffer.reshape(Shape,order='F')
            ArrayResCPU[i][:,:] = np.sum(Buffer,axis=2)/self.ZoneCount
        nref=sys.getrefcount(self.mex_buffer[7])
        # print('after 1st copying',nref)
        # print(gc.get_referrers(self.mex_buffer[7]))
        for i in ['Pressure']:
            SizeCopy = ArrayResCPU[i].size * self.ZoneCount
            sz=ArrayResCPU[i].shape
            Shape = (sz[0],sz[1],self.ZoneCount)
            Buffer=np.frombuffer(self.mex_buffer[self._IndexDataMetal[i]],dtype=np.float32)[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
            Buffer=Buffer.reshape(Shape,order='F')
            ArrayResCPU[i][:,:] = np.sum(Buffer,axis=2)/self.ZoneCount
        # nref=sys.getrefcount(self.mex_buffer[7])
        # print('after 2nd copying',nref)
        # print(gc.get_referrers(self.mex_buffer[7]))
        # print(gc.get_referrers(self.mex_buffer[8]))
        del self.constant_buffer_uint
        # del self.constant_buffer_mex
        del self.index_mex
        del self.index_uint 
        del self.uint_buffer
        while len(self.mex_buffer)>0:
            handle = self.mex_buffer.pop(0)
            # nref=sys.getrefcount(handle)
            # print('mex nref',nref)
            del handle

def StaggeredFDTD_2D_METAL(arguments):
    Instance = StaggeredFDTD_2D_With_Relaxation_METAL_MetalCompute(arguments)
    Results = Instance.Results
    return Results
