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

class StaggeredFDTD_3D_With_Relaxation_METAL_own_swift(StaggeredFDTD_3D_With_Relaxation_BASE):
    '''
    This version is mainly for X64 and AMD processors using our initial implementation, which runs faster with that old version
    '''
    def __init__(self, arguments):
        #Begin with initializing Swift Functions, etc.
        os.environ['__BabelMetal'] =(os.path.dirname(os.path.abspath(__file__))+os.sep+'tools')
        print(os.environ['__BabelMetal'])
        os.environ['__BabelMetalDevice'] = arguments['DefaultGPUDeviceName']
        print('loading',os.path.dirname(os.path.abspath(__file__))+"/tools/libFDTDSwift.dylib") 
        self.swift_fun = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__))+"/tools/libFDTDSwift.dylib")
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
            "N1":0, "N2":1, "N3":2, "Limit_I_low_PML":3, "Limit_J_low_PML":4, "Limit_K_low_PML":5, "Limit_I_up_PML":6, "Limit_J_up_PML":7, "Limit_K_up_PML":8, 
            "SizeCorrI":9, "SizeCorrJ":10, "SizeCorrK":11, "PML_Thickness":12, "NumberSources":13, "NumberSensors":14, "TimeSteps":15, 
            "SizePML":16, "SizePMLxp1":17, "SizePMLyp1":18, "SizePMLzp1":19, "SizePMLxp1yp1zp1":20, "ZoneCount":21, "SelRMSorPeak":22, "SelMapsRMSPeak":23, "IndexRMSPeak_ALLV":24, 
            "IndexRMSPeak_Vx":25, "IndexRMSPeak_Vy":26, "IndexRMSPeak_Vz":27, "IndexRMSPeak_Sigmaxx":28, "IndexRMSPeak_Sigmayy":29, "IndexRMSPeak_Sigmazz":30,
            "IndexRMSPeak_Sigmaxy":31, "IndexRMSPeak_Sigmaxz":32, "IndexRMSPeak_Sigmayz":33, "NumberSelRMSPeakMaps":34, "SelMapsSensors":35, "IndexSensor_ALLV":36,
            "IndexSensor_Vx":37, "IndexSensor_Vy":38, "IndexSensor_Vz":39, "IndexSensor_Sigmaxx":40, "IndexSensor_Sigmayy":41, "IndexSensor_Sigmazz":42, "IndexSensor_Sigmaxy":43,
            "IndexSensor_Sigmaxz":44, "IndexSensor_Sigmayz":45, "NumberSelSensorMaps":46, "SensorSubSampling":47, "nStep":48, "TypeSource":49, "CurrSnap":50, "LengthSource":51, "SelK":52,
            "IndexRMSPeak_Pressure":53, "IndexSensor_Pressure":54, "SensorStart":55,
            "IndexSensorMap":0, "SourceMap":1, "MaterialMap": 2,
            # MEX
            "DT":0, "InvDXDTplus":1, "DXDTminus":1+self.MAX_SIZE_PML, "InvDXDTplushp":1+self.MAX_SIZE_PML*2, "DXDTminushp":1+self.MAX_SIZE_PML*3,
            "V_x_x":0, "V_y_x":1, "V_z_x":2, "V_x_y":3, "V_y_y":4, "V_z_y":5, "V_x_z":6, "V_y_z":7, "V_z_z":8, "Vx":9, "Vy":10, "Vz":11, "Rxx":12, "Ryy":13, "Rzz":14, "Rxy":15, "Rxz":16, "Ryz":17,
            "Sigma_x_xx":18, "Sigma_y_xx":19, "Sigma_z_xx":20, "Sigma_x_yy":21, "Sigma_y_yy":22, "Sigma_z_yy":23, "Sigma_x_zz":24, "Sigma_y_zz":25, "Sigma_z_zz":26, 
            "Sigma_x_xy":27, "Sigma_y_xy":28, "Sigma_x_xz":29, "Sigma_z_xz":30, "Sigma_y_yz":31, "Sigma_z_yz":32, "Sigma_xy":33, "Sigma_xz":34, "Sigma_yz":35, "Sigma_xx":36, "Sigma_yy":37, "Sigma_zz": 38,
            "SourceFunctions":39, "LambdaMiuMatOverH":40, "LambdaMatOverH":41, "MiuMatOverH":42, "TauLong":43, "OneOverTauSigma":44, "TauShear":45, "InvRhoMatH":46, "Ox":47, "Oy":48, "Oz":49,
            "Pressure":50, "SqrAcc":51, "SensorOutput":52, 
            }

        self.FUNCTION_LOCALS = {}
        for i in ['MAIN_1', "PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
            self.FUNCTION_LOCALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
        self.FUNCTION_GLOBALS = {}
        for i in ['MAIN_1', "PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
            self.FUNCTION_GLOBALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
                
        self.LENGTH_CONST_UINT = 56
        self.LENGTH_CONST_MEX = 1+self.MAX_SIZE_PML*4

        # Defines functions sent to Swift
        self.swift_fun.InitializeMetalDevices.argtypes = []
        self.swift_fun.ConstantBuffers.argtypes = [
            ctypes.c_int,
            ctypes.c_int]
        self.swift_fun.SymbolInitiation_uint.argtypes = [
            c_uint32,
            c_uint32]
        self.swift_fun.SymbolInitiation_mex.argtypes = [
            c_uint32,
            c_float]
        self.swift_fun.CompleteCopyMEX.argtypes = [
            c_int,
            ctypes.POINTER(ctypes.c_float),
            c_uint64,
            c_uint64]
        self.swift_fun.CompleteCopyUInt.argtypes = [
            c_int,
            ctypes.POINTER(c_uint32),
            c_uint64]
        self.swift_fun.IndexManipMEX.argtypes = [
            c_uint32,
            c_uint32,
            c_uint32]
        self.swift_fun.IndexManipUInt.argtypes = [
            c_uint32,
            c_uint32,
            c_uint32]
        self.swift_fun.IndexDidModify.argtypes = [
            c_uint64,
            c_uint64,
            c_uint64,
            c_uint64]
        self.swift_fun.GetMaxTotalThreadsPerThreadgroup.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.swift_fun.GetMaxTotalThreadsPerThreadgroup.restype = ctypes.c_uint32
        self.swift_fun.GetThreadExecutionWidth.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.swift_fun.GetThreadExecutionWidth.restype = ctypes.c_uint32 
        extra_params = {"BACKEND":"METAL"}
        super().__init__(arguments, extra_params)
        
    def _PostInitScript(self, arguments, extra_params):
        print("Attempting Metal Initiation...")
        if self.swift_fun.InitializeMetalDevices() == -1:
            raise ValueError("Something has gone horribly wrong.")
        if self.swift_fun.ConstantBuffers(c_int(self.LENGTH_CONST_UINT), c_int(self.LENGTH_CONST_MEX)) == -1:
            raise ValueError("Something has gone horribly wrong")
    

    def _InitSymbol(self, IP,_NameVar,td, SCode):
        if td == "float":
            self.swift_fun.SymbolInitiation_mex(c_uint32(self.C_IND[_NameVar]), c_float(IP[_NameVar]))
        elif td == "unsigned int": 
            self.swift_fun.SymbolInitiation_uint(c_uint32(self.C_IND[_NameVar]), c_uint32(IP[_NameVar]))
        else:
            raise ValueError("Something was passed incorrectly in symbol initiation.")
        
    
    def _InitSymbolArray(self, IP,_NameVar,td, SCode):
        if td == "float":
            for i in range(IP[_NameVar].size):
                self.swift_fun.SymbolInitiation_mex(c_uint32(self.C_IND[_NameVar] + i), c_float(IP[_NameVar][i])) #Double check second arg
        elif td == "unsigned int": 
            for i in range(IP[_NameVar].size):
                self.swift_fun.SymbolInitiation_uint(c_uint32(self.C_IND[_NameVar] + i), c_uint32(IP[_NameVar][i]))  #Second arg
        # I think this way runs faster since it's not doing the if check every loop?


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
        if Name in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH', 'Ox','Oy','Oz', 'SourceFunctions', 'SensorOutput','SqrAcc']: # float
            self.HOST_INDEX_MEX[self.C_IND[Name]][0] = np.uint64(self._c_mex_type[self._IndexDataMetal[Name]])
            self.HOST_INDEX_MEX[self.C_IND[Name]][1] = np.uint64(SizeCopy)
            self._c_mex_type[self._IndexDataMetal[Name]] += SizeCopy
        elif Name in ['IndexSensorMap','SourceMap','MaterialMap',]: # unsigned int
            self.HOST_INDEX_UINT[self.C_IND[Name]][0] = np.uint64(self._c_uint_type)
            self.HOST_INDEX_UINT[self.C_IND[Name]][1] = np.uint64(SizeCopy)
            self._c_uint_type += SizeCopy
    
    def _PreExecuteScript(self, arguments, ArraysGPUOp, outparams):
        print("Float entries:", np.sum(self._c_mex_type), "int entries:", self._c_uint_type)
        self.swift_fun.BufferIndexCreator.argtypes = [ctypes.POINTER(c_uint64), c_uint64, c_uint64, c_uint64]
        self.swift_fun.BufferIndexCreator(self._c_mex_type.ctypes.data_as(ctypes.POINTER(c_uint64)),c_uint64(np.uint64(self._c_uint_type)), c_uint64(self.LENGTH_INDEX_MEX), c_uint64(self.LENGTH_INDEX_UINT))

        self._IndexManip()

        self.swift_fun.IndexDidModify(c_uint64(self.LENGTH_INDEX_MEX), c_uint64(self.LENGTH_INDEX_UINT), c_uint64(self.LENGTH_CONST_MEX), c_uint64(self.LENGTH_CONST_UINT))

        for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                    'Ox','Oy','Oz','SourceFunctions']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "float")

        for k in ['IndexSensorMap','SourceMap','MaterialMap']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "unsigned int")

        if arguments['ManualLocalSize'][0]!=-1:
            self._SET_USER_LOCAL(arguments['ManualLocalSize'])
        else:
            self._CALC_USER_LOCAL("MAIN_1", "STRESS")
            self._CALC_USER_LOCAL("MAIN_1", "PARTICLE")
        
        for j in ["STRESS", "PARTICLE"]:
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
                self._CALC_USER_LOCAL(i, j)
        
        if arguments['ManualGroupSize'][0] != -1:
            self._SET_USER_GLOBAL(arguments['ManualGroupSize'])
        else:
            self._CALC_USER_GROUP_MAIN(arguments, outparams)

        self. _CALC_USER_GROUP_PML(outparams)

        self.swift_fun.maxThreadSensor.argtypes = []
        self.swift_fun.maxThreadSensor.restype = c_int
        
        self.localSensor = [self.swift_fun.maxThreadSensor(), 1, 1]
        self.globalSensor = [ceil(arguments['IndexSensorMap'].size / self.localSensor[0]), 1, 1]

    def _IndexManip(self):
        for i in range(self.LENGTH_INDEX_MEX):
            data = np.uint32(np.uint64(0xFFFFFFFF) & np.uint64(self.HOST_INDEX_MEX[i][0])) # Not exactly sure if this works
            data2 = np.uint64([self.HOST_INDEX_MEX[i][0]])>>32
            self.swift_fun.IndexManipMEX(c_uint32(np.uint32(data)), c_uint32(np.uint32(data2[0])), c_uint32(i))

        for i in range(self.LENGTH_INDEX_UINT):
            data = np.uint32(0xFFFFFFFF) & np.uint32(self.HOST_INDEX_UINT[i][0]) # Not exactly sure if this works
            data2 = np.uint64([self.HOST_INDEX_MEX[i][0]])>>32
            self.swift_fun.IndexManipUInt(c_uint32(np.uint32(data)), c_uint32(np.uint32(data2[0])), c_uint32(i))


    def _CompleteCopyToGPU(self, Name, args, SizeCopy, td):
        if td == "float":
            self.swift_fun.CompleteCopyMEX(c_int(SizeCopy), args[Name].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), c_uint64(self.HOST_INDEX_MEX[self.C_IND[Name]][0]), c_uint64(self._IndexDataMetal[Name]))
        elif td == "unsigned int":
            self.swift_fun.CompleteCopyUInt(c_int(SizeCopy), args[Name].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), c_uint64(self.HOST_INDEX_UINT[self.C_IND[Name]][0]))
        else:
            raise RuntimeError("Something has gone horribly wrong.")

    def _CALC_USER_LOCAL(self, Name, Type):
        if Type == "STRESS":
            Swift = 0
        elif Type == "PARTICLE":
            Swift = 1
        print(Name, Type)
        w = self.swift_fun.GetThreadExecutionWidth(ctypes.c_char_p(bytes(Name, 'utf-8')), c_int(Swift)) 
        h = self.swift_fun.GetMaxTotalThreadsPerThreadgroup(ctypes.c_char_p(bytes(Name, 'utf-8')), c_int(Swift)) / w
        z = 1
        if h % 2 == 0:
            h = h / 2
            z = 2
        self.FUNCTION_LOCALS[Name][Type][0] = w
        self.FUNCTION_LOCALS[Name][Type][1] = int(h)
        self.FUNCTION_LOCALS[Name][Type][2] = z
        print(Name, "local", Type + " = [" + str(w) + ", " + str(h) + ", " + str(z) + "]")
    
    def _SET_USER_LOCAL(self, ManualLocalSize):
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
                self.FUNCTION_LOCALS['MAIN_1'][Type][index] = ManualLocalSize[index] # Can probably change this
    
    def _SET_USER_GLOBAL(self, ManualGlobalSize):
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
               self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ManualGlobalSize[index] 

    def _CALC_USER_GROUP_MAIN(self, arguments, outparams):
        self._outparams = outparams
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
                self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ceil((arguments[('N'+str(index + 1))]-outparams['PML_Thickness']*2) / self.FUNCTION_LOCALS['MAIN_1'][Type][index])
            print("MAIN_1_global_" + Type, "=", str(self.FUNCTION_GLOBALS['MAIN_1'][Type]))
    
    def _CALC_USER_GROUP_PML(self, outparams):
        for Type in ['STRESS', 'PARTICLE']:
            self.FUNCTION_GLOBALS['PML_1'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][0])
            self.FUNCTION_GLOBALS['PML_1'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][1])
            self.FUNCTION_GLOBALS['PML_1'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][2])

            self.FUNCTION_GLOBALS['PML_2'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_2'][Type][0])
            self.FUNCTION_GLOBALS['PML_2'][Type][1] = ceil(outparams['SizeCorrJ'] / self.FUNCTION_LOCALS['PML_2'][Type][1])
            self.FUNCTION_GLOBALS['PML_2'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_2'][Type][2])

            self.FUNCTION_GLOBALS['PML_3'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_3'][Type][0])
            self.FUNCTION_GLOBALS['PML_3'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_3'][Type][1])
            self.FUNCTION_GLOBALS['PML_3'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_3'][Type][2])

            self.FUNCTION_GLOBALS['PML_4'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_4'][Type][0])
            self.FUNCTION_GLOBALS['PML_4'][Type][1] = ceil(outparams['SizeCorrJ'] / self.FUNCTION_LOCALS['PML_4'][Type][1])
            self.FUNCTION_GLOBALS['PML_4'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_4'][Type][2])

            self.FUNCTION_GLOBALS['PML_5'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_5'][Type][0])
            self.FUNCTION_GLOBALS['PML_5'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_5'][Type][1])
            self.FUNCTION_GLOBALS['PML_5'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_5'][Type][2])

            self.FUNCTION_GLOBALS['PML_6'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_6'][Type][0])
            self.FUNCTION_GLOBALS['PML_6'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_6'][Type][1])
            self.FUNCTION_GLOBALS['PML_6'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_6'][Type][2])
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
                print(i + "_global_" + Type + "=", str(self.FUNCTION_GLOBALS[i][Type]))

    def _InitiateCommands(self, AllC):
        pass

    def _Execution(self, arguments, ArrayResCPU, ArrayResOP):
        TimeSteps = arguments['TimeSteps']
        self.swift_fun.EncoderInit.argtypes = [] # Not sure if this is necessary
        self.swift_fun.EncodeCommit.argtypes = []
        self.swift_fun.EncodeSensors.argtypes = []
        self.swift_fun.SyncChange.argtypes = []
        self.swift_fun.EncodeStress.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.swift_fun.EncodeParticle.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.swift_fun.CopyFromGPUMEX.argtypes = [
            ctypes.c_uint64 # Can we do this instead of sending pointers of everything?
        ]
        self.swift_fun.CopyFromGPUMEX.restype = ctypes.POINTER(ctypes.c_float)
        self.swift_fun.EncodeSensors.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
    
        InitDict = {'nStep':0, 'TypeSource':int(arguments['TypeSource'])}
        outparams=self._outparams
        DimsKernel={}
        DimsKernel['PML_1']=[outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['PML_Thickness']*2]
        DimsKernel['PML_2']=[outparams['PML_Thickness']*2,outparams['N2']-outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_3']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_4']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['N2']-outparams['PML_Thickness']*2,outparams['PML_Thickness']*2]
        DimsKernel['PML_5']=[outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_6']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['PML_Thickness']*2]
        DimsKernel['MAIN_1']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['N2']-outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        for k in DimsKernel:
            DimsKernel[k]=[c_uint32(DimsKernel[k][0]),c_uint32(DimsKernel[k][1]),c_uint32(DimsKernel[k][2])]

        for nStep in range(TimeSteps):
            InitDict["nStep"] = nStep
            for i in ['nStep', 'TypeSource']:
                self._InitSymbol(InitDict, i, 'unsigned int', [])

            self.swift_fun.EncoderInit()

            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]:
                str_ptr = ctypes.c_char_p(bytes(i, 'utf-8'))
                glox_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["STRESS"][0])
                gloy_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["STRESS"][1])
                gloz_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["STRESS"][2])
                locx_ptr = c_uint32(self.FUNCTION_LOCALS[i]["STRESS"][0])
                locy_ptr = c_uint32(self.FUNCTION_LOCALS[i]["STRESS"][1])
                locz_ptr = c_uint32(self.FUNCTION_LOCALS[i]["STRESS"][2])
                dk=DimsKernel[i]
                self.swift_fun.EncodeStress(str_ptr, glox_ptr, gloy_ptr, gloz_ptr, locx_ptr, locy_ptr, locz_ptr,dk[0],dk[1],dk[2])
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]:
                str_ptr = ctypes.c_char_p(bytes(i, 'utf-8'))
                glox_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["PARTICLE"][0])
                gloy_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["PARTICLE"][1])
                gloz_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["PARTICLE"][2])
                locx_ptr = c_uint32(self.FUNCTION_LOCALS[i]["PARTICLE"][0])
                locy_ptr = c_uint32(self.FUNCTION_LOCALS[i]["PARTICLE"][1])
                locz_ptr = c_uint32(self.FUNCTION_LOCALS[i]["PARTICLE"][2])
                dk=DimsKernel[i]
                self.swift_fun.EncodeParticle(str_ptr, glox_ptr, gloy_ptr, gloz_ptr, locx_ptr, locy_ptr, locz_ptr,dk[0],dk[1],dk[2])
            
            self.swift_fun.EncodeCommit()
            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):

                glox_ptr = c_uint32(self.globalSensor[0])
                gloy_ptr = c_uint32(self.globalSensor[1])
                gloz_ptr = c_uint32(self.globalSensor[2])
                locx_ptr = c_uint32(self.localSensor[0])
                locy_ptr = c_uint32(self.localSensor[1])
                locz_ptr = c_uint32(self.localSensor[2])
            
                self.swift_fun.EncodeSensors(glox_ptr, gloy_ptr, gloz_ptr, locx_ptr, locy_ptr, locz_ptr)

        self.swift_fun.SyncChange()

        for i in ['SqrAcc', 'SensorOutput']:
            SizeCopy = ArrayResCPU[i].size
            Shape = ArrayResCPU[i].shape
            tempArray = (ctypes.c_float * SizeCopy)()
            Buffer = self.swift_fun.CopyFromGPUMEX(c_uint64(self._IndexDataMetal[i]))
            ctypes.memmove(tempArray, Buffer, SizeCopy * 4) # Metal only supports single precision
            tempArray = np.ctypeslib.as_array(tempArray)
            tempArray = tempArray[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
            ArrayResCPU[i][:,:,:] = np.reshape(tempArray, Shape,order='F')
     
        
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
            tempArray = (ctypes.c_float * SizeBuffer[self._IndexDataMetal[i]])()
            Buffer = self.swift_fun.CopyFromGPUMEX(c_uint64(self._IndexDataMetal[i]))
            ctypes.memmove(tempArray, Buffer, SizeBuffer[self._IndexDataMetal[i]] * 4)
            tempArray = np.ctypeslib.as_array(tempArray)
            tempArray=tempArray[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
            tempArray=np.reshape(tempArray,Shape,order='F')
            ArrayResCPU[i][:,:,:] = np.sum(tempArray,axis=3)/self.ZoneCount
          

        self.swift_fun.freeGPUextern.argtypes = []
        self.swift_fun.freeGPUextern.restype = None

        self.swift_fun.freeGPUextern()

class StaggeredFDTD_3D_With_Relaxation_METAL_MetalCompute(StaggeredFDTD_3D_With_Relaxation_BASE):
    '''
    This version is mainly for Mx processors and which is based in a modified+forked version of metalcompute. As X64 will be phased out, in the future the metalcompute version will take over
    '''
    def __init__(self, arguments):
        global mc
        import metalcompute
        mc = metalcompute
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
            "N1":0, "N2":1, "N3":2, "Limit_I_low_PML":3, "Limit_J_low_PML":4, "Limit_K_low_PML":5, "Limit_I_up_PML":6, "Limit_J_up_PML":7, "Limit_K_up_PML":8, 
            "SizeCorrI":9, "SizeCorrJ":10, "SizeCorrK":11, "PML_Thickness":12, "NumberSources":13, "NumberSensors":14, "TimeSteps":15, 
            "SizePML":16, "SizePMLxp1":17, "SizePMLyp1":18, "SizePMLzp1":19, "SizePMLxp1yp1zp1":20, "ZoneCount":21, "SelRMSorPeak":22, "SelMapsRMSPeak":23, "IndexRMSPeak_ALLV":24, 
            "IndexRMSPeak_Vx":25, "IndexRMSPeak_Vy":26, "IndexRMSPeak_Vz":27, "IndexRMSPeak_Sigmaxx":28, "IndexRMSPeak_Sigmayy":29, "IndexRMSPeak_Sigmazz":30,
            "IndexRMSPeak_Sigmaxy":31, "IndexRMSPeak_Sigmaxz":32, "IndexRMSPeak_Sigmayz":33, "NumberSelRMSPeakMaps":34, "SelMapsSensors":35, "IndexSensor_ALLV":36,
            "IndexSensor_Vx":37, "IndexSensor_Vy":38, "IndexSensor_Vz":39, "IndexSensor_Sigmaxx":40, "IndexSensor_Sigmayy":41, "IndexSensor_Sigmazz":42, "IndexSensor_Sigmaxy":43,
            "IndexSensor_Sigmaxz":44, "IndexSensor_Sigmayz":45, "NumberSelSensorMaps":46, "SensorSubSampling":47, "nStep":48, "TypeSource":49, "CurrSnap":50, "LengthSource":51, "SelK":52,
            "IndexRMSPeak_Pressure":53, "IndexSensor_Pressure":54, "SensorStart":55,
            "IndexSensorMap":0, "SourceMap":1, "MaterialMap": 2,
            # MEX
            "DT":0, "InvDXDTplus":1, "DXDTminus":1+self.MAX_SIZE_PML, "InvDXDTplushp":1+self.MAX_SIZE_PML*2, "DXDTminushp":1+self.MAX_SIZE_PML*3,
            "V_x_x":0, "V_y_x":1, "V_z_x":2, "V_x_y":3, "V_y_y":4, "V_z_y":5, "V_x_z":6, "V_y_z":7, "V_z_z":8, "Vx":9, "Vy":10, "Vz":11, "Rxx":12, "Ryy":13, "Rzz":14, "Rxy":15, "Rxz":16, "Ryz":17,
            "Sigma_x_xx":18, "Sigma_y_xx":19, "Sigma_z_xx":20, "Sigma_x_yy":21, "Sigma_y_yy":22, "Sigma_z_yy":23, "Sigma_x_zz":24, "Sigma_y_zz":25, "Sigma_z_zz":26, 
            "Sigma_x_xy":27, "Sigma_y_xy":28, "Sigma_x_xz":29, "Sigma_z_xz":30, "Sigma_y_yz":31, "Sigma_z_yz":32, "Sigma_xy":33, "Sigma_xz":34, "Sigma_yz":35, "Sigma_xx":36, "Sigma_yy":37, "Sigma_zz": 38,
            "SourceFunctions":39, "LambdaMiuMatOverH":40, "LambdaMatOverH":41, "MiuMatOverH":42, "TauLong":43, "OneOverTauSigma":44, "TauShear":45, "InvRhoMatH":46, "Ox":47, "Oy":48, "Oz":49,
            "Pressure":50, "SqrAcc":51, "SensorOutput":52, 
            }

        self.FUNCTION_LOCALS = {}
        for i in ['MAIN_1', "PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
            self.FUNCTION_LOCALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
        self.FUNCTION_GLOBALS = {}
        for i in ['MAIN_1', "PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
            self.FUNCTION_GLOBALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
                
        self.LENGTH_CONST_UINT = 56
        self.LENGTH_CONST_MEX = 1+self.MAX_SIZE_PML*4

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
        SCode.append("#define MAX_SIZE_PML 101\n")
        extra_params['SCode'] = SCode
        self.ctx = mc.Device(n)
        self.ConstantBufferUINT=np.zeros(self.LENGTH_CONST_UINT,np.uint32)
        self.ConstantBufferMEX=np.zeros(self.LENGTH_CONST_MEX,np.float32)
        print(self.ctx)
    

    def _InitSymbol(self, IP,_NameVar,td, SCode):
        if td == "float":
            self.ConstantBufferMEX[self.C_IND[_NameVar]]=IP[_NameVar]
        elif td == "unsigned int": 
            self.ConstantBufferUINT[self.C_IND[_NameVar]]=IP[_NameVar]
        else:
            raise ValueError("Something was passed incorrectly in symbol initiation.")
        
    def _UpdateSymbol(self, IP, _NameVar,td, SCode):
        if td == "float":
            self.constant_buffer_mex.modify(np.array([IP[_NameVar]],dtype=np.float32),int(self.C_IND[_NameVar]),1)
        elif td == "unsigned int": 
            self.constant_buffer_uint.modify(np.array(IP[_NameVar],dtype=np.uint32),int(self.C_IND[_NameVar]),1)
        else:
            raise ValueError("Something was passed incorrectly in symbol initiation.")
    
    def _InitSymbolArray(self, IP,_NameVar,td, SCode):
        if td == "float":
            self.ConstantBufferMEX[self.C_IND[_NameVar]:self.C_IND[_NameVar]+IP[_NameVar].size]=IP[_NameVar].flatten(order='F')
        elif td == "unsigned int": 
            self.ConstantBufferUINT[self.C_IND[_NameVar]:self.C_IND[_NameVar]+IP[_NameVar].size]=IP[_NameVar].flatten(order='F') 


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
        if Name in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH', 'Ox','Oy','Oz', 'SourceFunctions', 'SensorOutput','SqrAcc']: # float
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
            self.mex_buffer.append(self.ctx.buffer(nSizes*4))
        self.uint_buffer=self.ctx.buffer(self._c_uint_type*4)
        self.constant_buffer_uint=self.ctx.buffer(self.ConstantBufferUINT)
        self.constant_buffer_mex=self.ctx.buffer(self.ConstantBufferMEX)

        self._IndexManip()

        for k in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH',\
                    'Ox','Oy','Oz','SourceFunctions']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "float")

        for k in ['IndexSensorMap','SourceMap','MaterialMap']:
            self._CompleteCopyToGPU(k, arguments, arguments[k].size, "unsigned int")

        if arguments['ManualLocalSize'][0]!=-1:
            self._SET_USER_LOCAL(arguments['ManualLocalSize'])
        else:
            self._CALC_USER_LOCAL("MAIN_1", "STRESS")
            self._CALC_USER_LOCAL("MAIN_1", "PARTICLE")
        
        for j in ["STRESS", "PARTICLE"]:
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
                self._CALC_USER_LOCAL(i, j)
        
        if arguments['ManualGroupSize'][0] != -1:
            self._SET_USER_GLOBAL(arguments['ManualGroupSize'])
        else:
            self._CALC_USER_GROUP_MAIN(arguments, outparams)

        self. _CALC_USER_GROUP_PML(outparams)

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
            for index in range(3):
                self.FUNCTION_LOCALS['MAIN_1'][Type][index] = ManualLocalSize[index] # Can probably change this
    
    def _SET_USER_GLOBAL(self, ManualGlobalSize):
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
               self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ManualGlobalSize[index] 

    def _CALC_USER_GROUP_MAIN(self, arguments, outparams):
        self._outparams = outparams
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
                self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ceil((arguments[('N'+str(index + 1))]-outparams['PML_Thickness']*2) / self.FUNCTION_LOCALS['MAIN_1'][Type][index])
            print("MAIN_1_global_" + Type, "=", str(self.FUNCTION_GLOBALS['MAIN_1'][Type]))
    
    def _CALC_USER_GROUP_PML(self, outparams):
        for Type in ['STRESS', 'PARTICLE']:
            self.FUNCTION_GLOBALS['PML_1'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][0])
            self.FUNCTION_GLOBALS['PML_1'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][1])
            self.FUNCTION_GLOBALS['PML_1'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][2])

            self.FUNCTION_GLOBALS['PML_2'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_2'][Type][0])
            self.FUNCTION_GLOBALS['PML_2'][Type][1] = ceil(outparams['SizeCorrJ'] / self.FUNCTION_LOCALS['PML_2'][Type][1])
            self.FUNCTION_GLOBALS['PML_2'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_2'][Type][2])

            self.FUNCTION_GLOBALS['PML_3'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_3'][Type][0])
            self.FUNCTION_GLOBALS['PML_3'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_3'][Type][1])
            self.FUNCTION_GLOBALS['PML_3'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_3'][Type][2])

            self.FUNCTION_GLOBALS['PML_4'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_4'][Type][0])
            self.FUNCTION_GLOBALS['PML_4'][Type][1] = ceil(outparams['SizeCorrJ'] / self.FUNCTION_LOCALS['PML_4'][Type][1])
            self.FUNCTION_GLOBALS['PML_4'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_4'][Type][2])

            self.FUNCTION_GLOBALS['PML_5'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_5'][Type][0])
            self.FUNCTION_GLOBALS['PML_5'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_5'][Type][1])
            self.FUNCTION_GLOBALS['PML_5'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_5'][Type][2])

            self.FUNCTION_GLOBALS['PML_6'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_6'][Type][0])
            self.FUNCTION_GLOBALS['PML_6'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_6'][Type][1])
            self.FUNCTION_GLOBALS['PML_6'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_6'][Type][2])
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
                print(i + "_global_" + Type + "=", str(self.FUNCTION_GLOBALS[i][Type]))

    def _InitiateCommands(self, AllC):
        prg = self.ctx.kernel(AllC)
        PartsStress=['PML_1','PML_2','PML_3','PML_4','PML_5','PML_6','MAIN_1']
        self.AllStressKernels={}
        for k in PartsStress:
            self.AllStressKernels[k]=prg.function(k+"_StressKernel")

        PartsParticle=['PML_1','PML_2','PML_3','PML_4','PML_5','PML_6','MAIN_1']
        self.AllParticleKernels={}
        for k in PartsParticle:
            self.AllParticleKernels[k]=prg.function(k+"_ParticleKernel")
        
        self.SensorsKernel=prg.function('SensorsKernel')

    def _Execution(self, arguments, ArrayResCPU, ArrayResOP):
        TimeSteps = arguments['TimeSteps']
        InitDict = {'nStep':0, 'TypeSource':int(arguments['TypeSource'])}
        outparams=self._outparams
        DimsKernel={}
        DimsKernel['PML_1']=[outparams['PML_Thickness']*2,
                             outparams['PML_Thickness']*2,
                             outparams['PML_Thickness']*2]
        DimsKernel['PML_2']=[outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_3']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness']*2,
                            outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_4']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness']*2]
        DimsKernel['PML_5']=[outparams['PML_Thickness']*2,
                            outparams['PML_Thickness']*2,
                            outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_6']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['PML_Thickness']*2,
                            outparams['PML_Thickness']*2]
        DimsKernel['MAIN_1']=[outparams['N1']-outparams['PML_Thickness']*2,
                            outparams['N2']-outparams['PML_Thickness']*2,
                            outparams['N3']-outparams['PML_Thickness']*2]

        for nStep in range(TimeSteps):
            InitDict["nStep"] = nStep
            for i in ['nStep', 'TypeSource']:
                self._UpdateSymbol(InitDict, i, 'unsigned int', [])

            self.ctx.init_command_buffer()
            AllHandles=[]
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]:
                nSize=np.prod(DimsKernel[i])
                handle=self.AllStressKernels[i](nSize,self.constant_buffer_uint,
                                               self.constant_buffer_mex,
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

            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]:
                nSize=np.prod(DimsKernel[i])
                handle = self.AllParticleKernels[i](nSize,self.constant_buffer_uint,
                                               self.constant_buffer_mex,
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
            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):
                self.ctx.init_command_buffer()
                handle=self.SensorsKernel(np.prod(self.globalSensor),
                                self.constant_buffer_uint,
                                self.constant_buffer_mex,
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
                self.ctx.commit_command_buffer()   
                self.ctx.wait_command_buffer()
                del handle
 
        
        for i in ['SqrAcc', 'SensorOutput']:
            SizeCopy = ArrayResCPU[i].size
            Shape = ArrayResCPU[i].shape
            Buffer=np.frombuffer(self.mex_buffer[self._IndexDataMetal[i]],dtype=np.float32)[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
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
            Buffer=np.frombuffer(self.mex_buffer[self._IndexDataMetal[i]],dtype=np.float32)[int(self.HOST_INDEX_MEX[self.C_IND[i]][0]):int(self.HOST_INDEX_MEX[self.C_IND[i]][0]+SizeCopy)]
            Buffer=Buffer.reshape(Shape,order='F')
            ArrayResCPU[i][:,:,:] = np.sum(Buffer,axis=3)/self.ZoneCount
          
        del self.constant_buffer_uint
        del self.constant_buffer_mex
        del self.index_mex
        del self.index_uint 
        del self.uint_buffer
        while len(self.mex_buffer)>0:
            handle = self.mex_buffer.pop()
            del handle

def StaggeredFDTD_3D_METAL(arguments):
    if 'arm64' in platform.platform():
        Instance = StaggeredFDTD_3D_With_Relaxation_METAL_MetalCompute(arguments)
    else:
        Instance = StaggeredFDTD_3D_With_Relaxation_METAL_own_swift(arguments)
    Results = Instance.Results
    return Results
