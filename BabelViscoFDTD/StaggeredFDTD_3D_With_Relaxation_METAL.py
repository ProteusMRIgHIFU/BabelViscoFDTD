from unicodedata import name
import numpy as np
import os
from pathlib import Path
import _FDTDStaggered3D_with_relaxation_METAL_single as FDTD_single;

import time
import tempfile
from shutil import copyfile

import ctypes

import StaggeredFDTD_3D_With_Relaxation_BASE

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc


class StaggeredFDTD_3D_With_Relaxation_METAL(StaggeredFDTD_3D_With_Relaxation_BASE):
    def __init__(self, arguments):
        #Begin with initializing Swift Functions, etc.
        print('loading',os.path.dirname(os.path.abspath(__file__))+"/tools/libMetalSwift.dylib") # No idea if this is correct, I'll test when I get in the lab
        self.swift_fun = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__))+"/tools/libMetalSwift.dylib")
        # Definition of some constants, etc
        self.MAX_SIZE_PML = 101
        self._c_mex_type = np.zeros(12, np.single)
        self._c_uint_type = np.single(0)
        self.HOST_INDEX_MEX = np.zeros((53, 2), np.single)
        self.HOST_INDEX_UINT = np.zeros((3, 2), np.single)
        self.LENGTH_INDEX_MEX = 53
        self.LENGTH_INDEX_UINT = 3
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
            # MEX
            "DT":0, "InvDXDTplus":1, "DXDTminus":1+self.MAX_SIZE_PML, "InvDXDTplushp":1+self.MAX_SIZE_PML*2, "DXDTminushp":1+self.MAX_SIZE_PML*3,
            "V_x_x":0, "V_y_x":1, "V_z_x":2, "V_x_y":3, "V_y_y":4, "V_z_y":5, "V_x_z":6, "V_y_z":7, "V_z_z":8, "Vx":9, "Vy":10, "Vz":11, "Rxx":12, "Ryy":13, "Rzz":14, "Rxy":15, "Rxz":16, "Ryz":17,
            "Sigma_x_xx":18, "Sigma_y_xx":19, "Sigma_z_xx":20, "Sigma_x_yy":21, 
            }
        self.LENGTH_CONST_UINT = 56
        self.LENGTH_CONST_MEX = 1+self.MAX_SIZE_PML*4
        # Defines functions sent to Swift
        self.swift_fun.InitializeMetalDevices.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(ctypes.c_int)]
        self.swift_fun.ConstantBuffers.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.pointer(ctypes.c_int)]
        self.swift_fun.SymbolInitiation_uint.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32)]
        self.swift_fun.SymbolInitiation_mex.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float)]
        self.swift_fun.CompleteCopyMEX.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint64), # The UInt64 can probably get changed
            ctypes.POINTER(ctypes.c_uint64)]
        self.swift_fun.CompleteCopyUInt.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64)] # The UInt64 can probably get changed
        self.swift_fun.IndexManipMEX.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32)]
        self.swift_fun.IndexManipUInt.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32)]

        # These actually should probably be moved somewhere else
        self.swift_fun.InitializeMetalDevices(arguments['DefaultGPUDeviceName'], len(arguments['DefaultGPUDeviceName']))
        self.swift_fun.ConstantBuffers(self.LENGTH_CONST_UINT, self.LENGTH_CONST_MEX)

        extra_params = {"BACKEND":"METAL"}
        super.__init__(arguments, extra_params)
        

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):
        if td == "float":
            self.swift_fun.SymbolInitiation_mex(C_IND[_NameVar], IP[_NameVar])
        elif td == "unsigned int": #No idea if this will work or if it needs to be changed
            self.swift_fun.SymbolInitiation_uint(C_IND[_NameVar], IP[_NameVar])
        else:
            raise ValueError("Something was passed incorrectly in symbol initiation.")
        
    
    def _InitSymbolArray(self, IP,_NameVar,td,SCode=[]):
        global __Limit
        global C_IND
        if td == "float":
            for i in range(IP[_NameVar].size):
                self.swift_fun.SymbolInitiation_mex(C_IND[str(i) + _NameVar], IP[_NameVar][i]) #Double check second arg
        elif td == "unsigned int": 
            for i in range(IP[_NameVar].size):
                self.swift_fun.SymbolInitiation_uint(C_IND[str(i) + _NameVar], IP[_NameVar][i])  #Second arg
        # I think this way runs faster since it's not doing the if check every loop?


    def _ownGpuCalloc(self, Name,ctx,td,dims,ArraysGPUOp,flags):
        print("Allocating for", Name, dims, "elements")
        global C_IND # Would it be easier to make it part of the class?
        if td == "float":
            self.HOST_INDEX_MEX[C_IND[Name]][0] = self._c_mex_type[self._IndexDataMetal[Name]]
            self.HOST_INDEX_MEX[C_IND[Name]][1] = dims # Not sure if dims is equivalent to _size * INHOST(ZoneCount)
            self._c_mex_type[self._IndexDataMetal[Name]] += dims
        elif td == "unsigned int":
            self.HOST_INDEX_UINT[C_IND[Name]][0] = self._c_uint_type
            self.HOST_INDEX_UINT[C_IND[Name]][1] = dims
            self._c_uint_type += dims

    
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags):
        print("Allocating for", Name, ArrayResCPU[Name].size, "elements")
        global C_IND
        SizeCopy = ArrayResCPU[Name].size
        if Name in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH','TauLong','OneOverTauSigma','TauShear','InvRhoMatH', 'Ox','Oy','Oz', 'SourceFunctions']: # float
            self.HOST_INDEX_MEX[C_IND[Name]][0] = self._c_mex_type[self._IndexDataMetal[Name]]
            self.HOST_INDEX_MEX[C_IND[Name]][1] = SizeCopy
            self._c_mex_type[self._IndexDataMetal[Name]] += SizeCopy
        elif Name in ['IndexSensorMap','SourceMap','MaterialMap']: # unsigned int
            self.HOST_INDEX_UINT[C_IND[Name]][0] = self._c_uint_type
            self.HOST_INDEX_UINT[C_IND[Name]][1] = SizeCopy
            self._c_uint_type += SizeCopy

    def _IndexManip(self):
        # Can this be moved?
        for i in range(self.LENGTH_INDEX_MEX):
            data = 0xFFFFFFFF & self.HOST_INDEX_UINT[i][0] # Not exactly sure if this works
            data2 = self.HOST_INDEX_MEX[i][0]>>32
            self.swift_fun.IndexManipMEX(data, data2, i)

        for i in range(self.LENGTH_INDEX_UINT):
            data = 0xFFFFFFFF & self.HOST_INDEX_UINT[i][0] # Not exactly sure if this works
            data2 = self.HOST_INDEX_MEX[i][0]>>32
            self.swift_fun.IndexManipUInt(data, data2, i)

    def _Execution(self, arguments):
        raise NotImplementedError("This block must be implemented in a child class")

def StaggeredFDTD_3D_METAL(arguments):
    os.environ['__BabelMetal'] =(os.path.dirname(os.path.abspath(__file__))+os.sep+'tools')
    print(os.environ['__BabelMetal'])
    os.environ['__BabelMetalDevice'] = arguments['DefaultGPUDeviceName']
    IncludeDir=str(Path(__file__).parent.absolute())+os.sep

    filenames = [IncludeDir+'_indexing.h',IncludeDir+'_gpu_kernel.c']

    kernbinfile=IncludeDir+'tools'+os.sep+'Rayleigh.metallib'
    

    if (type(arguments)!=dict):
        raise TypeError( "The input parameter must be a dictionary")

    for key in arguments.keys():
        if type(arguments[key])==np.ndarray:
            if np.isfortran(arguments[key])==False:
                #print "StaggeredFDTD_3D: Converting ndarray " + key + " to Fortran convention";
                arguments[key] = np.asfortranarray(arguments[key]);
        elif type(arguments[key])!=str:
            arguments[key]=np.array((arguments[key]))
    t0 = time.time()
    arguments['PI_OCL_PATH']='' #unused in METAL but needed in the low level function for completeness
    arguments['kernelfile']=''
    arguments['kernbinfile']=kernbinfile
    
    if arguments['DT'].dtype==np.dtype('float32'):
        Results= FDTD_single.FDTDStaggered_3D(arguments)
    else:
        raise SystemError("Metal backend only supports single precision")
    t0=time.time()-t0
    print ('Time to run low level FDTDStaggered_3D =', t0)
    return Results
