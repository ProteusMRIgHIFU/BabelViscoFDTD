import numpy as np
import os
from pathlib import Path

import time
import tempfile
from shutil import copyfile

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc

# Do I need to put this in a try block?
import pyopencl as cl 
TotalAllocs=0

class BaseFunctions:
    def __init__(self, be):
        if be == "CUDA":
            self.backend = 0
        elif be == "OPENCL":
            self.backend = 1
        elif be == "METAL":
            self.backend = 2
        else:
            self.backend = -1

    def _InitSymbol(self, IP,_NameVar,td,SCode=[]):
        if self.backend == 1:
            if td in ['float','double']:
                res = '__constant ' + td  + ' ' + _NameVar + ' = %0.9g;\n' %(IP[_NameVar])
            else:
                lType =' _PT '
                res = '__constant '+ lType  + _NameVar + ' = %i;\n' %(IP[_NameVar])
            SCode.append(res)
        
    def _InitSymbolArray(self, IP,_NameVar,td,SCode):
        if self.backend == 1:
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
        if self.backend == 1:
            global TotalAllocs
            if td in ['float','unsigned int']:
                f=4
            else: # double
                f=8
            print('Allocating for',Name,dims,'elements')
            ArraysGPUOp[Name]=cl.Buffer(ctx, flags,size=dims*f)
            TotalAllocs+=1            

    def _CreateAndCopyFromMXVarOnGPU(self, Name,ctx,ArraysGPUOp,ArrayResCPU,flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR):
        if self.backend == 1:
            global TotalAllocs
            print('Allocating for',Name,ArrayResCPU[Name].size,'elements')
            ArraysGPUOp[Name]=cl.Buffer(ctx, flags,hostbuf=ArrayResCPU[Name])
            TotalAllocs+=1