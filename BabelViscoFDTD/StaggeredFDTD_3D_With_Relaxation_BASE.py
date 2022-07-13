import numpy as np
import os
from pathlib import Path

import time
import tempfile
from shutil import copyfile

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc

class BaseCreation:
    def _PrepParamsForKernel(arguments):
        print("Insert Func Here")

    def _InitSymbol(IP,_NameVar,td,SCode):
        print("Insert Func Here")
        
    def _InitSymbolArray(IP,_NameVar,td,SCode):
        print("Insert Func Here")
        
    def _ownGpuCalloc(Name,ctx,td,dims,ArraysGPUOp,flags=cl.mem_flags.READ_WRITE):
        print("Insert Func Here")

    def _CreateAndCopyFromMXVarOnGPU(Name,ctx,ArraysGPUOp,ArrayResCPU,flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR):
        print("Insert Func Here")
        
    