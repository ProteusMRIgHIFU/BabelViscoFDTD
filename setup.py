import os
import platform
import subprocess
import sys
from os import path
from pprint import pprint
from distutils import sysconfig
from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_ext import build_ext
import numpy as np
import glob
from shutil import copyfile, copytree

from distutils.command.install_headers import install_headers

def PrepareOpenCLKernel():
    #this function merges the kernel code to be usable for opencl
    with open('src'+os.sep+'GPU_KERNELS.h','r') as f:
        GPU_KERNELS=f.readlines()

    with open('_gpu_kernel.c','w') as f:
        for l in GPU_KERNELS:
            if "#include" not in l:
                f.write(l)
            else:
                incfile = l.split('"')[1]
                with open('src'+os.sep+incfile,'r') as g:
                    inclines=g.readlines()
                f.writelines(inclines)
    copyfile('src'+os.sep+'Indexing.h','_indexing.h')


npinc=np.get_include()+os.sep+'numpy'

CUDA_SAMPLES_LOCATION=os.environ.get('CUDA_SAMPLES_LOCATION',None)

# Filename for the C extension module library
c_module_name = '_FDTDStaggered3D_with_relaxation'

# Command line flags forwarded to CMake (for debug purpose)
cmake_cmd_args = []
for f in sys.argv:
    if f.startswith('-D'):
        cmake_cmd_args.append(f)

for f in cmake_cmd_args:
    sys.argv.remove(f)


def _get_env_variable(name, default='OFF'):
    if name not in os.environ.keys():
        return default
    return os.environ[name]


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def build_extensions(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        if platform.system() in ['Darwin']:
            ## There are no easy rules yet in CMAKE to do this through CMakeFiles, but 
            ## since the compilation is very simple, we can do this manually
            print('Compiling Rayleigh Metal interface')
            copytree('src/Metal',self.build_temp )
            command=['xcrun','-sdk', 'macosx', 'metal', '-c','Sources/RayleighMetal/Rayleigh.metal','-o', 'Sources/RayleighMetal/Rayleig.air']
            subprocess.check_call(command,cwd=self.build_temp)
            command=['xcrun','-sdk', 'macosx', 'metallib', 'Sources/RayleighMetal/Rayleig.air','-o', 'Sources/RayleighMetal/Rayleigh.metallib']
            subprocess.check_call(command,cwd=self.build_temp)
            command=['swift','build', '-c', 'release']
            subprocess.check_call(command,cwd=self.build_temp)

            for fn in ['libRayleighMetal.dylib']:
                copyfile(self.build_temp+'/.build/release/'+fn,self.build_lib+'/BabelViscoFDTD/tools/'+fn)
            for fn in ['Rayleigh.metallib']:
                copyfile(self.build_temp+'/Sources/RayleighMetal/'+fn,self.build_lib+'/BabelViscoFDTD/tools/'+fn)
            

        for ext in self.extensions:
            print('ext',ext.name)
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = 'Debug' if _get_env_variable('STAGGERED_DEBUG') == 'ON' else 'Release'
            #'-DSTAGGERED_OMP_SUPPORT=%s' % ('OFF' if platform.system()=='Darwin' else 'ON') ,
            cmake_args =[
                '-DSTAGGERED_DEBUG=%s' % ('ON' if cfg == 'Debug' else 'OFF'),
                '-DSTAGGERED_OPT=%s' % _get_env_variable('STAGGERED_OPT'),
                '-DSTAGGERED_SINGLE=%s' % ('ON' if 'single' in ext.name else 'OFF') ,
                '-DSTAGGERED_OMP_SUPPORT=%s' % ('OFF' if ('OPENCL' in ext.name or platform.system()=='Darwin' ) else 'ON') ,
                '-DSTAGGERED_CUDA_SUPPORT=%s' % ('ON' if 'CUDA' in ext.name else 'OFF') ,
                '-DSTAGGERED_OPENCL_SUPPORT=%s' % ('ON' if 'OPENCL' in ext.name else 'OFF') ,
                '-DSTAGGERED_METAL_SUPPORT=%s' % ('ON' if 'METAL' in ext.name else 'OFF') ,
                '-DSTAGGERED_PYTHON_SUPPORT=ON',
                '-DSTAGGERED_MACOS=%s' % ('ON' if platform.system()=='Darwin' else 'OFF') ,
                '-DCUDA_SAMPLES_LOCATION=%s' %(CUDA_SAMPLES_LOCATION),
                '-DSTAGGERED_PYTHON_C_MODULE_NAME=%s%s' % (ext.name,path.splitext(sysconfig.get_config_var('EXT_SUFFIX'))[0]),
                '-DCMAKE_BUILD_TYPE=%s' % cfg,
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), self.build_temp),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable)]

            # if platform.system()=='Darwin' and 'OPENCL' not in ext.name:
            #     cmake_args.append('-DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang')
            #     cmake_args.append('-DCMAKE_C_COMPILER_WORKS=1')
                #cmake_args.append('-DC_INCLUDE_DIRS=/usr/local/opt/llvm/include')
                #cmake_args.append('-DOPENMP_LIBRARIES=/usr/local/Cellar/llvm/11.0.0/lib/')
                #cmake_args.append('-DOPENMP_INCLUDES=/usr/local/Cellar/llvm/11.0.0/lib/clang/11.0.0/include/')

            if platform.system() == 'Windows':
                plat = ('x64' if platform.architecture()[0] == '64bit' else 'Win32')
                cmake_args += [
                    '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                ]
                if self.compiler.compiler_type == 'msvc':
                    cmake_args += [
                        '-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
                    ]
                else:
                    cmake_args += [
                        '-G', 'MinGW Makefiles',
                    ]

            cmake_args += cmake_cmd_args

            pprint(cmake_args)

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            #force delete of object files, as CMake may sklp compilation
            PrevSO=glob.glob('./**/*FDTDStaggered3D_with_relaxation_python.c.o',recursive=True)
            if len(PrevSO)>0:
                os.remove(PrevSO[0])
            # Config and build the extension
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.', '--config', cfg],
                                  cwd=self.build_temp)



# The following line is parsed by Sphinx
version = '0.9.1'

print()
print('Adding  CPU')
modules=[CMakeExtension(c_module_name+'_single'),
            CMakeExtension(c_module_name+'_double')]
if platform.system() in ['Linux','Windows']:
    print('Adding CUDA')
    modules+=[CMakeExtension(c_module_name+'_CUDA_single'),
             CMakeExtension(c_module_name+'_CUDA_double')]

#if(STAGGERED_MACOS)
#  set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
#  set(MACOS_OMP_INCLUDE "/usr/local/Cellar/llvm/11.0.0/lib/clang/11.0.0/include/omp.h")
#else()
#  set(MACOS_OMP_INCLUDE "")
#endif()

modules.append(CMakeExtension(c_module_name+'_OPENCL_single'))
modules.append(CMakeExtension(c_module_name+'_OPENCL_double'))

if platform.system() in ['Darwin']:
    modules.append(CMakeExtension(c_module_name+'_METAL_single'))


PrepareOpenCLKernel()


setup(name='BabelViscoFDTD',
      #packages=['BabelViscoFDTD'],
      packages=find_packages(),
      version=version,
      description='GPU/CPU 3D FDTD solution of viscoelastic equation',
      author='Samuel Pichardo',
      author_email='sammeuly@gmail.com',
      #url='https://github.com/m-pilia/disptools',
      #download_url='https://github.com/m-pilia/disptools/archive/v%s.tar.gz' % version,
      keywords=['FDTD', 'CUDA', 'viscoelastic'],
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      install_requires=['numpy>=1.15.1', 'scipy>=1.1.0', 'h5py>=2.9.0','pydicom>=1.3.0','pyopencl>=2020.1'],
      ext_modules=modules,
      headers=['_gpu_kernel.c','_indexing.h'],
      cmdclass={'build_ext': CMakeBuild,
                'install_headers': install_headers},
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: MacOS",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX :: Linux",
      ],
      )
