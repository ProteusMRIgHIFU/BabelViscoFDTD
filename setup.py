import os
import platform
import subprocess
import sys
from os import path
from pprint import pprint
from distutils import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

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

        for ext in self.extensions:
            print('ext',ext.name)
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = 'Debug' if _get_env_variable('STAGGERED_DEBUG') == 'ON' else 'Release'
            cmake_args =[
                '-DSTAGGERED_DEBUG=%s' % ('ON' if cfg == 'Debug' else 'OFF'),
                '-DSTAGGERED_OPT=%s' % _get_env_variable('STAGGERED_OPT'),
                '-DSTAGGERED_SINGLE=%s' % ('ON' if 'single' in ext.name else 'OFF') ,
                '-DSTAGGERED_CUDA_SUPPORT=%s' % ('ON' if 'CUDA' in ext.name else 'OFF') ,
                '-DSTAGGERED_PYTHON_SUPPORT=ON',
                '-DCUDA_SAMPLES_LOCATION=%s' %(CUDA_SAMPLES_LOCATION),
                '-DSTAGGERED_PYTHON_C_MODULE_NAME=%s%s' % (ext.name,path.splitext(sysconfig.get_config_var('EXT_SUFFIX'))[0]),
                '-DCMAKE_BUILD_TYPE=%s' % cfg,
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), self.build_temp),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable)]

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

            # Config and build the extension
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.', '--config', cfg],
                                  cwd=self.build_temp)

# The following line is parsed by Sphinx
version = '1.0.0'

print()

modules=[CMakeExtension(c_module_name+'_CUDA_single'),
         CMakeExtension(c_module_name+'_CUDA_double'),
         CMakeExtension(c_module_name+'_single'),
         CMakeExtension(c_module_name+'_double')]
# if platform.system() != 'Windows':
#     modules.append(Extension('_FDTDStaggered3D_with_relaxation_OPENCL_double',
#             ['FDTDStaggered3D_with_relaxation_python.c'],
#             extra_compile_args = ["-DOPENCL"],
#             extra_link_args=["-lOpenCL"]))
#     modules.append(Extension('_FDTDStaggered3D_with_relaxation_OPENCL_single',
#             ['FDTDStaggered3D_with_relaxation_python.c'],
#             extra_compile_args = ["-DOPENCL","-DSINGLE_PREC"],
#             extra_link_args=["-lOpenCL"]))


setup(name='FDTDStaggered3D_with_relaxation',
      packages=['FDTDStaggered3D_with_relaxation'],
      version=version,
      description='GPU/CPU 3D FDTD solution of viscoelastic equation',
      author='Samuel Pichardo',
      author_email='sammeuly@gmail.com',
      #url='https://github.com/m-pilia/disptools',
      #download_url='https://github.com/m-pilia/disptools/archive/v%s.tar.gz' % version,
      keywords=['FDTD', 'CUDA', 'viscoelastic'],
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      install_requires=['numpy>=1.15.1', 'scipy>=1.1.0', 'h5py>=2.9.0','pydicom>=1.3.0'],
      ext_modules=modules,
      py_modules=['__init__','PropagationModel','H5pySimple','StaggeredFDTD_3D_With_Relaxation_CUDA','StaggeredFDTD_3D_With_Relaxation','StaggeredFDTD_3D_With_Relaxation_OPENCL'],
      cmdclass={'build_ext': CMakeBuild},
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: MacOS",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX :: Linux",
      ],
      )
