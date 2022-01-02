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
from distutils.unixccompiler import UnixCCompiler
from shutil import copyfile, copytree,rmtree

from distutils.command.install_headers import install_headers

version = '0.9.3'
npinc=np.get_include()+os.sep+'numpy'
# Filename for the C extension module library
c_module_name = '_FDTDStaggered3D_with_relaxation'

bRayleighMetalCompiled=False
def CompileRayleighMetal(build_temp,build_lib):
    global bRayleighMetalCompiled
    if not bRayleighMetalCompiled:
        print('Compiling Metal Rayleigh')
        ## There are no easy rules yet in CMAKE to do this through CMakeFiles, but 
        ## since the compilation is very simple, we can do this manually
        print('Compiling Rayleigh Metal interface')
        copytree('src/Metal',build_temp )
        for fn in ['Indexing.h','GPU_KERNELS.h','kernelparamsMetal.h','StressKernel.h',
                    'ParticleKernel.h','SensorsKernel.h','kernelparamsMetal.h']:
            copyfile('src'+os.sep+fn,build_temp+'/Sources/RayleighMetal/'+fn)

        command=['xcrun','-sdk', 'macosx', 'metal','-c','Sources/RayleighMetal/Rayleigh.metal','-o', 'Sources/RayleighMetal/Rayleig.air']
        subprocess.check_call(command,cwd=build_temp)
        command=['xcrun','-sdk', 'macosx', 'metallib', 'Sources/RayleighMetal/Rayleig.air','-o', 'Sources/RayleighMetal/Rayleigh.metallib']
        subprocess.check_call(command,cwd=build_temp)
        command=['swift','build','-c', 'release']
        subprocess.check_call(command,cwd=build_temp)
        for fn in ['libRayleighMetal.dylib']:
            copyfile(build_temp+'/.build/release/'+fn,build_lib+'/BabelViscoFDTD/tools/'+fn)
        for fn in ['Rayleigh.metallib']:
            copyfile(build_temp+'/Sources/RayleighMetal/'+fn,build_lib+'/BabelViscoFDTD/tools/'+fn)
        bRayleighMetalCompiled=True

def PrepareOpenCLKernel():
    #this function merges the kernel code to be usable for opencl
    with open('src'+os.sep+'GPU_KERNELS.h','r') as f:
        GPU_KERNELS=f.readlines()

    with open('BabelViscoFDTD'+os.sep+'_gpu_kernel.c','w') as f:
        for l in GPU_KERNELS:
            if "#include" not in l:
                f.write(l)
            else:
                incfile = l.split('"')[1]
                with open('src'+os.sep+incfile,'r') as g:
                    inclines=g.readlines()
                f.writelines(inclines)
    copyfile('src'+os.sep+'Indexing.h','BabelViscoFDTD'+os.sep+'_indexing.h')
    
if 'arm64' not in platform.platform():
    CUDA_SAMPLES_LOCATION=os.environ.get('CUDA_SAMPLES_LOCATION',None)



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
                CompileRayleighMetal(self.build_temp,self.build_lib)
                ## There are no easy rules yet in CMAKE to do this through CMakeFiles, but 
                ## since the compilation is very simple, we can do this manually
                

            for ext in self.extensions:
                print('ext',ext.name)
                extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
                cfg = 'Debug' if _get_env_variable('STAGGERED_DEBUG') == 'ON' else 'Release'
                if ext.name =='pi_ocl':
                    cmake_args =['-DCMAKE_BUILD_TYPE=%s' % cfg,
                                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir+os.sep+'BabelViscoFDTD')]
                else:
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

                pprint(["cmake_args",cmake_args])

                if not os.path.exists(self.build_temp):
                    os.makedirs(self.build_temp)
                else:
                    rmtree(self.build_temp)
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
    print('Adding  CPU')
    modules=[CMakeExtension(c_module_name+'_single'),
                CMakeExtension(c_module_name+'_double')]
    if platform.system() in ['Linux','Windows']:
        print('Adding CUDA')
        modules+=[CMakeExtension(c_module_name+'_CUDA_single'),
                CMakeExtension(c_module_name+'_CUDA_double')]


    if platform.system() in ['Darwin']:
        modules.append(CMakeExtension(c_module_name+'_OPENCL_single',extra_compile_args = ["-mmacosx-version-min=11.3"]))
        modules.append(CMakeExtension(c_module_name+'_OPENCL_double',extra_compile_args = ["-mmacosx-version-min=11.3"]))
        modules.append(CMakeExtension(c_module_name+'_METAL_single',extra_compile_args = ["-mmacosx-version-min=11.3"]))
        modules.append(CMakeExtension('pi_ocl',cmake_lists_dir='pi_ocl'))
    else:
        modules.append(CMakeExtension(c_module_name+'_OPENCL_single'))
        modules.append(CMakeExtension(c_module_name+'_OPENCL_double'))



    PrepareOpenCLKernel()

    setup(name='BabelViscoFDTD',
        packages=['BabelViscoFDTD','BabelViscoFDTD.tools','pi_ocl'],
        include_package_data=True,
        package_data={'BabelViscoFDTD': ['_gpu_kernel.c','_indexing.h']},
        version=version,
        description='GPU/CPU 3D FDTD solution of viscoelastic equation',
        author='Samuel Pichardo',
        author_email='samuel.pichardo@ucalgary.ca',
        keywords=['FDTD', 'CUDA', 'viscoelastic'],
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        install_requires=['numpy>=1.15.1', 'scipy>=1.1.0', 'h5py>=2.9.0','pydicom>=1.3.0','pyopencl>=2020.1'],
        ext_modules=modules,
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
else:
    PrepareOpenCLKernel()

    class DarwinInteropBuildExt(build_ext):
        def initialize_options(self):

            # add support for ".mm" files
            UnixCCompiler.src_extensions.append(".mm")
            UnixCCompiler.language_map[".mm"] = "objc"

            # then intercept and patch the compile and link methods to add needed flags
            unpatched_compile = UnixCCompiler._compile
            unpatched_link = UnixCCompiler.link

            def patched_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
                # define language specific compile flags here
                if ext == ".cpp":
                    patched_postargs = extra_postargs + ["-std=c++11"]
                elif ext == ".mm":
                    patched_postargs = extra_postargs + ["-std=c++11",
                        "-ObjC++",
                    ]
                else:
                    patched_postargs = extra_postargs
                unpatched_compile(self, obj, src, ext, cc_args, patched_postargs, pp_opts)

            def patched_link(
                    self,
                    target_desc,
                    objects,
                    output_filename,
                    output_dir=None,
                    libraries=None,
                    library_dirs=None,
                    runtime_library_dirs=None,
                    export_symbols=None,
                    debug=0,
                    extra_preargs=None,
                    extra_postargs=None,
                    build_temp=None,
                    target_lang=None,
            ):
                # define additional linking arguments here if needed
                existing_postargs = extra_postargs or []
                unpatched_link(
                    self,
                    target_desc,
                    objects,
                    output_filename,
                    output_dir,
                    libraries,
                    library_dirs,
                    runtime_library_dirs,
                    export_symbols,
                    debug,
                    extra_preargs,
                    existing_postargs ,
                    build_temp,
                    target_lang,
                )
            UnixCCompiler._compile = patched_compile
            UnixCCompiler.link = patched_link
            super().initialize_options()

        def build_extensions(self):
            print('building extension')
            if platform.system() in ['Darwin']:
                CompileRayleighMetal(self.build_temp,self.build_lib)
            super().build_extensions()

    from mmap import PAGESIZE
    bIncludePagememory=np.__version__ >="1.22.0"
    ext_modules=[Extension(c_module_name+'_single', 
                    ["src/FDTDStaggered3D_with_relaxation_python.c"],
                    define_macros=[("SINGLE_PREC",None),
                                ("USE_OPENMP",None)],
                    extra_compile_args=['-Xclang','-fopenmp'],
                    extra_link_args=['-lomp'],
                    include_dirs=[npinc]),
                Extension(c_module_name+'_double', 
                    ["src/FDTDStaggered3D_with_relaxation_python.c"],
                    define_macros=[("USE_OPENMP",None)],
                    extra_compile_args=['-Xclang','-fopenmp'],
                    extra_link_args=['-lomp'],
                    include_dirs=[npinc]),
                Extension(c_module_name+'_METAL_single', 
                    ["src/FDTDStaggered3D_with_relaxation_python.cpp",
                    "src/mtlpp/mtlpp.mm"],
                    define_macros=[("SINGLE_PREC",None),
                                ("METAL",None)],
                    include_dirs=[npinc],
                    extra_compile_args=['-std=c++11','-mmacosx-version-min=11.5'],
                    extra_link_args=['-Wl',
                                    '-framework',
                                    'Metal',
                                    '-Wl',
                                    '-framework',
                                    'MetalKit',
                                    '-Wl',
                                    '-framework',
                                    'Cocoa',
                                    '-Wl',
                                    '-framework',
                                    'CoreFoundation',
                                    '-fobjc-link-runtime'])]
    if bIncludePagememory:
        ext_modules.append(Extension('BabelViscoFDTD.tools._page_memory', 
                            ["src/page_memory.c"],
                            define_macros=[("PAGE_SIZE",str(PAGESIZE))],
                            include_dirs=[npinc]))

    setup(name="BabelViscoFDTD",
            version=version,
            packages=['BabelViscoFDTD','BabelViscoFDTD.tools'],
            description='GPU/CPU 3D FDTD solution of viscoelastic equation',
            package_data={'BabelViscoFDTD': ['_gpu_kernel.c','_indexing.h']},
            author_email='samuel.pichardo@ucalgary.ca',
            keywords=['FDTD', 'CUDA', 'viscoelastic'],
            long_description=open('README.md').read(),
            long_description_content_type='text/markdown',
            cmdclass={'build_ext': DarwinInteropBuildExt},
            ext_modules=ext_modules,
            zip_safe=False,
            classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: MacOS",
                "Operating System :: Microsoft :: Windows",
                "Operating System :: POSIX :: Linux",
            ])

