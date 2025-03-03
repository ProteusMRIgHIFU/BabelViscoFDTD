'''
Samuel Pichardo, Ph.D.
University of Calgary

Very simple GPU-accelerated Rayleigh integral with support to OpenCL and CUDA

CUDA is automatically selected if running Windows or Linux, while OpenCL is selected in MacOs
'''

import gc
import numpy as np
import os
from pathlib import Path
import sys
from sysconfig import get_paths
import ctypes
import sys 
import platform

_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir =  os.path.abspath(os.path.join(os.path.dirname(__file__)))
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

Platforms = None
queue = None
prgcl = None
ctx = None
clp = None
sel_device = None

def SpeedofSoundWater(Temperature):
    Xcoeff =  [0.00000000314643 ,-0.000001478,0.000334199,-0.0580852,5.03711,1402.32]
    speed = np.polyval(Xcoeff,Temperature)
    return speed 

def GenerateSurface(lstep,Diam,Foc):
    Tx = {}
    rInt=0
    rExt=Diam/2

    Beta1= np.arcsin(rInt/Foc)
    Beta2= np.arcsin(rExt/Foc)

    DBeta= Beta2-Beta1

    ArcC = DBeta*Foc

    nrstep = np.ceil(ArcC/lstep);

    BetaStep = DBeta/nrstep;

    BetaC = np.arange(Beta1+BetaStep/2,Beta1+BetaStep*(1/2 + nrstep),BetaStep)
    
    Ind=0

    SingElem = np.zeros((0,3))
    N = np.zeros((0,3))
    ds = np.zeros((0,1))

    VertDisplay=  np.zeros((0,3))
    FaceDisplay= np.zeros((0,4),int)

    for nr in range(len(BetaC)):

        Perim = np.sin(BetaC[nr])*Foc*2*np.pi

        nAlpha = np.ceil(Perim/lstep)
        sAlpha = 2*np.pi/nAlpha

        AlphaC = np.arange(sAlpha/2,sAlpha*(1/2 + nAlpha ),sAlpha)


        SingElem=np.vstack((SingElem,np.zeros((len(AlphaC),3))))
        N  = np.vstack((N,np.zeros((len(AlphaC),3))))
        ds = np.vstack((ds,np.zeros((len(AlphaC),1))))

        VertDisplay= np.vstack((VertDisplay,np.zeros((len(AlphaC)*4,3))))
        FaceDisplay= np.vstack((FaceDisplay,np.zeros((len(AlphaC),4),int)))


        zc = -np.cos(BetaC[nr])*Foc
        Rc = np.sin(BetaC[nr])*Foc

        B1 = BetaC[nr]-BetaStep/2
        B2 = BetaC[nr]+BetaStep/2
        if nr==0:
            Rc1=0
        else:
            Rc1 = np.sin(B1)*Foc
        
        Rc2 = np.sin(B2)*Foc

        zc1 =-np.cos(B1)*Foc
        zc2 =-np.cos(B2)*Foc
        
        SingElem[Ind:,0] = Rc*np.cos(AlphaC)
        SingElem[Ind:,1] = Rc*np.sin(AlphaC)
        SingElem[Ind:,2] = zc
        
        A1 = AlphaC-sAlpha/2;
        A2 = AlphaC+sAlpha/2;
        ds[Ind:,0]=Foc**2 *(np.cos(B1) - np.cos(B2))*(A2-A1)
        N[Ind:,:] =SingElem[Ind:,:]/np.repeat(np.linalg.norm(SingElem[Ind:,:],axis=1).reshape((len(AlphaC),1)),3,axis=1)
        VertDisplay[Ind*4::4,0]= Rc1*np.cos(A1)
        VertDisplay[Ind*4::4,1]= Rc1*np.sin(A1)
        VertDisplay[Ind*4::4,2]= zc1

        VertDisplay[Ind*4+1::4,0]= Rc1*np.cos(A2)
        VertDisplay[Ind*4+1::4,1]= Rc1*np.sin(A2)
        VertDisplay[Ind*4+1::4,2]= zc1

        VertDisplay[Ind*4+2::4,0]= Rc2*np.cos(A1)
        VertDisplay[Ind*4+2::4,1]= Rc2*np.sin(A1)
        VertDisplay[Ind*4+2::4,2]= zc2

        VertDisplay[Ind*4+3::4,0]= Rc2*np.cos(A2)
        VertDisplay[Ind*4+3::4,1]= Rc2*np.sin(A2)
        VertDisplay[Ind*4+3::4,2]= zc2

        FaceDisplay[Ind:,0] =(Ind+np.arange(len(AlphaC)))*4
        FaceDisplay[Ind:,1] =(Ind+np.arange(len(AlphaC)))*4+1
        FaceDisplay[Ind:,2] =(Ind+np.arange(len(AlphaC)))*4+3
        FaceDisplay[Ind:,3] =(Ind+np.arange(len(AlphaC)))*4+2
        Ind+=len(AlphaC)

    Tx['center'] = SingElem 
    Tx['ds'] = ds
    Tx['normal'] = N
    Tx['VertDisplay'] = VertDisplay 
    Tx['FaceDisplay'] = FaceDisplay 
    Tx['Beta1']=Beta1
    Tx['Beta2']=Beta2
    return Tx

def GenerateFocusTx(f,Foc,Diam,c,PPWSurface=4):
    wavelength = c/f;
    lstep = wavelength/PPWSurface;

    Tx = GenerateSurface(lstep,Diam,Foc)
    return Tx

def GetKernelFiles():

    kernel_files = [
       os.path.join(resource_path(), 'rayleigh.cpp'),
       os.path.join(resource_path(), 'BHTE.cpp'),
    ]
    
    with open(os.path.join(resource_path(), 'rayleighAndBHTE.hpp'), 'r') as f:
        header = f.read()
            
    return header, kernel_files

def AssembleKernel(preamble,kernel_files,mlx_backend=False):
    
    if mlx_backend:
        kernels = {}
        for kf in kernel_files:
            with open(kf['file'], 'r') as f:
                lines = f.readlines()
            kernel_code = ''.join(lines[:-1]) # Remove last bracket
            
            kernel = clp.fast.metal_kernel(name = kf['name'],
                                        input_names = kf['input_names'],
                                        output_names = kf['output_names'],
                                        atomic_outputs = kf['atomic_outputs'],
                                        header = preamble,
                                        source = kernel_code)
            
            kernels[kf['name']] = kernel
                
        return kernels
    else:
        kernel_codes = [preamble]
        for k_file in kernel_files:
            with open(k_file, 'r') as f:
                kernel_code = f.read()
                kernel_codes.append(kernel_code)
        complete_kernel = '\n'.join(kernel_codes)
        
        return complete_kernel
    
def InitCUDA(DeviceName=None):
    global clp
    global prgcuda
    
    import cupy as cp
    clp = cp
    
    # Get kernel files
    header,kernel_files = GetKernelFiles()
    
    # Obtain list of gpu devices
    devCount = cp.cuda.runtime.getDeviceCount()
    if devCount == 0:
        raise SystemError("There are no CUDA devices.")
    
    # Select device that matches specified name
    if DeviceName is not None:
        sel_device = None
        for deviceID in range(0, devCount):
            d=cp.cuda.runtime.getDeviceProperties(deviceID)
            if DeviceName in d['name'].decode('UTF-8'):
                sel_device=cp.cuda.Device(deviceID)
                break
        sel_device.use()
    
    # Assemble kernel code
    kernel_code = AssembleKernel(preamble="#define _CUDA\n"+header,kernel_files=kernel_files)
    

    # Build program from source code   
    # Windows sometimes has issues finding CUDA
    if platform.system()=='Windows':
        sys.executable.split('\\')[:-1]
        options=('-I',os.path.join(os.getenv('CUDA_PATH'),'Library','Include'),
                    '-I',str(resource_path()),
                    '--ptxas-options=-v')
    else:
        options=('-I',str(resource_path()))
    
    prgcuda = cp.RawModule(code=kernel_code,options=options)
 
def InitOpenCL(DeviceName=None):
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global clp
    global sel_device
    
    import pyopencl as pocl
    clp = pocl
    
    # Get kernel files
    header,kernel_files = GetKernelFiles()
    
    # Obtain list of openCL platforms
    Platforms=pocl.get_platforms()
    if len(Platforms)==0:
        raise SystemError("No OpenCL platforms")
    
    # Obtain list of available devices and select one
    sel_device=None
    for device in Platforms[0].get_devices():
        print(device.name)
        if DeviceName in device.name:
            sel_device=device
    if sel_device is None:
        raise SystemError("No OpenCL device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', sel_device.name)
        
    # Create context for selected device
    ctx = pocl.Context([sel_device])
    
    # Assemble kernel code
    kernel_code = AssembleKernel(preamble="#define _OPENCL\n"+header,kernel_files=kernel_files)
    
    # Build kernel
    prgcl = pocl.Program(ctx,kernel_code).build()


    # Create command queue for selected device
    queue = pocl.CommandQueue(ctx)

def InitMetal(DeviceName=None):
    global ctx
    global prgcl 
    global clp
    global sel_device
    global swift_fun
    
    import metalcomputebabel as mc
    clp = mc
    
    
    # Loads METAL interface for Rayleigh
    os.environ['__BabelMetal'] =os.path.dirname(os.path.abspath(__file__))
    print('loading',os.path.dirname(os.path.abspath(__file__))+"/libBabelMetal.dylib")
    swift_fun = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__))+"/libBabelMetal.dylib")

    swift_fun.ForwardSimpleMetal.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)]


    swift_fun.PrintMetalDevices()
    print("loaded Metal",str(swift_fun))

    def StartMetaCapture(deviceName='M1'):
        os.environ['__BabelMetalDevice'] =deviceName
        swift_fun.StartCapture()

    def Stopcapture():
        swift_fun.Stopcapture()

    # Setup for BHTE
    header,kernel_files = GetKernelFiles()
    
    # Obtain list of gpu devices     
    devices = mc.get_devices()
    sel_device=None
    for n,dev in enumerate(devices):
        if DeviceName in dev.deviceName:
            sel_device=dev
            break
    if sel_device is None:
        raise SystemError("No Metal device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', dev.deviceName)
    
    # Create context for selected device
    ctx = mc.Device(n)
    print(ctx)
    if 'arm64' not in platform.platform():
        ctx.set_external_gpu(1) 
        
    # Assemble kernel code
    kernel_code = AssembleKernel(preamble="#define _METAL\n"+header,kernel_files=[kernel_files[1]])
    
    # Build program from source code
    prgcl = ctx.kernel(kernel_code)

def InitMLX(DeviceName=None):
    global clp
    global prgcl 
    global sel_device
    
    import mlx.core as mx
    clp = mx
    
    # Setup for BHTE
    header,kernel_files = GetKernelFiles()
    
    # select gpu devices     
    sel_device = mx.default_device()
    print('Selecting device: ', sel_device)
    
    # MLX requires functions to be separate
    kernel_functions = [{'name': 'ForwardPropagationKernel',
                        'file': kernel_files[0],
                        'input_names': ["mr2", "c_wvnb_real", "c_wvnb_imag", "MaxDistance", "mr1",
                                        "r2pr","r1pr","a1pr","u1_real","u1_imag","mr1step"],
                        'output_names': ["py_data_u2_real","py_data_u2_imag"],
                        'atomic_outputs': False},
                        {'name': 'BHTEFDTDKernel',
                        'file': kernel_files[1],
                        'input_names': ["d_input","d_input2","d_bhArr","d_perfArr","d_labels","d_Qarr",
                                        "d_pointsMonitoring","floatParams","intparams"],
                        'output_names': ["d_output","d_output2","d_MonitorSlice","d_Temppoints"],
                        'atomic_outputs': False}]
    
    # Assemble kernel
    prgcl = AssembleKernel(preamble="#define _MLX\n"+header,kernel_files=kernel_functions,mlx_backend=True)
    
def ForwardSimple(cwvnb,center,ds,u0,rf,MaxDistance=-1.0,u0step=0,gpu_backend='OpenCL',gpu_device='M1'): #MacOsPlatform='Metal',deviceMetal='6800'):
    '''
    MAIN function to call for ForwardRayleigh , returns the complex values of particle speed
    cwvnb is the complex speed of sound
    center is an [Mx3] array with the position of the decomposed transducer elements
    ds is an [M] array with the transducer element surface area
    u0 is [M] complex array with the particle speed at each transducer element
    rf is [Nx3] array with the positons where Rayleigh integral will be calculated
    
    Function returns a [N] complex array of particle speed at locations rf
    
    '''
    
    mr2=rf.shape[0]
    
    if u0step!=0:
        mr1=u0step
        assert(mr1*rf.shape[0]==u0.shape[0])
        assert(mr1==center.shape[0])
        assert(mr1==ds.shape[0])
    else:
        mr1=center.shape[0]
    
    u2 = np.zeros((rf.shape[0]),dtype=np.complex64)
    
    # Setup for kernel call
    if gpu_backend == 'CUDA':
        # Transfer input data to gpu
        d_r2pr= clp.asarray(rf)
        d_r1pr= clp.asarray(center)
        d_a1pr= clp.asarray(ds)
        d_u1complex= clp.asarray(u0)
        
        # Get kernel function
        ForwardPropagationKernel = prgcuda.get_function("ForwardPropagationKernel")
        
        # Determine kernel call dimensions
        CUDA_THREADBLOCKLENGTH = 512 
        MAX_ELEMS_IN_CONSTANT = 2730
        dimThreadBlock=(CUDA_THREADBLOCKLENGTH, 1,1)
        
        nBlockSizeX = int((mr2 - 1) / dimThreadBlock[0]) + 1
        nBlockSizeY = 1

        if (nBlockSizeX > 65534 ):
            nBlockSizeY = int(nBlockSizeX / 65534)
            if (nBlockSizeX %65534 !=0):
                nBlockSizeY+=1
            nBlockSizeX = int(nBlockSizeX/nBlockSizeY)+1
        
        dimBlockGrid=(nBlockSizeX, nBlockSizeY,1)

        # Output array
        d_u2complex= clp.zeros(rf.shape[0],clp.complex64)
        
        # Deploy kernel
        ForwardPropagationKernel(dimBlockGrid,
                                dimThreadBlock,
                                (mr2, 
                                 cwvnb,
                                 MaxDistance,
                                 mr1,
                                 d_r2pr, 
                                 d_r1pr,
                                 d_a1pr,
                                 d_u1complex, 
                                 d_u2complex, 
                                 u0step))
        
        # Change back to numpy array
        u2 = d_u2complex.get()

        return u2
        
    elif gpu_backend == 'OpenCL':
        mf = clp.mem_flags
        
        # Transfer input data to gpu
        d_r2pr = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rf)
        d_r1pr = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=center)
        d_u1realpr=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.real(u0).copy())
        d_u1imagpr=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.imag(u0).copy())
        d_a1pr = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ds)
        
        # Get kernel Function
        knl = prgcl.ForwardPropagationKernel
        
    elif gpu_backend == 'Metal':
        # Determine if data is aligned
        os.environ['__BabelMetalDevice'] = gpu_device
        bUseMappedMemory=0
        if np.__version__ >="1.22.0":
            if 'arm64' in platform.platform() and\
                np.core.multiarray.get_handler_name(center)=="page_data_allocator":
                bUseMappedMemory=1
            #We assume arrays were allocated with page_data_allocator to have aligned date

        # Convert inputs to numpy arrrays
        mr1=np.array([mr1])
        mr2=np.array([mr2])
        u0step_a=np.array([u0step])
        MaxDistance_a=np.array([MaxDistance]).astype(np.float32)
        ibUseMappedMemory =np.array([bUseMappedMemory])
        cwvnb_real=np.array([np.real(cwvnb)])
        cwvnb_imag=np.array([np.imag(cwvnb)])
        
        # Create pointers to use with swift call
        mr1_ptr = mr1.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        mr2_ptr = mr2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        u0step_ptr = u0step_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        bUseMappedMemory_ptr =ibUseMappedMemory.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        cwvnb_real_ptr = cwvnb_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cwvnb_imag_ptr = cwvnb_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cwvnb_imag_ptr = cwvnb_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        MaxDistance_ptr= MaxDistance_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        r1_ptr=center.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        r2_ptr=rf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        a1_ptr=ds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        u1_real_ptr=np.real(u0).copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        u1_imag_ptr=np.imag(u0).copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        deviceName_ptr=ctypes.c_char_p(gpu_device.encode())
        u2_real = np.zeros(rf.shape[0],np.float32)
        u2_imag = np.zeros(rf.shape[0],np.float32)
        u2_real_ptr = u2_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        u2_imag_ptr = u2_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call swift code that already handles looping
        ret = swift_fun.ForwardSimpleMetal(mr2_ptr,
                                cwvnb_real_ptr,
                                cwvnb_imag_ptr,
                                MaxDistance_ptr,
                                mr1_ptr,
                                r2_ptr,
                                r1_ptr,
                                a1_ptr,
                                u1_real_ptr,
                                u1_imag_ptr,
                                deviceName_ptr,
                                u2_real_ptr,
                                u2_imag_ptr,
                                bUseMappedMemory_ptr,
                                u0step_ptr)
        if ret ==1:
            raise ValueError("Unable to run simulation (mostly likely name of GPU is incorrect)")

        # Return result
        return u2_real+1j*u2_imag
    else:
        assert(gpu_backend == 'MLX')
        
        # Convert input data to MLX arrays
        d_r1pr = clp.array(center)
        d_u1realpr=clp.array(np.real(u0))
        d_u1imagpr=clp.array(np.imag(u0))
        d_a1pr = clp.array(ds)
        
        # Get kernel function
        knl = prgcl['ForwardPropagationKernel']
    
    # Determine step size for looping
    NonBlockingstep = int(24000e6)
    step = int(NonBlockingstep/mr1)

    if step > mr2:
        step = mr2
    if step < 5:
        step = 5
    
    # Loop gpu kernel calls until all outputs have been calculated
    detection_point_start_index = 0        
    while detection_point_start_index < mr2:
        detection_point_end_index = min(detection_point_start_index + step, mr2)
        chunk_size = detection_point_end_index - detection_point_start_index
        
        print(f"Working on detection points {detection_point_start_index} to {detection_point_end_index} out of {u2.shape[0]}")
        
        # Grab section of data
        data_section = rf[detection_point_start_index:detection_point_end_index,:]
            
        if gpu_backend == 'CUDA':
            pass
            
        elif gpu_backend == 'OpenCL':
            
            # Determine kernel call dimensions
            if data_section.shape[0] % 64 ==0:
                ks = [data_section.shape[0]]
            else:
                ks = [int(data_section.shape[0]/64)*64+64]
            
            # Output section arrays
            u2_real = np.zeros_like(data_section[:,0])
            u2_imag = np.zeros_like(data_section[:,0])
        
            # Move to gpu
            d_r2pr = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_section) 
            d_u2realpr = clp.Buffer(ctx, mf.WRITE_ONLY,data_section.nbytes)
            d_u2imagpr = clp.Buffer(ctx, mf.WRITE_ONLY,data_section.nbytes)
            
            # Deploy kernel
            knl(queue, ks, [64],
                np.int32(rf.shape[0]),
                np.float32(np.real(cwvnb)),
                np.float32(np.imag(cwvnb)),
                np.float32(MaxDistance),
                np.int32(mr1),
                d_r2pr,
                d_r1pr,
                d_a1pr,
                d_u1realpr,
                d_u1imagpr,
                d_u2realpr,
                d_u2imagpr,
                np.int32(u0step))
            queue.finish()

            # Change back to numpy array 
            clp.enqueue_copy(queue, u2_real,d_u2realpr)
            clp.enqueue_copy(queue, u2_imag,d_u2imagpr)
            
            # Combine real & imag parts
            u2_section = u2_real+1j*u2_imag
            
            # Update final array
            u2[detection_point_start_index:detection_point_end_index] = u2_section
            
            # Clean up pocl arrays
            del d_u2realpr,d_u2imagpr
            gc.collect()
            
        elif gpu_backend == 'Metal':
            pass
        else:
            assert(gpu_backend == 'MLX')
            
            # Grab section of data
            d_r2pr = clp.array(data_section) 
            d_u2realpr = clp.zeros_like(d_r2pr[:,0])
            d_u2imagpr = clp.zeros_like(d_r2pr[:,0])
            
            # Deploy kernel
            d_u2realpr,d_u2imagpr = knl(inputs = [np.int32(detection_point_end_index),
                                                np.float32(np.real(cwvnb)),
                                                np.float32(np.imag(cwvnb)),
                                                np.float32(MaxDistance),
                                                np.int32(mr1),
                                                d_r2pr,
                                                d_r1pr,
                                                d_a1pr,
                                                d_u1realpr,
                                                d_u1imagpr,
                                                np.int32(u0step)],
                                        output_shapes = [(chunk_size,),(chunk_size,)],
                                        output_dtypes = [clp.float32,clp.float32],
                                        grid=(chunk_size,1,1),
                                        threadgroup=(256, 1, 1),
                                        verbose=False,
                                        stream=sel_device)
            clp.eval([d_u2realpr,d_u2imagpr])

            # Change back to numpy array 
            u2_real = np.array(d_u2realpr)
            u2_imag = np.array(d_u2imagpr)
            
            # Combine real & imag parts
            u2_section = u2_real+1j*u2_imag
            
            # Update final array
            u2[detection_point_start_index:detection_point_end_index] = u2_section
            
            # Clean up mlx arrays
            # del d_u2realpr,d_u2imagpr
            # gc.collect()
            
        # Update starting location
        detection_point_start_index += step
    
    return u2
    

def getBHTECoefficient( kappa,rho,c_t,h,t_int,dt=0.1):
    """ calculates the Bioheat Transfer Equation coefficient required (time step/density*conductivity*voxelsize"""
    # get the bioheat coefficients for a tissue type -- independent of surrounding tissue types
    # dt = t_int/nt
    # h - voxel resolution - default 1e-3

    bhc_coeff = kappa * dt / (rho * c_t * h**2)
    if bhc_coeff >= (1 / 6):
        best_nt = np.ceil(6 * kappa * t_int) / (rho * c_t *h**2)
        print("The conditions %f,%f,%f does not meet the C-F-L condition and may not be stable. Use nt of %f or greater." %\
            (dt,t_int,bhc_coeff,best_nt))
    return bhc_coeff

def getPerfusionCoefficient( w_b,rho,blood_rho=1050.0,dt=0.1):
    """Calculates the perfusion coefficient based on the simulation parameters and time step """
    # get the perfusion coeff for a speicfic tissue type and time period  -- independent of surrounding tissue types
    # wb is in ml/min/kg, needs to be converted to kg/m3/s (1min/60 * 1e-6 m3/ml) x blood density x tissue density
    # Camilleri et al. 2022. https://doi.org/10.3390/

    coeff = w_b/60 *1e-6 * blood_rho * rho  * dt

    return coeff

def getQCoeff(rho,SoS,alpha,c_t,Absorption,h,dt):
    coeff=dt/(2*rho**2*SoS*h*c_t)*Absorption*(1-np.exp(-2*h*alpha))
    return coeff

def factors_gpu(x):
    res=[]
    for i in range(2, x + 1):
        if x % i == 0:
            res.append(i)
    res=np.array(res)
    return res

def BHTE(Pressure,MaterialMap,MaterialList,dx,
                TotalDurationSteps,nStepsOn,LocationMonitoring,
                nFactorMonitoring=1,
                dt=0.1,blood_rho=1050,blood_ct=3617,
                stableTemp=37.0,DutyCycle=1.0,
                Backend='OpenCL',
                MonitoringPointsMap=None,
                initT0=None,
                initDose=None):

    # Verify valid initT0, initDose values if provided
    for k in [initT0,initDose]:
        if k is not None:
            assert(MaterialMap.shape[0]==k.shape[0] and \
            MaterialMap.shape[1]==k.shape[1] and \
            MaterialMap.shape[2]==k.shape[2])

            assert(k.dtype==np.float32)

    # Verify valid MonitoringPointsMap (i.e. grid points of interest) if provided
    if  MonitoringPointsMap is not None:
        assert(MaterialMap.shape[0]==MonitoringPointsMap.shape[0] and \
            MaterialMap.shape[1]==MonitoringPointsMap.shape[1] and \
            MaterialMap.shape[2]==MonitoringPointsMap.shape[2])

        assert(MonitoringPointsMap.dtype==np.uint32)

    # Calculate perfusion, bioheat, and Q coefficients for each material in grid
    perfArr=np.zeros(MaterialMap.max()+1,np.float32)
    bhArr=np.zeros(MaterialMap.max()+1,np.float32)
    if initT0 is None:
        initTemp = np.zeros(MaterialMap.shape, dtype=np.float32)
    else:
        initTemp = initT0

    Qarr=np.zeros(MaterialMap.shape, dtype=np.float32) 

    for n in range(MaterialMap.max()+1):
        bhArr[n]=getBHTECoefficient(MaterialList['Conductivity'][n],MaterialList['Density'][n],
                                    MaterialList['SpecificHeat'][n],dx,TotalDurationSteps,dt=dt)
        perfArr[n]=getPerfusionCoefficient(MaterialList['Perfusion'][n],
                                           MaterialList['Density'][n],
                                           blood_rho=blood_rho,dt=dt)
        if initT0 is None:
            initTemp[MaterialMap==n]=MaterialList['InitTemperature'][n]
        #print(n,(MaterialMap==n).sum(),Pressure[MaterialMap==n].mean())

        Qarr[MaterialMap==n]=Pressure[MaterialMap==n]**2*getQCoeff(MaterialList['Density'][n],
                                                                  MaterialList['SoS'][n],
                                                                  MaterialList['Attenuation'][n],
                                                                  MaterialList['SpecificHeat'][n],
                                                                  MaterialList['Absorption'][n],
                                                                  dx,dt)*DutyCycle

    # Dimensions of grid
    N1=np.int32(Pressure.shape[0])
    N2=np.int32(Pressure.shape[1])
    N3=np.int32(Pressure.shape[2])
    
    # If InitDose not supplied, create array of zeros
    if initDose is None:
        initDose = np.zeros(MaterialMap.shape, dtype=np.float32)

    # Create array of temperature points to monitor based on MonitoringPointsMap
    if MonitoringPointsMap is not None:
        MonitoringPoints = MonitoringPointsMap
        TotalPointsMonitoring=np.sum((MonitoringPointsMap>0).astype(int))
        TemperaturePoints=np.zeros((TotalPointsMonitoring,TotalDurationSteps),np.float32)
    else:
        MonitoringPoints = np.zeros(MaterialMap.shape, dtype=np.uint32)
        TemperaturePoints=np.zeros((10),np.float32) #just dummy array

     # Ensure valid number of time points for monitoring
    TotalStepsMonitoring=int(TotalDurationSteps/nFactorMonitoring)
    if TotalStepsMonitoring % nFactorMonitoring!=0:
        TotalStepsMonitoring+=1
    MonitorSlice=np.zeros((MaterialMap.shape[0],MaterialMap.shape[2],TotalStepsMonitoring),np.float32)

    # Inital temperature and thermal dose
    T1 = np.zeros(initTemp.shape,dtype=np.float32)
    Dose0 = initDose
    Dose1 = np.zeros(MaterialMap.shape,dtype=np.float32)

    nFraction=int(TotalDurationSteps/10)
    if nFraction ==0:
        nFraction=1

    if Backend=='OpenCL':

        mf = clp.mem_flags

        d_perfArr=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=perfArr)
        d_bhArr=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bhArr)
        d_Qarr=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Qarr)
        d_MaterialMap=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MaterialMap)
        d_MonitoringPoints=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MonitoringPoints)


        d_T0 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=initTemp)
        d_T1 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=T1)

        
        d_Dose0 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose0)
        d_Dose1 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose1)

        d_MonitorSlice = clp.Buffer(ctx, mf.WRITE_ONLY, MonitorSlice.nbytes)
        d_TemperaturePoints = clp.Buffer(ctx, mf.WRITE_ONLY, TemperaturePoints.nbytes)

        knl = prgcl.BHTEFDTDKernel

        l1=factors_gpu(MaterialMap.shape[0])
        if len(l1)>0:
            local=[l1[0],l1[0],1]
        else:
            local=None
            
        gl=[]
        for n in range(3):
            m=MaterialMap.shape[n]
            while(not np.any(factors_gpu(m)==4)):
                m+=1
            gl.append(m)
        
        for n in range(TotalDurationSteps):
            if n<nStepsOn:
                dUS=1
            else:
                dUS=0
            if (n%2==0):
                knl(queue,gl , None,
                    d_T1,
                    d_Dose1,
                    d_T0,
                    d_Dose0,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    np.float32(stableTemp),
                    np.int32(dUS),
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    np.int32(TotalStepsMonitoring),
                    np.int32(nFactorMonitoring),
                    np.int32(n),
                    np.int32(LocationMonitoring),
                    np.uint32(0),
                    np.uint32(TotalDurationSteps))

            else:
                knl(queue, gl , None,
                    d_T0,
                    d_Dose0,
                    d_T1,
                    d_Dose1,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    np.float32(stableTemp),
                    np.int32(dUS),
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    np.int32(TotalStepsMonitoring),
                    np.int32(nFactorMonitoring),
                    np.int32(n),
                    np.int32(LocationMonitoring),
                    np.uint32(0),
                    np.uint32(TotalDurationSteps))
            queue.finish()
            if n % nFraction ==0:
                print(n,TotalDurationSteps)

        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0


        print('Done BHTE')                               
        clp.enqueue_copy(queue, T1,ResTemp)
        clp.enqueue_copy(queue, Dose1,ResDose)
        clp.enqueue_copy(queue, MonitorSlice,d_MonitorSlice)
        clp.enqueue_copy(queue, TemperaturePoints,d_TemperaturePoints)
        queue.finish()

    elif Backend=='CUDA':

        dimBlockBHTE = (4,4,4)

        dimGridBHTE  = (int(N1/dimBlockBHTE[0]+1),
                        int(N2/dimBlockBHTE[1]+1),
                        int(N3/dimBlockBHTE[2]+1))

        d_perfArr=clp.asarray(perfArr)
        d_bhArr=clp.asarray(bhArr)
        d_Qarr=clp.asarray(Qarr)
        d_MaterialMap=clp.asarray(MaterialMap)
        d_T0 = clp.asarray(initTemp)
        d_T1 = clp.asarray(T1)
        d_Dose0 = clp.asarray(Dose0)
        d_Dose1 = clp.asarray(Dose1)
        d_MonitoringPoints=clp.asarray(MonitoringPoints)
     
        d_MonitorSlice = clp.zeros(MonitorSlice.shape,clp.float32)
        d_TemperaturePoints=clp.zeros(TemperaturePoints.shape,clp.float32)

        BHTEKernel = prgcuda.get_function("BHTEFDTDKernel")

        for n in range(TotalDurationSteps):
            if n<nStepsOn:
                dUS=1
            else:
                dUS=0
            if (n%2==0):
                BHTEKernel(dimGridBHTE,
                    dimBlockBHTE,
                    (d_T1,
                    d_Dose1,
                    d_T0,
                    d_Dose0,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    np.float32(stableTemp),
                    np.int32(dUS),
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    TotalStepsMonitoring,
                    nFactorMonitoring,
                    n,
                    LocationMonitoring,
                    0,
                    TotalDurationSteps))

            else:
                BHTEKernel(dimGridBHTE,
                    dimBlockBHTE,
                    (d_T0,
                    d_Dose0,
                    d_T1,
                    d_Dose1,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    np.float32(stableTemp),
                    dUS,
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    TotalStepsMonitoring,
                    nFactorMonitoring,
                    n,
                    LocationMonitoring,
                    0,
                    TotalDurationSteps))
            clp.cuda.Stream.null.synchronize()
            if n % nFraction ==0:
                print(n,TotalDurationSteps)
        
        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0
        
        T1=ResTemp.get()
        Dose1=ResDose.get()
        MonitorSlice=d_MonitorSlice.get() 
        TemperaturePoints=d_TemperaturePoints.get() 

    elif Backend=='MLX':

        # Create mlx arrays
        d_perfArr = clp.array(perfArr)
        d_bhArr = clp.array(bhArr)
        d_Qarr = clp.array(Qarr)
        d_MaterialMap = clp.array(MaterialMap)
        d_T0 = clp.array(initTemp)
        d_Dose0 = clp.array(Dose0)
        d_MonitoringPoints = clp.array(MonitoringPoints)

        # Build program from source code
        knl = prgcl['BHTEFDTDKernel']

        # Float variables to be passed to kernel
        floatparams = np.array([stableTemp,dt],dtype=np.float32)
        d_floatparams= clp.array(floatparams)
        
        # Determine number of loops, we evaluate mlx arrays
        # if n_eval gets too large, the lazy computation of mlx cause the compute graph 
        # to become too large and signficantly impacts performance
        n_eval = int(250e6/d_T0.size) # 250e6 array size was found to be a good point for ensuring fast performance
        n_eval = max(n_eval,1)
        
        # Calculate BHTE for each time point
        for n in range(TotalDurationSteps):
            if n<nStepsOn:
                dUS=1 # Ultrasound on
            else:
                dUS=0 # Ultrasound off
                
            # Int variables to be passed to kernel
            intparams = np.array([dUS,N1,N2,N3,TotalStepsMonitoring,nFactorMonitoring,n,LocationMonitoring,0,TotalDurationSteps],dtype=np.uint32)
            d_intparams= clp.array(intparams)

            # We only initialize output arrays for first two loops since
            # initialization significantly impacts performance
            if n == 0 or n == 1:
                init_value_mlx = 0 # output arrays get initialized to 0
            else:
                init_value_mlx = None # we skip output arrays initalization
                
            # At each time point, the previous output is used as the current input (e.g. d_T0 and d_T1 alternate, same with Dose0 and Dose1)
            if (n%2==0):
                d_T1,d_Dose1,d_MonitorSlice,d_TemperaturePoints = knl(inputs= [d_T0,
                                                                               d_Dose0,
                                                                               d_bhArr,
                                                                               d_perfArr,
                                                                               d_MaterialMap,
                                                                               d_Qarr,
                                                                               d_MonitoringPoints,
                                                                               d_floatparams,
                                                                               d_intparams],
                                                                      output_shapes = [T1.shape,Dose1.shape,MonitorSlice.shape,TemperaturePoints.shape],
                                                                      output_dtypes = [clp.float32,clp.float32,clp.float32,clp.float32],
                                                                      grid=(N1*N2*N3,1,1),
                                                                      threadgroup=(256, 1, 1),
                                                                      verbose=False,
                                                                      stream=sel_device,
                                                                      init_value=init_value_mlx)

            else:
                d_T0,d_Dose0,d_MonitorSlice,d_TemperaturePoints = knl(inputs = [d_T1,
                                                                                d_Dose1,
                                                                                d_bhArr,
                                                                                d_perfArr,
                                                                                d_MaterialMap,
                                                                                d_Qarr,
                                                                                d_MonitoringPoints,
                                                                                d_floatparams,
                                                                                d_intparams],
                                                                      output_shapes = [initTemp.shape,Dose0.shape,MonitorSlice.shape,TemperaturePoints.shape],
                                                                      output_dtypes = [clp.float32,clp.float32,clp.float32,clp.float32],
                                                                      grid=(N1*N2*N3,1,1),
                                                                      threadgroup=(256, 1, 1),
                                                                      verbose=False,
                                                                      stream=sel_device,
                                                                      init_value=init_value_mlx)
            
            # Evaluate values to reset compute graph   
            if n%n_eval==0:
                clp.eval(d_T1)
            
            # Track progress of BHTE calculation
            if n % nFraction ==0:
                print(n,TotalDurationSteps)

        # Grab final output depending on time point number
        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0
        print('Done BHTE')

        # Transfer back numpy array                           
        T1 = np.array(ResTemp,dtype=np.float32)
        Dose1 = np.array(ResDose,dtype=np.float32)
        MonitorSlice = np.array(d_MonitorSlice,dtype=np.float32)
        TemperaturePoints = np.array(d_TemperaturePoints,dtype=np.float32)
        
    else:
        assert(Backend=='Metal')

        d_perfArr=ctx.buffer(perfArr)
        d_bhArr=ctx.buffer(bhArr)
        d_Qarr=ctx.buffer(Qarr)
        d_MaterialMap=ctx.buffer(MaterialMap)
        d_T0 = ctx.buffer(initTemp)
        d_T1 = ctx.buffer(T1)
        d_Dose0 = ctx.buffer(Dose0)
        d_Dose1 = ctx.buffer(Dose1)
        d_MonitorSlice = ctx.buffer(MonitorSlice.nbytes)
        d_MonitoringPoints=ctx.buffer(MonitoringPoints)
        d_TemperaturePoints=ctx.buffer(TemperaturePoints.nbytes)

        knl = prgcl.function('BHTEFDTDKernel')

        floatparams = np.array([stableTemp,dt],dtype=np.float32)
        d_floatparams= ctx.buffer(floatparams)
        for n in range(TotalDurationSteps):
            if n<nStepsOn:
                dUS=1
            else:
                dUS=0
            intparams = np.array([dUS,N1,N2,N3,TotalStepsMonitoring,nFactorMonitoring,n,LocationMonitoring,0,TotalDurationSteps],dtype=np.uint32)
            d_intparams= ctx.buffer(intparams)
            ctx.init_command_buffer()
            if (n%2==0):
                handle=knl(N1*N2*N3,
                    d_T1,
                    d_Dose1,
                    d_T0,
                    d_Dose0,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    d_floatparams,
                    d_intparams)

            else:
                handle=knl(N1*N2*N3,
                    d_T0,
                    d_Dose0,
                    d_T1,
                    d_Dose1,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    d_floatparams,
                    d_intparams)
            ctx.commit_command_buffer()
            ctx.wait_command_buffer()
            del handle
            if n % nFraction ==0:
                print(n,TotalDurationSteps)

        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0
    
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((ResTemp,ResDose,d_MonitorSlice,d_TemperaturePoints))
        print('Done BHTE')                               
        T1=np.frombuffer(ResTemp,dtype=np.float32).reshape((N1,N2,N3))
        Dose1=np.frombuffer(ResDose,dtype=np.float32).reshape((N1,N2,N3))
        MonitorSlice=np.frombuffer(d_MonitorSlice,dtype=np.float32).reshape((MaterialMap.shape[0],MaterialMap.shape[2],TotalStepsMonitoring))
        TemperaturePoints=np.frombuffer(d_TemperaturePoints,dtype=np.float32).reshape(TemperaturePoints.shape)

    if MonitoringPointsMap is not None:
        return T1,Dose1,MonitorSlice,Qarr,TemperaturePoints
    else:
        return T1,Dose1,MonitorSlice,Qarr


def BHTEMultiplePressureFields(PressureFields,
                MaterialMap,
                MaterialList,
                dx,
                TotalDurationSteps,
                nStepsOnOffList,
                LocationMonitoring,
                nFactorMonitoring=1,
                dt=0.1,
                blood_rho=1050,
                blood_ct=3617,
                stableTemp=37.0,
                Backend='OpenCL',
                MonitoringPointsMap=None,
                initT0=None,
                initDose=None):

    assert(PressureFields.shape[1]==MaterialMap.shape[0] and \
           PressureFields.shape[2]==MaterialMap.shape[1] and \
           PressureFields.shape[3]==MaterialMap.shape[2])

    for k in [initT0,initDose]:
        if k is not None:
            assert(MaterialMap.shape[0]==k.shape[0] and \
            MaterialMap.shape[1]==k.shape[1] and \
            MaterialMap.shape[2]==k.shape[2])

            assert(k.dtype==np.float32)
    if  MonitoringPointsMap is not None:
        assert(MaterialMap.shape[0]==MonitoringPointsMap.shape[0] and \
            MaterialMap.shape[1]==MonitoringPointsMap.shape[1] and \
            MaterialMap.shape[2]==MonitoringPointsMap.shape[2])

        assert(MonitoringPointsMap.dtype==np.uint32)

    assert(nStepsOnOffList.shape[0]==PressureFields.shape[0])

    perfArr=np.zeros(MaterialMap.max()+1,np.float32)
    bhArr=np.zeros(MaterialMap.max()+1,np.float32)

    if initT0 is None:
        initTemp = np.zeros(MaterialMap.shape, dtype=np.float32) 
    else:
        initTemp = initT0

    
    QArrList=np.zeros(PressureFields.shape, dtype=np.float32) 
    
    for n in range(MaterialMap.max()+1):
        bhArr[n]=getBHTECoefficient(MaterialList['Conductivity'][n],MaterialList['Density'][n],
                                    MaterialList['SpecificHeat'][n],dx,TotalDurationSteps,dt=dt)
        perfArr[n]=getPerfusionCoefficient(MaterialList['Perfusion'][n],
                                           MaterialList['Density'][n],
                                           blood_rho=blood_rho,
                                           dt=dt)
        if initT0 is None:
            initTemp[MaterialMap==n]=MaterialList['InitTemperature'][n]
        for m in range(PressureFields.shape[0]):
            QArrList[m,:,:,:][MaterialMap==n]=PressureFields[m,:,:,:][MaterialMap==n]**2*getQCoeff(MaterialList['Density'][n],
                                                                    MaterialList['SoS'][n],
                                                                    MaterialList['Attenuation'][n],
                                                                    MaterialList['SpecificHeat'][n],
                                                                    MaterialList['Absorption'][n],dx,dt)
    TimingFields=np.zeros((nStepsOnOffList.shape[0],3),np.int32)
    #we prepare the index location when each field is on and off
    TimingFields[0,1]=nStepsOnOffList[0,0]
    TimingFields[0,2]=nStepsOnOffList[0,1]+nStepsOnOffList[0,0]
    for m in range(1,nStepsOnOffList.shape[0]):
        TimingFields[m,0]=TimingFields[m-1,2]
        TimingFields[m,1]=TimingFields[m,0]+nStepsOnOffList[m,0]
        TimingFields[m,2]=TimingFields[m,1]+nStepsOnOffList[m,1]

    print('TimingFields',TimingFields)
    NstepsPerCycle=TimingFields[-1,2]
    

    N1=np.int32(MaterialMap.shape[0])
    N2=np.int32(MaterialMap.shape[1])
    N3=np.int32(MaterialMap.shape[2])
    coreTemp = np.array([stableTemp],np.float32)

    if initDose is None:
        initDose = np.zeros(MaterialMap.shape, dtype=np.float32)
    
    if MonitoringPointsMap is not None:
        MonitoringPoints = MonitoringPointsMap
        TotalPointsMonitoring=np.sum((MonitoringPointsMap>0).astype(int))
        TemperaturePoints=np.zeros((TotalPointsMonitoring,TotalDurationSteps),np.float32)
    else:
        MonitoringPoints = np.zeros(MaterialMap.shape, dtype=np.uint32)
        TemperaturePoints=np.zeros((10),np.float32) #just dummy array

    T1 = np.zeros(initTemp.shape,dtype=np.float32)
    Dose0 = initDose
    Dose1 = np.zeros(MaterialMap.shape,dtype=np.float32)

    TotalStepsMonitoring=int(TotalDurationSteps/nFactorMonitoring)
    if TotalStepsMonitoring % nFactorMonitoring!=0:
        TotalStepsMonitoring+=1
        
    MonitorSlice=np.zeros((MaterialMap.shape[0],MaterialMap.shape[2],TotalStepsMonitoring),np.float32)
    nFraction=int(TotalDurationSteps/10)
    
    if nFraction ==0:
        nFraction=1

    if Backend == 'OpenCL':

        mf = clp.mem_flags

        d_perfArr=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=perfArr)
        d_bhArr=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bhArr)
        d_QArrList=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=QArrList)
        d_MaterialMap=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MaterialMap)
        d_MonitoringPoints=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MonitoringPoints)

        d_T0 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=initTemp)
        d_T1 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=T1)
        d_Dose0 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose0)
        d_Dose1 = clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose1)

        d_MonitorSlice = clp.Buffer(ctx, mf.WRITE_ONLY, MonitorSlice.nbytes)
        d_TemperaturePoints = clp.Buffer(ctx, mf.WRITE_ONLY, TemperaturePoints.nbytes)

        knl = prgcl.BHTEFDTDKernel

        l1=factors_gpu(MaterialMap.shape[0])
        if len(l1)>0:
            local=[l1[0],l1[0],1]
        else:
            local=None


        gl=[]
        for n in range(3):
            m=MaterialMap.shape[n]
            while(not np.any(factors_gpu(m)==4)):
                m+=1
            gl.append(m)
        
        for n in range(TotalDurationSteps):
            mStep=n % NstepsPerCycle
            QSegment=np.where((TimingFields[:,0]<=mStep) & (TimingFields[:,2]>mStep))[0][0]
            if mStep<TimingFields[QSegment,1]:
                dUS=1
            else:
                assert(mStep>=TimingFields[QSegment,1])
                dUS=0
            StartIndexQ=np.prod(np.array(QArrList.shape[1:]))*QSegment
            if (n%2==0):
                knl(queue,gl , None,
                    d_T1,
                    d_Dose1,
                    d_T0,
                    d_Dose0,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_QArrList,
                    d_MonitoringPoints,
                    np.float32(coreTemp),
                    np.int32(dUS),
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    np.int32(TotalStepsMonitoring),
                    np.int32(nFactorMonitoring),
                    np.int32(n),
                    np.int32(LocationMonitoring),
                    np.uint32(StartIndexQ),
                    np.uint32(TotalDurationSteps))

            else:
                knl(queue, gl , None,
                    d_T0,
                    d_Dose0,
                    d_T1,
                    d_Dose1,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_QArrList,
                    d_MonitoringPoints,
                    np.float32(coreTemp),
                    np.int32(dUS),
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    np.int32(TotalStepsMonitoring),
                    np.int32(nFactorMonitoring),
                    np.int32(n),
                    np.int32(LocationMonitoring),
                    np.uint32(StartIndexQ),
                    np.uint32(TotalDurationSteps))
            queue.finish()
            if n % nFraction ==0:
                print(n,TotalDurationSteps)

        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0


        print('Done BHTE')                               
        clp.enqueue_copy(queue, T1,ResTemp)
        clp.enqueue_copy(queue, Dose1,ResDose)
        clp.enqueue_copy(queue, MonitorSlice,d_MonitorSlice)
        clp.enqueue_copy(queue, TemperaturePoints,d_TemperaturePoints)
        queue.finish()
    elif Backend=='CUDA':
        dimBlockBHTE = (4,4,4)

        dimGridBHTE  = (int(N1/dimBlockBHTE[0]+1),
                        int(N2/dimBlockBHTE[1]+1),
                        int(N3/dimBlockBHTE[2]+1))

        d_perfArr=clp.asarray(perfArr)
        d_bhArr=clp.asarray(bhArr)
        d_QArrList=clp.asarray(QArrList)
        d_MaterialMap=clp.asarray(MaterialMap)
        d_T0 = clp.asarray(initTemp)
        d_T1 = clp.asarray(T1)
        d_Dose0 = clp.asarray(Dose0)
        d_Dose1 = clp.asarray(Dose1)
        d_MonitoringPoints=clp.asarray(MonitoringPoints)
     
        d_MonitorSlice = clp.zeros(MonitorSlice.shape,clp.float32)
        d_TemperaturePoints=clp.zeros(TemperaturePoints.shape,clp.float32)

        BHTEKernel = prgcuda.get_function("BHTEFDTDKernel")

        for n in range(TotalDurationSteps):
            mStep=n % NstepsPerCycle
            QSegment=np.where((TimingFields[:,0]<=mStep) & (TimingFields[:,2]>mStep))[0][0]
            if mStep<TimingFields[QSegment,1]:
                dUS=1
            else:
                assert(mStep>=TimingFields[QSegment,1])
                dUS=0
            StartIndexQ=np.prod(np.array(QArrList.shape[1:]))*QSegment

            if (n%2==0):
                BHTEKernel(dimGridBHTE,
                        dimBlockBHTE,
                    (d_T1,
                    d_Dose1,
                    d_T0,
                    d_Dose0,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_QArrList,
                    d_MonitoringPoints,
                    np.float32(stableTemp),
                    dUS,
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    TotalStepsMonitoring,
                    nFactorMonitoring,
                    n,
                    LocationMonitoring,
                    StartIndexQ,
                    TotalDurationSteps))
                    
            else:
                BHTEKernel(dimGridBHTE,
                        dimBlockBHTE,
                    (d_T0,
                    d_Dose0,
                    d_T1,
                    d_Dose1,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_QArrList,
                    d_MonitoringPoints,
                    np.float32(stableTemp),
                    dUS,
                    N1,
                    N2,
                    N3,
                    np.float32(dt),
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    TotalStepsMonitoring,
                    nFactorMonitoring,
                    n,
                    LocationMonitoring,
                    StartIndexQ,
                    TotalDurationSteps))
            clp.cuda.Stream.null.synchronize()
            if n % nFraction ==0:
                print(n,TotalDurationSteps)
        
        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0
        T1=ResTemp.get()
        Dose1=ResDose.get()
        MonitorSlice=d_MonitorSlice.get() 
        TemperaturePoints=d_TemperaturePoints.get()  
    elif Backend == 'MLX':
        d_perfArr=clp.array(perfArr)
        d_bhArr=clp.array(bhArr)
        d_QArrList=clp.array(QArrList)
        d_MaterialMap=clp.array(MaterialMap)
        d_T0 = clp.array(initTemp)
        d_Dose0 = clp.array(Dose0)
        d_MonitoringPoints=clp.array(MonitoringPoints)

        # Build program from source code
        knl = prgcl['BHTEFDTDKernel']

        # Float variables to be passed to kernel
        floatparams = np.array([stableTemp,dt],dtype=np.float32)
        d_floatparams= clp.array(floatparams)
        
        # Determine number of loops, we evaluate mlx arrays
        # if n_eval gets too large, the lazy computation of mlx cause the compute graph 
        # to become too large and signficantly impacts performance
        n_eval = int(250e6/d_T0.size) # 250e6 array size was found to be a good point for ensuring fast performance
        n_eval = max(n_eval,1)
        
        # Calculate BHTE for each time point
        for n in range(TotalDurationSteps):
            mStep=n % NstepsPerCycle
            QSegment=np.where((TimingFields[:,0]<=mStep) & (TimingFields[:,2]>mStep))[0][0]
            if mStep<TimingFields[QSegment,1]:
                dUS=1 # Ultrasound on
            else:
                assert(mStep>=TimingFields[QSegment,1])
                dUS=0 # Ultrasound off
            StartIndexQ=np.prod(np.array(QArrList.shape[1:]))*QSegment

            # Int variables to be passed to kernel
            intparams = np.array([dUS,N1,N2,N3,TotalStepsMonitoring,nFactorMonitoring,n,LocationMonitoring,StartIndexQ,TotalDurationSteps],dtype=np.uint32)
            d_intparams= clp.array(intparams)
            
            # We only initialize output arrays for first two loops since
            # initialization significantly impacts performance
            if n == 0 or n == 1:
                init_value_mlx = 0 # output arrays get initialized to 0
            else:
                init_value_mlx = None # we skip output arrays initalization
            
            # At each time point, the previous output is used as the current input (e.g. d_T0 and d_T1 alternate, same with Dose0 and Dose1)
            if (n%2==0):
                d_T1,d_Dose1,d_MonitorSlice,d_TemperaturePoints = knl(inputs= [d_T0,
                                                                               d_Dose0,
                                                                               d_bhArr,
                                                                               d_perfArr,
                                                                               d_MaterialMap,
                                                                               d_QArrList,
                                                                               d_MonitoringPoints,
                                                                               d_floatparams,
                                                                               d_intparams],
                                                                      output_shapes = [T1.shape,Dose1.shape,MonitorSlice.shape,TemperaturePoints.shape],
                                                                      output_dtypes = [clp.float32,clp.float32,clp.float32,clp.float32],
                                                                      grid=(N1*N2*N3,1,1),
                                                                      threadgroup=(256, 1, 1),
                                                                      verbose=False,
                                                                      stream=sel_device,
                                                                      init_value=init_value_mlx)

            else:
                d_T0,d_Dose0,d_MonitorSlice,d_TemperaturePoints = knl(inputs = [d_T1,
                                                                                d_Dose1,
                                                                                d_bhArr,
                                                                                d_perfArr,
                                                                                d_MaterialMap,
                                                                                d_QArrList,
                                                                                d_MonitoringPoints,
                                                                                d_floatparams,
                                                                                d_intparams],
                                                                      output_shapes = [initTemp.shape,Dose0.shape,MonitorSlice.shape,TemperaturePoints.shape],
                                                                      output_dtypes = [clp.float32,clp.float32,clp.float32,clp.float32],
                                                                      grid=(N1*N2*N3,1,1),
                                                                      threadgroup=(256, 1, 1),
                                                                      verbose=False,
                                                                      stream=sel_device,
                                                                      init_value=init_value_mlx)
            
            # Evaluate values to reset compute graph   
            if n%n_eval==0:
                clp.eval(d_T1)
            
            # Track progress of BHTE calculation
            if n % nFraction ==0:
                print(n,TotalDurationSteps)

        # Grab final output depending on time point number
        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0
        print('Done BHTE')
        
        # Transfer back numpy array                                  
        T1=np.array(ResTemp,dtype=np.float32)
        Dose1=np.array(ResDose,dtype=np.float32)
        MonitorSlice=np.array(d_MonitorSlice,dtype=np.float32)
        TemperaturePoints=np.array(d_TemperaturePoints,dtype=np.float32)
    else:
        assert(Backend=='Metal')

        d_perfArr=ctx.buffer(perfArr)
        d_bhArr=ctx.buffer(bhArr)
        d_QArrList=ctx.buffer(QArrList)
        d_MaterialMap=ctx.buffer(MaterialMap)
        d_T0 = ctx.buffer(initTemp)
        d_T1 = ctx.buffer(T1)
        d_Dose0 = ctx.buffer(Dose0)
        d_Dose1 = ctx.buffer(Dose1)
        d_MonitorSlice = ctx.buffer(MonitorSlice.nbytes)
        d_MonitoringPoints=ctx.buffer(MonitoringPoints)
        d_TemperaturePoints=ctx.buffer(TemperaturePoints.nbytes)

        knl = prgcl.function('BHTEFDTDKernel')

        floatparams = np.array([stableTemp,dt],dtype=np.float32)
        d_floatparams= ctx.buffer(floatparams)
        for n in range(TotalDurationSteps):
            mStep=n % NstepsPerCycle
            QSegment=np.where((TimingFields[:,0]<=mStep) & (TimingFields[:,2]>mStep))[0][0]
            if mStep<TimingFields[QSegment,1]:
                dUS=1
            else:
                assert(mStep>=TimingFields[QSegment,1])
                dUS=0
            StartIndexQ=np.prod(np.array(QArrList.shape[1:]))*QSegment

            intparams = np.array([dUS,N1,N2,N3,TotalStepsMonitoring,nFactorMonitoring,n,LocationMonitoring,StartIndexQ,TotalDurationSteps],dtype=np.uint32)
            d_intparams= ctx.buffer(intparams)
            ctx.init_command_buffer()
            if (n%2==0):
                handle=knl(N1*N2*N3,
                    d_T1,
                    d_Dose1,
                    d_T0,
                    d_Dose0,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_QArrList,
                    d_MonitoringPoints,
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    d_floatparams,
                    d_intparams)

            else:
                handle=knl(N1*N2*N3,
                    d_T0,
                    d_Dose0,
                    d_T1,
                    d_Dose1,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_QArrList,
                    d_MonitoringPoints,
                    d_MonitorSlice,
                    d_TemperaturePoints,
                    d_floatparams,
                    d_intparams)
            ctx.commit_command_buffer()
            ctx.wait_command_buffer()
            del handle
            if n % nFraction ==0:
                print(n,TotalDurationSteps)

        if (n%2==0):
            ResTemp=d_T1
            ResDose=d_Dose1
        else:
            ResTemp=d_T0
            ResDose=d_Dose0

        if 'arm64' not in platform.platform():
            ctx.sync_buffers((ResTemp,ResDose,d_MonitorSlice,d_TemperaturePoints))
        print('Done BHTE')                               
        T1=np.frombuffer(ResTemp,dtype=np.float32).reshape((N1,N2,N3))
        Dose1=np.frombuffer(ResDose,dtype=np.float32).reshape((N1,N2,N3))
        MonitorSlice=np.frombuffer(d_MonitorSlice,dtype=np.float32).reshape((MaterialMap.shape[0],MaterialMap.shape[2],TotalStepsMonitoring))
        TemperaturePoints=np.frombuffer(d_TemperaturePoints,dtype=np.float32).reshape(TemperaturePoints.shape)
    if MonitoringPointsMap is not None:
        return T1,Dose1,MonitorSlice,QArrList,TemperaturePoints
    else:
        return T1,Dose1,MonitorSlice,QArrList
