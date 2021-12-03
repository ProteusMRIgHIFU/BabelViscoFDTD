'''
Samuel Pichardo, Ph.D.
University of Calgary

Very simple GPU-accelerated Rayleigh integral with support to OpenCL and CUDA

CUDA is automatically selected if running Windows or Linux, while OpenCL is selected in MacOs
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys
from sysconfig import get_paths
import ctypes
import sys 
import platform

npyInc=np.get_include()
info = get_paths()

KernelCoreSourceBHTE="""
    #define Tref 43.0
    int DzDy=outerDimz*outerDimy;
    int coord = gtidx*DzDy + gtidy*outerDimz + gtidz;
    
    float R1,R2,dtp;
    if(gtidx > 0 && gtidx < outerDimx-1 && gtidy > 0 && gtidy < outerDimy-1 && gtidz > 0 && gtidz < outerDimz-1)
    {

            const int label = d_labels[coord];

            d_output[coord] = d_input[coord] + d_bhArr[label] * ( 
                     d_input[coord + 1] + d_input[coord - 1] + d_input[coord + outerDimz] + d_input[coord - outerDimz] +
                      d_input[coord + DzDy] + d_input[coord - DzDy] - 6.0 * d_input[coord]) +
                    + d_perfArr[label] * (CoreTemp - d_input[coord]) ;
            if (sonication)
            {
                d_output[coord]+=d_Qarr[coord+StartIndexQ];
            }
            
            R2 = (d_output[coord] >= Tref)?0.5:0.25; 
            R1 = (d_input[coord] >= Tref)?0.5:0.25;

            if(fabs(d_output[coord]-d_input[coord])<0.0001)
            {
                d_output2[coord] = d_input2[coord] + dt * pow(float(R1),float((Tref-d_input[coord])));
            }
            else
            {
                if(R1 == R2)
                {
                    d_output2[coord] = d_input2[coord] + (pow(float(R2),float((Tref-d_output[coord]))) - pow(float(R1),float((Tref-d_input[coord])))) / 
                                   ( -(d_output[coord]-d_input[coord])/ dt * log(R1));
                }
                else
                {
                    dtp = dt * (Tref - d_input[coord])/(d_output[coord] - d_input[coord]);

                    d_output2[coord] = d_input2[coord] + (1 - pow(float(R1),float((Tref-d_input[coord]))))     / (- (Tref - d_input[coord])/ dtp * log(R1)) + 
                                   (pow(float(R2),float((Tref-d_output[coord]))) - 1) / (-(d_output[coord] - Tref)/(dt - dtp) * log(R2));
                }
            }

            if (gtidy==SelJ && (n_Step % nFactorMonitoring ==0))
            {
                 d_MonitorSlice[gtidx*outerDimz*TotalStepsMonitoring+gtidz*TotalStepsMonitoring+ n_Step/nFactorMonitoring] =d_output[coord];
            }
        }
        else if(gtidx < outerDimx && gtidy < outerDimy && gtidz < outerDimz){
            d_output[coord] = d_input[coord];
            d_output2[coord] = d_input2[coord];

        }

}
"""

if sys.platform == "darwin":
    import pyopencl as cl
    
    RayleighOpenCLSource="""

    #define pi 3.141592653589793

    typedef float FloatingType;

    __kernel  void ForwardPropagationKernel(  __global const int *mr2,
                                              __global const FloatingType *c_wvnb_real,
                                              __global const FloatingType *c_wvnb_imag,
                                              __global const int *mr1,
                                             __global const FloatingType *r2pr, 
                                             __global const FloatingType *r1pr, 
                                             __global const FloatingType *a1pr, 
                                             __global const FloatingType *u1_real, 
                                             __global const FloatingType *u1_imag,
                                             __global  FloatingType  *py_data_u2_real,
                                             __global  FloatingType  *py_data_u2_imag
                                             )
        {
        int si2 = get_global_id(0);		// Grid is a "flatten" 1D, thread blocks are 1D

        FloatingType dx,dy,dz,R,r2x,r2y,r2z;
        FloatingType temp_r,tr ;
        FloatingType temp_i,ti,pCos,pSin ;


        if ( si2 < *mr2)  
        {
            temp_r = 0;
            temp_i = 0;
            r2x=r2pr[si2*3];
            r2y=r2pr[si2*3+1];
            r2z=r2pr[si2*3+2];

            for (int si1=0; si1<*mr1; si1++)
            {
                // In matlab we have a Fortran convention, in Python-numpy, we have the C-convention for matrixes (hoorray!!!)
                dx=r1pr[si1*3]-r2x;
                dy=r1pr[si1*3+1]-r2y;
                dz=r1pr[si1*3+2]-r2z;


                R=sqrt(dx*dx+dy*dy+dz*dz);
                ti=(exp(R*c_wvnb_imag[0])*a1pr[si1]/R);

                tr=ti;
                pSin=sincos(R*c_wvnb_real[0],&pCos);
                tr*=(u1_real[si1]*pCos+u1_imag[si1]*pSin);
                            ti*=(u1_imag[si1]*pCos-u1_real[si1]*pSin);

                temp_r +=tr;
                temp_i +=ti;	
            }
            
            R=temp_r;

            temp_r = -temp_r*c_wvnb_imag[0]-temp_i*c_wvnb_real[0];
            temp_i = R*c_wvnb_real[0]-temp_i*c_wvnb_imag[0];

            py_data_u2_real[si2]=temp_r/(2*pi);
            py_data_u2_imag[si2]=temp_i/(2*pi);
        }
        }
      """

    OpenCLHeaderBHTE="""
        __kernel  void BHTEFDTDKernel( __global float 		*d_output, 
                                        __global float 		*d_output2,
                                        __global const float 			*d_input, 
                                        __global const float 			*d_input2,
                                        __global const float 			*d_bhArr,
                                        __global const float 			*d_perfArr, 
                                        __global const unsigned int		*d_labels,
                                        __global const float 			*d_Qarr,
                                            const float 			CoreTemp,
                                            const  int				sonication,
                                            const  int				outerDimx, 
                                            const  int              outerDimy, 
                                            const  int              outerDimz,
                                            const float 			dt,
                                            __global  float 	*d_MonitorSlice,
                                            const  int TotalStepsMonitoring,
                                            const  int nFactorMonitoring,
                                            const  int n_Step,
                                            const int SelJ,
                                            const unsigned int StartIndexQ)	
        {
            const int gtidx =  get_global_id(0);
            const int gtidy =  get_global_id(1);
            const int gtidz =  get_global_id(2);
        """

    OpenCLKernelBHTE =OpenCLHeaderBHTE + KernelCoreSourceBHTE

    Platforms=None
    queue = None
    prg = None
    ctx = None

    # Loads METAL interface
    os.environ['__RayleighMetal'] =os.path.dirname(os.path.abspath(__file__))
    
    swift_fun = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__))+"/libRayleighMetal.dylib")

    swift_fun.ForwardSimpleMetal.argtypes = [
        ctypes.POINTER(ctypes.c_int),
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
        ctypes.POINTER(ctypes.c_int)]

    swift_fun.PrintMetalDevices()
    print("loaded Metal",str(swift_fun))

else:
    prg = None
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    RayleighCUDASource="""
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <ndarrayobject.h>

    #define pi 3.141592653589793

    typedef float FloatingType;
    typedef npy_cfloat FloatingTypeComplex;

    #define MAX_ELEMS_IN_CONSTANT  2730 // the total constant memory can't be greater than 64k bytes

    __constant__ float PartSource [MAX_ELEMS_IN_CONSTANT*3];
    __constant__ float Part_a1pr [MAX_ELEMS_IN_CONSTANT];
    __constant__ npy_cfloat Part_u1complex[MAX_ELEMS_IN_CONSTANT];


    __global__ void ForwardPropagationKernel(int mr2,
                                             npy_cfloat c_wvnb,
                                             FloatingType *r2pr, 
                                             npy_cfloat *py_data_u2, 
                                             int mr1step,
                                             int LastPart)
        {
        const int si2 = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x ;		// Grid is a "flatten" 1D, thread blocks are 1D

        FloatingType dx,dy,dz,R,r2x,r2y,r2z;
        FloatingType temp_r,tr ;
        FloatingType temp_i,ti,pCos,pSin ;


        if ( si2 < mr2)  
        {
            temp_r = 0;
            temp_i = 0;
            r2x=r2pr[si2*3];
            r2y=r2pr[si2*3+1];
            r2z=r2pr[si2*3+2];

            for (int si1=0; si1<mr1step; si1++)
            {
                /*
                dx=PartSource[si1]-r2pr[si2];
                dy=PartSource[si1+mr1step]-r2pr[si2+mr2];
                dz=PartSource[si1+2*mr1step]-r2pr[si2+2*mr2];
                */
                // In matlab we have a Fortran convention, in Python-numpy, we have the C-convention for matrixes (hoorray!!!)
                dx=PartSource[si1*3]-r2x;
                dy=PartSource[si1*3+1]-r2y;
                dz=PartSource[si1*3+2]-r2z;


                R=sqrt(dx*dx+dy*dy+dz*dz);
                ti=(__expf(R*c_wvnb.imag)*Part_a1pr[si1]/R);

                tr=ti;
                __sincosf(R*c_wvnb.real,&pSin,&pCos);
                tr*=(Part_u1complex[si1].real*pCos+Part_u1complex[si1].imag*pSin);
                            ti*=(Part_u1complex[si1].imag*pCos-Part_u1complex[si1].real*pSin);

                temp_r +=tr;
                temp_i +=ti;	
            }

                py_data_u2[si2].real+=temp_r;
                py_data_u2[si2].imag+=temp_i;

            if (LastPart==1)
            {

                temp_r=py_data_u2[si2].real;
                temp_i=py_data_u2[si2].imag;

                R=temp_r;

                temp_r = -temp_r*c_wvnb.imag-temp_i*c_wvnb.real;
                            temp_i = R*c_wvnb.real-temp_i*c_wvnb.imag;

                py_data_u2[si2].real=temp_r/(2*pi);
                py_data_u2[si2].imag=temp_i/(2*pi);
            }
        }
        }
      """


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
    FaceDisplay= np.zeros((0,4),np.int)

    for nr in range(len(BetaC)):

        Perim = np.sin(BetaC[nr])*Foc*2*np.pi

        nAlpha = np.ceil(Perim/lstep)
        sAlpha = 2*np.pi/nAlpha

        AlphaC = np.arange(sAlpha/2,sAlpha*(1/2 + nAlpha ),sAlpha)


        SingElem=np.vstack((SingElem,np.zeros((len(AlphaC),3))))
        N  = np.vstack((N,np.zeros((len(AlphaC),3))))
        ds = np.vstack((ds,np.zeros((len(AlphaC),1))))

        VertDisplay= np.vstack((VertDisplay,np.zeros((len(AlphaC)*4,3))))
        FaceDisplay= np.vstack((FaceDisplay,np.zeros((len(AlphaC),4),np.int)))


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

def InitCuda():
    global prg
    from pycuda.compiler import SourceModule
    prg  = SourceModule(RayleighCUDASource,include_dirs=[npyInc+os.sep+'numpy',info['include']])

def InitOpenCL(DeviceName='AMD'):
    global Platforms
    global queue 
    global prg 
    global ctx
    
    Platforms=cl.get_platforms()
    if len(Platforms)==0:
        raise SystemError("No OpenCL platforms")
    SelDevice=None
    for device in Platforms[0].get_devices():
        print(device.name)
        if DeviceName in device.name:
            SelDevice=device
    if SelDevice is None:
        raise SystemError("No OpenCL device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', SelDevice.name)
    ctx = cl.Context([SelDevice])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, RayleighOpenCLSource+OpenCLKernelBHTE).build()

    
def ForwardSimpleCUDA(cwvnb,center,ds,u0,rf):
    mr1=center.shape[0]
    mr2=rf.shape[0]
    u2=np.zeros(rf.shape[0],np.complex64)
    
    d_r2pr= cuda.mem_alloc(rf.nbytes)
    d_u2complex= cuda.mem_alloc(u2.nbytes)
    
    cuda.memcpy_htod(d_r2pr, rf)
    cuda.memcpy_htod(d_u2complex, u2)

    mr1start = 0
    LastPart = 0
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
                      
    ForwardPropagationKernel = prg.get_function("ForwardPropagationKernel")
    PartSource=prg.get_global("PartSource")
    Part_a1pr=prg.get_global("Part_a1pr")
    Part_u1complex=prg.get_global("Part_u1complex")

    while(mr1start < mr1):
        mr1stop = mr1start+MAX_ELEMS_IN_CONSTANT

        if (mr1stop >= mr1):
            mr1stop =  mr1;
            LastPart = 1
        mr1step = mr1stop - mr1start
        
        cuda.memcpy_htod(PartSource[0], center[mr1start:mr1start+mr1step,:])
        cuda.memcpy_htod(Part_a1pr[0], ds[mr1start:mr1start+mr1step])
        cuda.memcpy_htod(Part_u1complex[0], u0[mr1start:mr1start+mr1step])
        
        ForwardPropagationKernel(np.int32(mr2), cwvnb, d_r2pr, d_u2complex, np.int32(mr1step), np.int32(LastPart),block=dimThreadBlock, grid=dimBlockGrid)

        pycuda.autoinit.context.synchronize()
        mr1start += mr1step
    
    cuda.memcpy_dtoh(u2,d_u2complex)  
    return u2

def ForwardSimpleOpenCL(cwvnb,center,ds,u0,rf):
    global queue 
    global prg 
    global ctx
    mr1=np.array([center.shape[0]])
    mr2=np.array([rf.shape[0]])
    cwvnb_real=np.array([np.real(cwvnb)])
    cwvnb_imag=np.array([np.imag(cwvnb)])
    
    
    mf = cl.mem_flags
    
    d_mr1pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mr1)
    d_mr2pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mr2)
    d_cwvnb_realpr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cwvnb_real)
    d_cwvnb_imagpr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cwvnb_imag)
    
    d_r2pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rf)
    d_r1pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=center)
    d_u1realpr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.real(u0).copy())
    d_u1imagpr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.imag(u0).copy())
    d_a1pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ds)
    
    
    u2_real = np.zeros((rf.shape[0]),dtype=np.float32)
    u2_imag = np.zeros((rf.shape[0]),dtype=np.float32)
    
    d_u2realpr = cl.Buffer(ctx, mf.WRITE_ONLY, u2_real.nbytes)
    d_u2imagpr = cl.Buffer(ctx, mf.WRITE_ONLY, u2_real.nbytes)
    
    
    
    knl = prg.ForwardPropagationKernel  # Use this Kernel object for repeated calls
    knl(queue, u2_real.shape, None,
        d_mr2pr,
        d_cwvnb_realpr,
        d_cwvnb_imagpr,
        d_mr1pr,
        d_r2pr,
        d_r1pr,
        d_a1pr,
        d_u1realpr,
        d_u1imagpr,
        d_u2realpr,
        d_u2imagpr)
    
    cl.enqueue_copy(queue, u2_real,d_u2realpr)
    cl.enqueue_copy(queue, u2_imag,d_u2imagpr)
    u2=u2_real+1j*u2_imag
                                            
    return u2

def ForwardSimpleMetal(cwvnb,center,ds,u0,rf,deviceName):
    os.environ['__RayleighMetalDevice'] =deviceName
    bUseMappedMemory=0
    if 'arm64' in platform.platform() and\
        np.core.multiarray.get_handler_name(center)=="page_data_allocator":
        bUseMappedMemory=1
        #We assume arrays were allocated with page_data_allocator to have aligned date
        
    mr1=np.array([center.shape[0]])
    mr2=np.array([rf.shape[0]])
    ibUseMappedMemory =np.array([bUseMappedMemory])
    cwvnb_real=np.array([np.real(cwvnb)])
    cwvnb_imag=np.array([np.imag(cwvnb)])
    
    mr1_ptr = mr1.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mr2_ptr = mr2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bUseMappedMemory_ptr =ibUseMappedMemory.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cwvnb_real_ptr = cwvnb_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cwvnb_imag_ptr = cwvnb_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    r1_ptr=center.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    r2_ptr=rf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    a1_ptr=ds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    u1_real_ptr=np.real(u0).copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    u1_imag_ptr=np.imag(u0).copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    deviceName_ptr=ctypes.c_char_p(deviceName.encode())
    u2_real = np.zeros(rf.shape[0],np.float32)
    u2_imag = np.zeros(rf.shape[0],np.float32)
    u2_real_ptr = u2_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    u2_imag_ptr = u2_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    

    ret = swift_fun.ForwardSimpleMetal(mr2_ptr,
                                cwvnb_real_ptr,
                                cwvnb_imag_ptr,
                                mr1_ptr,
                                r2_ptr,
                                r1_ptr,
                                a1_ptr,
                                u1_real_ptr,
                                u1_imag_ptr,
                                deviceName_ptr,
                                u2_real_ptr,
                                u2_imag_ptr,
                                bUseMappedMemory_ptr)
    if ret ==1:
        raise ValueError("Unable to run simulation (mostly likely name of GPU is incorrect)")
    return u2_real+1j*u2_imag

def ForwardSimple(cwvnb,center,ds,u0,rf,MacOsPlatform='Metal',deviceMetal='6800'):
    '''
    MAIN function to call for ForwardRayleigh , returns the complex values of particle speed
    cwvnb is the complex speed of sound
    center is an [Mx3] array with the position of the decomposed transducer elements
    ds is an [M] array with the transducer element surface area
    u0 is [M] complex array with the particle speed at eact transducer element
    rf is [Nx3] array with the positons where Rayleigh integral will be calculated
    
    Function returns a [N] complex array of particle speed at locations rf
    
    '''
    global prg 
    if sys.platform == "darwin":
        if MacOsPlatform=='Metal':
            return ForwardSimpleMetal(cwvnb,center,ds,u0,rf,deviceMetal)
        else:
            return ForwardSimpleOpenCL(cwvnb,center,ds,u0,rf)
    else:
        return ForwardSimpleCUDA(cwvnb,center,ds,u0,rf)

    

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

def getPerfusionCoefficient( w_b,c_t,blood_rho,blood_ct,dt=0.1):
    """Calculates the perfusion coefficient based on the simulation parameters and time step """
    # get the perfusion coeff for a speicfic tissue type and time period  -- independent of surrounding tissue types
    # wb is in ml/min/kg, needs to be converted to m3/s/kg (1min/60 * 1e-6 m3/ml)

    coeff = w_b/60*1.0e-6* blood_rho * blood_ct * dt / c_t  

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

def BHTEeOpenCL(Pressure,MaterialMap,MaterialList,dx,
                TotalDurationSeps,nStepsOn,LocationMonitoring,
                nFactorMonitoring=1,
                dt=0.1,blood_rho=1050,blood_ct=3617,stableTemp=37.0,DutyCycle=1.0):
    global queue 
    global prg 
    global ctx

    perfArr=np.zeros(MaterialMap.max()+1,np.float32)
    bhArr=np.zeros(MaterialMap.max()+1,np.float32)
    initTemp = np.zeros(MaterialMap.shape, dtype=np.float32) 
    Qarr=np.zeros(MaterialMap.shape, dtype=np.float32) 

    for n in range(MaterialMap.max()+1):
        bhArr[n]=getBHTECoefficient(MaterialList['Conductivity'][n],MaterialList['Density'][n],
                                    MaterialList['SpecificHeat'][n],dx,TotalDurationSeps,dt=dt)
        perfArr[n]=getPerfusionCoefficient(MaterialList['Perfusion'][n],MaterialList['SpecificHeat'][n],
                                           blood_rho,blood_ct,dt=dt)
        initTemp[MaterialMap==n]=MaterialList['InitTemperature'][n]
        print(n,(MaterialMap==n).sum(),Pressure[MaterialMap==n].mean())

        Qarr[MaterialMap==n]=Pressure[MaterialMap==n]**2*getQCoeff(MaterialList['Density'][n],
                                                                  MaterialList['SoS'][n],
                                                                  MaterialList['Attenuation'][n],
                                                                  MaterialList['SpecificHeat'][n],
                                                                  MaterialList['Absorption'][n],
                                                                  dx,dt)*DutyCycle

    N1=np.int32(Pressure.shape[0])
    N2=np.int32(Pressure.shape[1])
    N3=np.int32(Pressure.shape[2])
    coreTemp = np.array([stableTemp],np.float32)
    initDose = np.zeros(MaterialMap.shape, dtype=np.float32)

    mf = cl.mem_flags

    d_perfArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=perfArr)
    d_bhArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bhArr)
    d_Qarr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Qarr)
    d_MaterialMap=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MaterialMap)

    T0 = np.zeros(initTemp.shape,dtype=np.float32)
    T1 = np.zeros(initTemp.shape,dtype=np.float32)

    d_T0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=initTemp)
    d_T1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=T1)

    Dose0 = initDose
    Dose1 = np.zeros(MaterialMap.shape,dtype=np.float32)

    d_Dose0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose0)
    d_Dose1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose1)

    TotalStepsMonitoring=int(TotalDurationSeps/nFactorMonitoring)
    if TotalStepsMonitoring % nFactorMonitoring!=0:
        TotalStepsMonitoring+=1
    MonitorSlice=np.zeros((MaterialMap.shape[0],MaterialMap.shape[2],TotalStepsMonitoring),np.float32)
    d_MonitorSlice = cl.Buffer(ctx, mf.WRITE_ONLY, MonitorSlice.nbytes)

    knl = prg.BHTEFDTDKernel

    l1=factors_gpu(MaterialMap.shape[0])
    if len(l1)>0:
        local=[l1[0],l1[0],1]
    else:
        local=None

    #this will calculate the optimal global size to make it multiple of [4,4,4]

    gl=[]
    for n in range(3):
        m=MaterialMap.shape[n]
        while(not np.any(factors_gpu(m)==4)):
            m+=1
        gl.append(m)
    nFraction=TotalDurationSeps/10
    for n in range(TotalDurationSeps):
        if n<nStepsOn:
            dUS=1
        else:
            dUS=0
        if (n%2==0):
            knl(queue,gl , [4,4,4],
                d_T1,
                d_Dose1,
                d_T0,
                d_Dose0,
                d_bhArr,
                d_perfArr,
                d_MaterialMap,
                d_Qarr,
                np.float32(coreTemp),
                np.int32(dUS),
                N1,
                N2,
                N3,
                np.float32(dt),
                d_MonitorSlice,
                np.int32(TotalStepsMonitoring),
                np.int32(nFactorMonitoring),
                np.int32(n),
                np.int32(LocationMonitoring),
                np.uint32(0))

        else:
            knl(queue, gl , [4,4,4],
                d_T0,
                d_Dose0,
                d_T1,
                d_Dose1,
                d_bhArr,
                d_perfArr,
                d_MaterialMap,
                d_Qarr,
                np.float32(coreTemp),
                np.int32(dUS),
                N1,
                N2,
                N3,
                np.float32(dt),
                d_MonitorSlice,
                np.int32(TotalStepsMonitoring),
                np.int32(nFactorMonitoring),
                np.int32(n),
                np.int32(LocationMonitoring),
                np.uint32(0))
        queue.finish()
        if n % nFraction ==0:
            print(n,TotalDurationSeps)

    if (n%2==0):
        ResTemp=d_T1
        ResDose=d_Dose1
    else:
        ResTemp=d_T0
        ResDose=d_Dose0


    print('Done BHTE')                               
    cl.enqueue_copy(queue, T1,ResTemp)
    cl.enqueue_copy(queue, Dose1,ResDose)
    cl.enqueue_copy(queue, MonitorSlice,d_MonitorSlice)
    queue.finish()
    return T1,Dose1,MonitorSlice,Qarr


def BHTEeOpenCLMultiplePressureFields(PressureFields,
                MaterialMap,
                MaterialList,
                dx,
                TotalDurationSeps,
                nStepsOnOffList,
                LocationMonitoring,
                nFactorMonitoring=1,
                dt=0.1,
                blood_rho=1050,
                blood_ct=3617,
                stableTemp=37.0):
    global queue 
    global prg 
    global ctx

    assert(PressureFields.shape[1]==MaterialMap.shape[0] and \
           PressureFields.shape[2]==MaterialMap.shape[1] and \
           PressureFields.shape[3]==MaterialMap.shape[2])

    assert(nStepsOnOffList.shape[0]==PressureFields.shape[0])

    perfArr=np.zeros(MaterialMap.max()+1,np.float32)
    bhArr=np.zeros(MaterialMap.max()+1,np.float32)
    initTemp = np.zeros(MaterialMap.shape, dtype=np.float32) 
    
    QArrList=np.zeros(PressureFields.shape, dtype=np.float32) 
    
    for n in range(MaterialMap.max()+1):
        bhArr[n]=getBHTECoefficient(MaterialList['Conductivity'][n],MaterialList['Density'][n],
                                    MaterialList['SpecificHeat'][n],dx,TotalDurationSeps,dt=dt)
        perfArr[n]=getPerfusionCoefficient(MaterialList['Perfusion'][n],MaterialList['SpecificHeat'][n],
                                           blood_rho,blood_ct,dt=dt)
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
    NstepsPerCycle=TimingFields[-1,3]
    

    N1=np.int32(MaterialMap.shape[0])
    N2=np.int32(MaterialMap.shape[1])
    N3=np.int32(MaterialMap.shape[2])
    coreTemp = np.array([stableTemp],np.float32)
    initDose = np.zeros(MaterialMap.shape, dtype=np.float32)

    mf = cl.mem_flags

    d_perfArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=perfArr)
    d_bhArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bhArr)
    d_QArrList=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=QArrList)
    d_MaterialMap=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MaterialMap)

    T0 = np.zeros(initTemp.shape,dtype=np.float32)
    T1 = np.zeros(initTemp.shape,dtype=np.float32)

    d_T0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=initTemp)
    d_T1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=T1)

    Dose0 = initDose
    Dose1 = np.zeros(MaterialMap.shape,dtype=np.float32)

    d_Dose0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose0)
    d_Dose1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose1)

    TotalStepsMonitoring=int(TotalDurationSeps/nFactorMonitoring)
    if TotalStepsMonitoring % nFactorMonitoring!=0:
        TotalStepsMonitoring+=1
    MonitorSlice=np.zeros((MaterialMap.shape[0],MaterialMap.shape[2],TotalStepsMonitoring),np.float32)
    d_MonitorSlice = cl.Buffer(ctx, mf.WRITE_ONLY, MonitorSlice.nbytes)

    knl = prg.BHTEFDTDKernel

    l1=factors_gpu(MaterialMap.shape[0])
    if len(l1)>0:
        local=[l1[0],l1[0],1]
    else:
        local=None

    #this will calculate the optimal global size to make it multiple of [4,4,4]

    gl=[]
    for n in range(3):
        m=MaterialMap.shape[n]
        while(not np.any(factors_gpu(m)==4)):
            m+=1
        gl.append(m)
    nFraction=TotalDurationSeps/10
    for n in range(TotalDurationSeps):
        mStep=n % NstepsPerCycle
        QSegment=np.where((TimingFields[:,0]<=mStep) & (TimingFields[:,2]>mStep))[0][0]
        if mStep<TimingFields[QSegment,1]:
            dUS=1
        else:
            assert(mStep>=TimingFields[QSegment,1])
            dUS=0
        StartIndexQ=np.prod(np.array(QArrList.shape[1:]))*QSegment
        if (n%2==0):
            knl(queue,gl , [4,4,4],
                d_T1,
                d_Dose1,
                d_T0,
                d_Dose0,
                d_bhArr,
                d_perfArr,
                d_MaterialMap,
                d_QArrList,
                np.float32(coreTemp),
                np.int32(dUS),
                N1,
                N2,
                N3,
                np.float32(dt),
                d_MonitorSlice,
                np.int32(TotalStepsMonitoring),
                np.int32(nFactorMonitoring),
                np.int32(n),
                np.int32(LocationMonitoring),
                np.uint32(StartIndexQ))

        else:
            knl(queue, gl , [4,4,4],
                d_T0,
                d_Dose0,
                d_T1,
                d_Dose1,
                d_bhArr,
                d_perfArr,
                d_MaterialMap,
                d_QArrList,
                np.float32(coreTemp),
                np.int32(dUS),
                N1,
                N2,
                N3,
                np.float32(dt),
                d_MonitorSlice,
                np.int32(TotalStepsMonitoring),
                np.int32(nFactorMonitoring),
                np.int32(n),
                np.int32(LocationMonitoring),
                np.uint32(StartIndexQ))
        queue.finish()
        if n % nFraction ==0:
            print(n,TotalDurationSeps)

    if (n%2==0):
        ResTemp=d_T1
        ResDose=d_Dose1
    else:
        ResTemp=d_T0
        ResDose=d_Dose0


    print('Done BHTE')                               
    cl.enqueue_copy(queue, T1,ResTemp)
    cl.enqueue_copy(queue, Dose1,ResDose)
    cl.enqueue_copy(queue, MonitorSlice,d_MonitorSlice)
    queue.finish()
    return T1,Dose1,MonitorSlice,QArrList