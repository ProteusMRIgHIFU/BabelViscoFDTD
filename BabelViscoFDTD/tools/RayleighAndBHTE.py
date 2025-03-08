'''
Samuel Pichardo, Ph.D.
University of Calgary

Very simple GPU-accelerated Rayleigh integral with support to OpenCL and CUDA

CUDA is automatically selected if running Windows or Linux, while OpenCL is selected in MacOs
'''

import numpy as np
import os
import sys
from sysconfig import get_paths
import ctypes
import sys 
import platform

KernelCoreSourceBHTE="""
    #define Tref 43.0
     unsigned int DzDy=outerDimz*outerDimy;
     unsigned int coord = gtidx*DzDy + gtidy*outerDimz + gtidz;
    
    float R1,R2,dtp;
    if(gtidx > 0 && gtidx < outerDimx-1 && gtidy > 0 && gtidy < outerDimy-1 && gtidz > 0 && gtidz < outerDimz-1)
    {

            const  unsigned int label = d_labels[coord];

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
                d_output2[coord] = d_input2[coord] + dt * pow((float)R1,(float)(Tref-d_input[coord]));
            }
            else
            {
                if(R1 == R2)
                {
                    d_output2[coord] = d_input2[coord] + (pow((float)R2,(float)(Tref-d_output[coord])) - pow((float)R1,(float)(Tref-d_input[coord]))) / 
                                   ( -(d_output[coord]-d_input[coord])/ dt * log(R1));
                }
                else
                {
                    dtp = dt * (Tref - d_input[coord])/(d_output[coord] - d_input[coord]);

                    d_output2[coord] = d_input2[coord] + (1 - pow((float)R1,(float)(Tref-d_input[coord])))     / (- (Tref - d_input[coord])/ dtp * log(R1)) + 
                                   (pow((float)R2,(float)(Tref-d_output[coord])) - 1) / (-(d_output[coord] - Tref)/(dt - dtp) * log(R2));
                }
            }

            if (gtidy==SelJ && (n_Step % nFactorMonitoring ==0))
            {
                 d_MonitorSlice[gtidx*outerDimz*TotalStepsMonitoring+gtidz*TotalStepsMonitoring+ n_Step/nFactorMonitoring] =d_output[coord];
            }

            if (d_pointsMonitoring[coord]>0)
            {
                d_Temppoints[TotalSteps*(d_pointsMonitoring[coord]-1)+n_Step]=d_output[coord];
            }
        }
        else if(gtidx < outerDimx && gtidy < outerDimy && gtidz < outerDimz){
            d_output[coord] = d_input[coord];
            d_output2[coord] = d_input2[coord];

        }

}
"""

import pyopencl as cl

RayleighOpenCLMetalSource="""
#ifdef _METAL
#include <metal_stdlib>
using namespace metal;
#endif

#define pi 3.141592653589793
#define ppCos &pCos

typedef float FloatingType;

#ifdef _OPENCL
__kernel  void ForwardPropagationKernel(  const int mr2,
                                            const FloatingType c_wvnb_real,
                                            const FloatingType c_wvnb_imag,
                                            const FloatingType MaxDistance,
                                            const int mr1,
                                            __global const FloatingType *r2pr, 
                                            __global const FloatingType *r1pr, 
                                            __global const FloatingType *a1pr, 
                                            __global const FloatingType *u1_real, 
                                            __global const FloatingType *u1_imag,
                                            __global  FloatingType  *py_data_u2_real,
                                            __global  FloatingType  *py_data_u2_imag,
                                            const int mr1step
                                            )
    {
    int si2 = get_global_id(0);		// Grid is a "flatten" 1D, thread blocks are 1D

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

        for (int si1=0; si1<mr1; si1++)
        {
            // In matlab we have a Fortran convention, in Python-numpy, we have the C-convention for matrixes (hoorray!!!)
            dx=r1pr[si1*3]-r2x;
            dy=r1pr[si1*3+1]-r2y;
            dz=r1pr[si1*3+2]-r2z;


            R=sqrt(dx*dx+dy*dy+dz*dz);
            if (MaxDistance>0.0)
                if (R>MaxDistance)
                    continue;
            ti=(exp(R*c_wvnb_imag)*a1pr[si1]/R);

            tr=ti;
            pSin=sincos(R*c_wvnb_real,ppCos);

            tr*=(u1_real[si1+mr1step*si2]*pCos+u1_imag[si1+mr1step*si2]*pSin);
                        ti*=(u1_imag[si1+mr1step*si2]*pCos-u1_real[si1+mr1step*si2]*pSin);

            temp_r +=tr;
            temp_i +=ti;	
        }
        
        R=temp_r;

        temp_r = -temp_r*c_wvnb_imag-temp_i*c_wvnb_real;
        temp_i = R*c_wvnb_real-temp_i*c_wvnb_imag;

        py_data_u2_real[si2]=temp_r/(2*pi);
        py_data_u2_imag[si2]=temp_i/(2*pi);
    }
    }
#endif
    """

OpenCLMetalHeaderBHTE="""
#ifdef _OPENCL
    __kernel  void BHTEFDTDKernel( __global float 		*d_output, 
                                    __global float 		*d_output2,
                                    __global const float 			*d_input, 
                                    __global const float 			*d_input2,
                                    __global const float 			*d_bhArr,
                                    __global const float 			*d_perfArr, 
                                    __global const unsigned int		*d_labels,
                                    __global const float 			*d_Qarr,
                                    __global const unsigned int		*d_pointsMonitoring,
                                        const float 			CoreTemp,
                                        const  unsigned int				sonication,
                                        const  unsigned int				outerDimx, 
                                        const  unsigned int              outerDimy, 
                                        const  unsigned int              outerDimz,
                                        const float 			dt,
                                        __global  float 	*d_MonitorSlice,
                                        __global float      *d_Temppoints,
                                        const  unsigned int TotalStepsMonitoring,
                                        const  unsigned int nFactorMonitoring,
                                        const  unsigned int n_Step,
                                        const unsigned int SelJ,
                                        const unsigned int StartIndexQ,
                                        const  unsigned TotalSteps)	
    {
        const int gtidx =  get_global_id(0);
        const int gtidy =  get_global_id(1);
        const int gtidz =  get_global_id(2);
#endif
#ifdef _METAL
        kernel  void BHTEFDTDKernel( device float 		*d_output [[ buffer(0) ]], 
                                    device float 		*d_output2 [[ buffer(1) ]],
                                    device const float 			*d_input [[ buffer(2) ]], 
                                    device const float 			*d_input2 [[ buffer(3) ]],
                                    device const float 			*d_bhArr [[ buffer(4) ]],
                                    device const float 			*d_perfArr [[ buffer(5) ]], 
                                    device const unsigned int		*d_labels [[ buffer(6) ]],
                                    device const float 			*d_Qarr [[ buffer(7) ]],
                                    device const unsigned int		*d_pointsMonitoring [[ buffer(8) ]],
                                    device  float 	*d_MonitorSlice [[ buffer(9) ]],
                                    device  float 	*d_Temppoints [[ buffer(10) ]],
                                        constant float * floatParams [[ buffer(11) ]],
                                        constant unsigned int * intparams [[ buffer(12) ]],
                                        uint gid[[thread_position_in_grid]])	
    {

        #define CoreTemp floatParams[0]
        #define dt floatParams[1]
        #define sonication intparams[0]
        #define outerDimx intparams[1]
        #define outerDimy intparams[2]
        #define outerDimz intparams[3]
        #define TotalStepsMonitoring intparams[4]
        #define nFactorMonitoring intparams[5]
        #define n_Step intparams[6]
        #define SelJ intparams[7]
        #define StartIndexQ intparams[8]
        #define TotalSteps intparams[9]
        const int gtidx =  gid/(outerDimy*outerDimz);
        const int gtidy =  (gid - gtidx*outerDimy*outerDimz)/outerDimz;
        const int gtidz =  gid - gtidx*outerDimy*outerDimz - gtidy*outerDimz;

#endif
    """

OpenCLKernelBHTE =OpenCLMetalHeaderBHTE + KernelCoreSourceBHTE


Platforms=None
queue = None
prgcl = None
ctx = None

if sys.platform == "darwin":
    
    import metalcomputebabel as mc

    # Loads METAL interface
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

else:

    prgcuda = None

    import cupy as cp
    
    RayleighCUDASource="""
   #include <cupy/complex.cuh>

    #define pi 3.141592653589793

    typedef float FloatingType;

    #define MAX_ELEMS_IN_CONSTANT  2730 // the total constant memory can't be greater than 64k bytes


    __device__ __forceinline__ complex<float> cuexpf (complex<float> z)

        {
            float res_i,res_r;
            sincosf(z.imag(), &res_i, &res_r);
            return expf (z.real())*complex<float> (res_r,res_i);;
        }
    
    extern "C" __global__ void ForwardPropagationKernel(int mr2,
                                             complex<float> c_wvnb,
                                             FloatingType MaxDistance, 
                                             FloatingType *r2pr, 
                                             FloatingType *r1pr, 
                                             FloatingType *a1pr,
                                             complex<float> * u1complex,
                                             complex<float> *py_data_u2, 
                                             int mr1,
                                             int mr1step)
        {
        const int si2 = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x ;		// Grid is a "flatten" 1D, thread blocks are 1D

        complex<float> cj=complex<float>(0.0,1);
        complex<float> temp,temp2;
        
        FloatingType dx,dy,dz,R,r2x,r2y,r2z;
        if ( si2 < mr2)  
        {
            temp*=0;
            

            r2x=r2pr[si2*3];
            r2y=r2pr[si2*3+1];
            r2z=r2pr[si2*3+2];

            for (int si1=0; si1<mr1; si1++)
            {

                // In matlab we have a Fortran convention, in Python-numpy, we have the C-convention for matrixes (hoorray!!!)
                dx=r1pr[si1*3]-r2x;
                dy=r1pr[si1*3+1]-r2y;
                dz=r1pr[si1*3+2]-r2z;

                R=sqrt(dx*dx+dy*dy+dz*dz);
                if (MaxDistance>0.0)
                    if (R>MaxDistance)
                        continue;

                temp2=cj*c_wvnb;
                temp2=temp2*(-R);
                temp2=cuexpf(temp2);
                temp2=temp2*u1complex[si1+mr1step*si2];
                temp2=temp2*a1pr[si1]/R;
                temp=temp+temp2;
            }

            temp2=cj*c_wvnb;
            temp=temp*temp2;

            py_data_u2[si2]=temp/((float)(2*pi));
            
        }
        }
      """
    
    CUDAHeaderBHTE="""

        extern "C" __global__   void BHTEFDTDKernel(  float 		        *d_output, 
                                        float 		            *d_output2,
                                        const float 			*d_input, 
                                        const float 			*d_input2,
                                        const float 			*d_bhArr,
                                        const float 			*d_perfArr, 
                                        const unsigned int		*d_labels,
                                        const float 			*d_Qarr,
                                        const unsigned int		*d_pointsMonitoring,
                                        const float 			CoreTemp,
                                        const  int				sonication,
                                        const  int				outerDimx, 
                                        const  int              outerDimy, 
                                        const  int              outerDimz,
                                        const float 			dt,
                                        float 	                *d_MonitorSlice,
                                        float 			        *d_Temppoints,
                                        const  int              TotalStepsMonitoring,
                                        const  int              nFactorMonitoring,
                                        const  int              n_Step,
                                        const int               SelJ,
                                        const unsigned int StartIndexQ,
                                        const  unsigned TotalSteps)	
        {
            const int gtidx = (blockIdx.x * blockDim.x + threadIdx.x);
            const int gtidy = (blockIdx.y * blockDim.y + threadIdx.y);
            const int gtidz = (blockIdx.z * blockDim.z + threadIdx.z);
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

def InitCuda(DeviceName=None):
    global prgcuda
    devCount = cp.cuda.runtime.getDeviceCount()
    if devCount == 0:
        raise SystemError("There are no CUDA devices.")
        
    if DeviceName is not None:
        selDevice = None
        for deviceID in range(0, devCount):
            d=cp.cuda.runtime.getDeviceProperties(deviceID)
            if DeviceName in d['name'].decode('UTF-8'):
                selDevice=cp.cuda.Device(deviceID)
                break
        selDevice.use()
    AllCudaCode=RayleighCUDASource + CUDAHeaderBHTE + KernelCoreSourceBHTE
    prgcuda  = cp.RawModule(code= AllCudaCode)
 
def InitOpenCL(DeviceName='AMD'):
    global Platforms
    global queue 
    global prgcl 
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
    prgcl = cl.Program(ctx, '#define _OPENCL\n'+RayleighOpenCLMetalSource+OpenCLKernelBHTE).build()

def InitMetal(DeviceName='AMD'):
    global ctx
    global prgcl 

    devices = mc.get_devices()
    SelDevice=None
    for n,dev in enumerate(devices):
        if DeviceName in dev.deviceName:
            SelDevice=dev
            break
    if SelDevice is None:
        raise SystemError("No Metal device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', dev.deviceName)
    
    ctx = mc.Device(n)
    print(ctx)
    if 'arm64' not in platform.platform():
        ctx.set_external_gpu(1) 
    prgcl = ctx.kernel('#define _METAL\n'+RayleighOpenCLMetalSource+OpenCLKernelBHTE)
    
def ForwardSimpleCUDA(cwvnb,center,ds,u0,rf,MaxDistance=-1.0,u0step=0):
    if u0step!=0:
        mr1=u0step
        assert(mr1*rf.shape[0]==u0.shape[0])
        assert(mr1==center.shape[0])
        assert(mr1==ds.shape[0])
    else:
        mr1=center.shape[0]
    mr2=rf.shape[0]

    d_r2pr= cp.asarray(rf)
    d_centerpr= cp.asarray(center)
    d_dspr= cp.asarray(ds)
    d_u0complex= cp.asarray(u0)
    d_u2complex= cp.zeros(rf.shape[0],cp.complex64)

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
                      
    ForwardPropagationKernel = prgcuda.get_function("ForwardPropagationKernel")
    
        
    ForwardPropagationKernel(dimBlockGrid,
                            dimThreadBlock,
                            (mr2, 
                             cwvnb,
                             MaxDistance,
                             d_r2pr, 
                             d_centerpr,
                             d_dspr,
                             d_u0complex, 
                             d_u2complex, 
                             mr1,
                             u0step))

    u2=d_u2complex.get()
    return u2

def ForwardSimpleOpenCL(cwvnb,center,ds,u0,rf,MaxDistance=-1.0,u0step=0):
    global queue 
    global prg 
    global ctx
    mf = cl.mem_flags
    
    if u0step!=0:
        mr1=u0step
        assert(mr1*rf.shape[0]==u0.shape[0])
        assert(mr1==center.shape[0])
        assert(mr1==ds.shape[0])
    else:
        mr1=center.shape[0]
    
     
    d_r2pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rf)
    d_r1pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=center)
    d_u1realpr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.real(u0).copy())
    d_u1imagpr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.imag(u0).copy())
    d_a1pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ds)
    
    
    u2_real = np.zeros((rf.shape[0]),dtype=np.float32)
    u2_imag = np.zeros((rf.shape[0]),dtype=np.float32)
    
    d_u2realpr = cl.Buffer(ctx, mf.WRITE_ONLY, u2_real.nbytes)
    d_u2imagpr = cl.Buffer(ctx, mf.WRITE_ONLY, u2_real.nbytes)
    
    
    
    knl = prgcl.ForwardPropagationKernel  # Use this Kernel object for repeated calls
    if u2_real.shape[0] % 64 ==0:
        ks=[u2_real.shape[0]]
    else:
        ks=[int(u2_real.shape[0]/64)*64+64]
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
    
    cl.enqueue_copy(queue, u2_real,d_u2realpr)
    cl.enqueue_copy(queue, u2_imag,d_u2imagpr)
    u2=u2_real+1j*u2_imag
                                            
    return u2

def ForwardSimpleMetal(cwvnb,center,ds,u0,rf,deviceName,MaxDistance=-1.0,u0step=0):
    os.environ['__BabelMetalDevice'] =deviceName
    bUseMappedMemory=0
    if np.__version__ >="1.22.0":
        if 'arm64' in platform.platform() and\
            np.core.multiarray.get_handler_name(center)=="page_data_allocator":
            bUseMappedMemory=1
        #We assume arrays were allocated with page_data_allocator to have aligned date
        
    
    mr2=np.array([rf.shape[0]])

    if u0step!=0:
        mr1=u0step
        assert(mr1*rf.shape[0]==u0.shape[0])
        assert(mr1==center.shape[0])
        assert(mr1==ds.shape[0])
    else:
        mr1=center.shape[0]

    mr1=np.array([mr1])
    u0step_a=np.array([u0step])
    MaxDistance_a=np.array([MaxDistance]).astype(np.float32)

    ibUseMappedMemory =np.array([bUseMappedMemory])
    cwvnb_real=np.array([np.real(cwvnb)])
    cwvnb_imag=np.array([np.imag(cwvnb)])
    
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
    deviceName_ptr=ctypes.c_char_p(deviceName.encode())
    u2_real = np.zeros(rf.shape[0],np.float32)
    u2_imag = np.zeros(rf.shape[0],np.float32)
    u2_real_ptr = u2_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    u2_imag_ptr = u2_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    

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

    return u2_real+1j*u2_imag

def ForwardSimple(cwvnb,center,ds,u0,rf,MaxDistance=-1.0,u0step=0,MacOsPlatform='Metal',deviceMetal='6800'):
    '''
    MAIN function to call for ForwardRayleigh , returns the complex values of particle speed
    cwvnb is the complex speed of sound
    center is an [Mx3] array with the position of the decomposed transducer elements
    ds is an [M] array with the transducer element surface area
    u0 is [M] complex array with the particle speed at eact transducer element
    rf is [Nx3] array with the positons where Rayleigh integral will be calculated
    
    Function returns a [N] complex array of particle speed at locations rf
    
    '''
    global prgcuda 
    if sys.platform == "darwin":
        if MacOsPlatform=='Metal':
            return ForwardSimpleMetal(cwvnb,center,ds,u0,rf,deviceMetal,MaxDistance=MaxDistance,u0step=u0step)
        else:
            return ForwardSimpleOpenCL(cwvnb,center,ds,u0,rf,MaxDistance=MaxDistance,u0step=u0step)
    else:
        return ForwardSimpleCUDA(cwvnb,center,ds,u0,rf,MaxDistance=MaxDistance,u0step=u0step)

    

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

def  getPerfusionCoefficient( w_b,c_t,blood_rho,blood_ct,dt=0.1):
    """Calculates the perfusion coefficient based on the simulation parameters and time step """
    # get the perfusion coeff for a speicfic tissue type and time period  -- independent of surrounding tissue types
    # wb is in ml/min/kg, needs to be converted to kg/m3/s (1min/60 * 1e-6 m3/ml) x blood density x tissue density
    # Camilleri et al. 2022. https://doi.org/10.3390/
    # 
    #In the BHTE numerical solution, wb is part of the term  dt wb c_b / (rho c) (T - Tb)
    #which makes that the coefficients as below (rho tissue disappears)

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

def BHTE(Pressure,MaterialMap,MaterialList,dx,
                TotalDurationSteps,nStepsOn,LocationMonitoring,
                nFactorMonitoring=1,
                dt=0.1,blood_rho=1050,blood_ct=3617,
                stableTemp=37.0,DutyCycle=1.0,
                Backend='OpenCL',
                MonitoringPointsMap=None,
                initT0=None,
                initDose=None):
    global queue 
    global prgcl 
    global prgcuda
    global ctx

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
                                           MaterialList['SpecificHeat'][n],
                                           blood_rho,
                                           blood_ct,
                                           dt=dt)
        if initT0 is None:
            initTemp[MaterialMap==n]=MaterialList['InitTemperature'][n]
        #print(n,(MaterialMap==n).sum(),Pressure[MaterialMap==n].mean())

        Qarr[MaterialMap==n]=Pressure[MaterialMap==n]**2*getQCoeff(MaterialList['Density'][n],
                                                                  MaterialList['SoS'][n],
                                                                  MaterialList['Attenuation'][n],
                                                                  MaterialList['SpecificHeat'][n],
                                                                  MaterialList['Absorption'][n],
                                                                  dx,dt)*DutyCycle

    N1=np.int32(Pressure.shape[0])
    N2=np.int32(Pressure.shape[1])
    N3=np.int32(Pressure.shape[2])
    
    if initDose is None:
        initDose = np.zeros(MaterialMap.shape, dtype=np.float32)

    if MonitoringPointsMap is not None:
        MonitoringPoints = MonitoringPointsMap
        TotalPointsMonitoring=np.sum((MonitoringPointsMap>0).astype(int))
        TemperaturePoints=np.zeros((TotalPointsMonitoring,TotalDurationSteps),np.float32)
    else:
        MonitoringPoints = np.zeros(MaterialMap.shape, dtype=np.uint32)
        TemperaturePoints=np.zeros((10),np.float32) #just dummy array

    TotalStepsMonitoring=int(TotalDurationSteps/nFactorMonitoring)
    if TotalStepsMonitoring % nFactorMonitoring!=0:
        TotalStepsMonitoring+=1
    MonitorSlice=np.zeros((MaterialMap.shape[0],MaterialMap.shape[2],TotalStepsMonitoring),np.float32)

    T1 = np.zeros(initTemp.shape,dtype=np.float32)

    Dose0 = initDose
    Dose1 = np.zeros(MaterialMap.shape,dtype=np.float32)

    nFraction=int(TotalDurationSteps/10)
    if nFraction ==0:
        nFraction=1

    if Backend=='OpenCL':

        mf = cl.mem_flags

        d_perfArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=perfArr)
        d_bhArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bhArr)
        d_Qarr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Qarr)
        d_MaterialMap=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MaterialMap)
        d_MonitoringPoints=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MonitoringPoints)


        d_T0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=initTemp)
        d_T1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=T1)

        
        d_Dose0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose0)
        d_Dose1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose1)

        d_MonitorSlice = cl.Buffer(ctx, mf.WRITE_ONLY, MonitorSlice.nbytes)
        d_TemperaturePoints = cl.Buffer(ctx, mf.WRITE_ONLY, TemperaturePoints.nbytes)

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
        cl.enqueue_copy(queue, T1,ResTemp)
        cl.enqueue_copy(queue, Dose1,ResDose)
        cl.enqueue_copy(queue, MonitorSlice,d_MonitorSlice)
        cl.enqueue_copy(queue, TemperaturePoints,d_TemperaturePoints)
        queue.finish()

    elif Backend=='CUDA':

        dimBlockBHTE = (4,4,4)

        dimGridBHTE  = (int(N1/dimBlockBHTE[0]+1),
                        int(N2/dimBlockBHTE[1]+1),
                        int(N3/dimBlockBHTE[2]+1))

        d_perfArr=cp.asarray(perfArr)
        d_bhArr=cp.asarray(bhArr)
        d_Qarr=cp.asarray(Qarr)
        d_MaterialMap=cp.asarray(MaterialMap)
        d_T0 = cp.asarray(initTemp)
        d_T1 = cp.asarray(T1)
        d_Dose0 = cp.asarray(Dose0)
        d_Dose1 = cp.asarray(Dose1)
        d_MonitoringPoints=cp.asarray(MonitoringPoints)
     
        d_MonitorSlice = cp.zeros(MonitorSlice.shape,cp.float32)
        d_TemperaturePoints=cp.zeros(TemperaturePoints.shape,cp.float32)

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
            cp.cuda.Stream.null.synchronize()
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
    global queue 
    global prgcl 
    global ctx

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
                                           MaterialList['SpecificHeat'][n],
                                           blood_rho,
                                           blood_ct,
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

        mf = cl.mem_flags

        d_perfArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=perfArr)
        d_bhArr=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bhArr)
        d_QArrList=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=QArrList)
        d_MaterialMap=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MaterialMap)
        d_MonitoringPoints=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MonitoringPoints)

        d_T0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=initTemp)
        d_T1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=T1)
        d_Dose0 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose0)
        d_Dose1 = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Dose1)

        d_MonitorSlice = cl.Buffer(ctx, mf.WRITE_ONLY, MonitorSlice.nbytes)
        d_TemperaturePoints = cl.Buffer(ctx, mf.WRITE_ONLY, TemperaturePoints.nbytes)

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
        cl.enqueue_copy(queue, T1,ResTemp)
        cl.enqueue_copy(queue, Dose1,ResDose)
        cl.enqueue_copy(queue, MonitorSlice,d_MonitorSlice)
        cl.enqueue_copy(queue, TemperaturePoints,d_TemperaturePoints)
        queue.finish()
    elif Backend=='CUDA':
        dimBlockBHTE = (4,4,4)

        dimGridBHTE  = (int(N1/dimBlockBHTE[0]+1),
                        int(N2/dimBlockBHTE[1]+1),
                        int(N3/dimBlockBHTE[2]+1))

        d_perfArr=cp.asarray(perfArr)
        d_bhArr=cp.asarray(bhArr)
        d_QArrList=cp.asarray(QArrList)
        d_MaterialMap=cp.asarray(MaterialMap)
        d_T0 = cp.asarray(initTemp)
        d_T1 = cp.asarray(T1)
        d_Dose0 = cp.asarray(Dose0)
        d_Dose1 = cp.asarray(Dose1)
        d_MonitoringPoints=cp.asarray(MonitoringPoints)
     
        d_MonitorSlice = cp.zeros(MonitorSlice.shape,cp.float32)
        d_TemperaturePoints=cp.zeros(TemperaturePoints.shape,cp.float32)

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
            cp.cuda.Stream.null.synchronize()
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
