
'''
SimpleBenchmark.py

Samuel Pichardo, Ph.D.
Assistant Professor, University of Calgary

usage: python SimpleBenchmark.py [options]

Run a quick simulation for benchmark Metal vs OpenCL

positional arguments:
  GPU_ID                Substring for GPU id such as "M1"
  {Metal,OpenCL}        Backend to test, it must be one of the available options shown on the left

optional arguments:
  -h, --help            show this help message and exit
  --EnableMetalCapture, --no-EnableMetalCapture
                        GPU Capture will be done when running Metal backend, be sure of running with environment variable
                        METAL_CAPTURE_ENABLED=1 (default: None)
  --ShortRun, --no-ShortRun
                        Enable short simulation (1/10 of normal length), useful when doing Metal capture to reduce size of
                        capture file (default: None)
'''

import numpy as np
import argparse
from BabelViscoFDTD.PropagationModel import PropagationModel
from BabelViscoFDTD.tools.RayleighAndBHTE import StartMetaCapture, Stopcapture


def RunTest(GPU_ID, Backend, EnableMetalCapture, ShortRun):

    PModel=PropagationModel()

    assert(Backend in ['OpenCL','Metal'])
    if Backend == 'OpenCL':
        COMPUTING_BACKEND=2 # 0 for CPU, 1 for CUDA, 2 for OpenCL, 3 for Metal
    else:
        COMPUTING_BACKEND=3 # 0 for CPU, 1 for CUDA, 2 for OpenCL, 3 for Metal

    #we define domain of simulation and ultrasound transducer 
    Frequency = 350e3  # Hz
    MediumSOS = 1500 # m/s - water
    MediumDensity=1000 # kg/m3

    ShortestWavelength =MediumSOS / Frequency
    SpatialStep =ShortestWavelength / 9 # A minimal step of 6 is recommnded

    DimDomain =  np.array([0.05,0.05,0.10])  # in m, x,y,z

    Amplitude= 100e3 #Pa
    AmplitudeDisplacement=Amplitude/MediumDensity/MediumSOS # We use a 100 kPa source, we just need to convert to particle displacement

    TxDiam = 0.03 # m, circular piston
    TxPlaneLocation = 0.01  # m , in XY plane at Z = 0.01 m

    PMLThickness = 12 # grid points for perect matching layer, 
    ReflectionLimit= 1.0000e-05 #reflection parameter for PML, 

    N1=int(np.ceil(DimDomain[0]/SpatialStep)+2*PMLThickness)
    N2=int(np.ceil(DimDomain[1]/SpatialStep)+2*PMLThickness)
    N3=int(np.ceil(DimDomain[2]/SpatialStep)+2*PMLThickness)
    print('Domain size',N1,N2,N3)
    TimeSimulation=np.sqrt(DimDomain[0]**2+DimDomain[1]**2+DimDomain[2]**2)/MediumSOS #time to cross one corner to another
    
    if ShortRun:
        #we reduce the number of temporal steps, useful when doing GPU capture
        TimeSimulation/=10.
    TemporalStep=1.5e-7 
    StepsForSensor=int((1/Frequency/8)/TemporalStep) # for the sensors, we do not need so much high temporal resolution, so we are keeping 8 time points per perioid

    # ## Material map definition
    # Simple Water type medium

    MaterialMap=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size
    MaterialList=np.zeros((1,5)) # one material in this examples
    MaterialList[0,0]=MediumDensity # water density
    MaterialList[0,1]=MediumSOS # water SoS
    #all other parameters are set to 0 


    # ## Source definition  - stress
    # The source can also be defined for stress. The **source map** is a N1$\times$N2$\times$N3 integer array where anything different from 0 is considered a source of stress ($\sigma_x$, $\sigma_y$, $\sigma_z$). Similar conditions apply for the **Pulse source** array as for the case of particle displacement. When using an stress source, the three stressess are asigned the same value over time. No orientation vector is applied when using an stress source.

    # Acoustic source definitions

    def MakeCircularSource(DimX,DimY,SpatialStep,Diameter):
        #simple defintion of a circular source centred in the domain
        XDim=np.arange(DimX)*SpatialStep
        YDim=np.arange(DimY)*SpatialStep
        XDim-=XDim.mean()
        YDim-=YDim.mean()
        XX,YY=np.meshgrid(XDim,YDim)
        MaskSource=(XX**2+YY**2)<=(Diameter/2.0)**2
        return (MaskSource*1.0).astype(np.uint32)

    SourceMask=MakeCircularSource(N1,N2,SpatialStep,TxDiam)

    SourceMap=np.zeros((N1,N2,N3),np.uint32)
    LocZ=int(np.round(TxPlaneLocation/SpatialStep))+PMLThickness
    SourceMap[:,:,LocZ]=SourceMask 

    Ox=np.zeros((N1,N2,N3))
    Oy=np.zeros((N1,N2,N3))
    Oz=np.zeros((N1,N2,N3))
    Oz[SourceMap>0]=1 #only Z has a value of 1

    LengthSource=4.0/Frequency #we will use 4 pulses
    TimeVectorSource=np.arange(0,LengthSource+TemporalStep,TemporalStep)

    PulseSource = np.sin(2*np.pi*Frequency*TimeVectorSource)

    #note we need expressively to arrange the data in a 2D array
    PulseSource=np.reshape(PulseSource,(1,len(TimeVectorSource))) 
    print(len(TimeVectorSource))


    # ## Sensor map definition

    SensorMap=np.zeros((N1,N2,N3),np.uint32)

    SensorMap[PMLThickness:-PMLThickness,int(N2/2),PMLThickness:-PMLThickness]=1


    # ## Execute simulation


    if EnableMetalCapture:
        StartMetaCapture(GPU_ID)
    

    Sensor,LastMap,DictRMSValue,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                            MaterialMap,
                                                            MaterialList,
                                                            Frequency,
                                                            SourceMap,
                                                            PulseSource,
                                                            SpatialStep,
                                                            TimeSimulation,
                                                            SensorMap,
                                                            Ox=Ox*AmplitudeDisplacement,
                                                            Oy=Oy*AmplitudeDisplacement,
                                                            Oz=Oz*AmplitudeDisplacement,
                                                            NDelta=PMLThickness,
                                                            ReflectionLimit=ReflectionLimit,
                                                            USE_SINGLE=True,
                                                            DT=TemporalStep,
                                                            QfactorCorrection=True,
                                                            SelRMSorPeak=1, #we select  only RMS data
                                                            SelMapsRMSPeakList=['Pressure'],
                                                            SelMapsSensorsList=['Vx','Vy','Vz','Pressure'],
                                                            SensorSubSampling=StepsForSensor,
                                                            TypeSource=0,
                                                            DefaultGPUDeviceName=GPU_ID,
                                                            COMPUTING_BACKEND=COMPUTING_BACKEND)
    if EnableMetalCapture:
        Stopcapture()


if __name__ == "__main__":
    import sys
    import os
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    parser = MyParser(prog='SimpleBenchmark', usage='python %(prog)s.py [options]',description='Run a quick simulation for benchmark Metal vs OpenCL',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('GPU_ID', type=str, nargs='+',help='Substring for GPU id such as "M1"')
    parser.add_argument('Backend', type=str, nargs='+',choices=['Metal','OpenCL'], help='Backend to test, it must be one of the available options shown on the left')

    parser.add_argument('--EnableMetalCapture',  action=argparse.BooleanOptionalAction,
                help='GPU Capture will be done when running Metal backend,\nbe sure of running with environment variable METAL_CAPTURE_ENABLED=1')
    parser.add_argument('--ShortRun',  action=argparse.BooleanOptionalAction,
                help='Enable short simulation (1/10 of normal length), useful when doing Metal capture to reduce size of capture file')
    
    args = parser.parse_args()
    if args.EnableMetalCapture:
        if 'METAL_CAPTURE_ENABLED' not in os.environ:
            raise SystemError("You need to set METAL_CAPTURE_ENABLED=1 prior to run the script when used with--EnableMetalCapture")
        if os.environ['METAL_CAPTURE_ENABLED']!='1':
            raise SystemError("You need to set METAL_CAPTURE_ENABLED=1 prior to run the script when used with--EnableMetalCapture")
    print("Running test with arguments")
    print(args)
    RunTest(args.GPU_ID[0],args.Backend[0],args.EnableMetalCapture,args.ShortRun)