import numpy as np
from BabelViscoFDTD.H5pySimple import ReadFromH5py,SaveToH5py
from BabelViscoFDTD.PropagationModel import PropagationModel
PModel=PropagationModel()

Frequency = 350e3  # Hz
MediumSOS = 1500 # m/s - water
MediumDensity=1000 # kg/m3

ShortestWavelength =MediumSOS / Frequency
SpatialStep =ShortestWavelength / 30.0 # A minimal step of 6 is recommnded

DimDomain =  np.array([0.05,0.05,0.05])  # in m, x,y,z

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
TemporalStep=3.4e-8 # if this step is too coarse a warning will be generated (but simulation will continue,) 
StepsForSensor=int((1/Frequency/8)/TemporalStep) # for the sensors, we do not need so much high temporal resolution, so we are keeping 8 time points per perioid

def MakeCircularSource(DimX,DimY,SpatialStep,Diameter):
    #simple defintion of a circular source centred in the domain
    XDim=np.arange(DimX)*SpatialStep
    YDim=np.arange(DimY)*SpatialStep
    XDim-=XDim.mean()
    YDim-=YDim.mean()
    XX,YY=np.meshgrid(XDim,YDim)
    MaskSource=(XX**2+YY**2)<=(Diameter/2.0)**2
    return (MaskSource*1.0).astype(np.uint32)

MaterialMap=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size
MaterialList=np.zeros((1,5)) # one material in this examples
MaterialList[0,0]=MediumDensity # water density
MaterialList[0,1]=MediumSOS # water SoS
#all other parameters are set to 0 


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

SensorMap=np.zeros((N1,N2,N3),np.uint32)

SensorMap[PMLThickness:-PMLThickness,int(N2/2),PMLThickness:-PMLThickness]=1

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
                                                         COMPUTING_BACKEND=3,
                                                         USE_SINGLE=True,
                                                         DT=TemporalStep,
                                                         QfactorCorrection=True,
                                                         SelRMSorPeak=2, #we select  only RMS data
                                                         SelMapsRMSPeakList=['Pressure'],
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Pressure'],
                                                         SensorSubSampling=StepsForSensor,
                                                         DefaultGPUDeviceName='W6800',
                                                         TypeSource=0)