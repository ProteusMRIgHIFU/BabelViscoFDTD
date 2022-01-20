import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import warnings

import pdb

from scipy.optimize import fmin_slsqp
from scipy.interpolate import interp1d

import logging
LOGGER_NAME = 'FDTDStaggered'
logger = logging.getLogger(LOGGER_NAME)

from .StaggeredFDTD_3D_With_Relaxation import StaggeredFDTD_3D
try:
    from .StaggeredFDTD_3D_With_Relaxation_CUDA import StaggeredFDTD_3D_CUDA
    print ("StaggeredFDTD_3D_CUDA loaded")
except:
    print ("StaggeredFDTD_3D_CUDA NOT loaded")
try:
    from .StaggeredFDTD_3D_With_Relaxation_OPENCL import StaggeredFDTD_3D_OPENCL
    print ("StaggeredFDTD_3D_OPENCL loaded")
except:
     print ("StaggeredFDTD_3D_OPENCL NOT loaded")
try:
    from .StaggeredFDTD_3D_With_Relaxation_METAL import StaggeredFDTD_3D_METAL
    print ("StaggeredFDTD_3D_METAL loaded")
except:
    print ("StaggeredFDTD_3D_METAL NOT loaded")

#############################################################################################################
#This global dictionary specifies the order in the MaterialProperties array (Nx5) where N is the numbe of materials
StaggeredConstants={}
StaggeredConstants['ColDensity']=0
StaggeredConstants['ColLongSOS']=1
StaggeredConstants['ColShearSOS']=2
StaggeredConstants['ColLongAtt']=3
StaggeredConstants['ColShearAtt']=4

class PropagationModel:
    def StaggeredFDTD_3D_with_relaxation(self,MaterialMap,
                                         MaterialProperties,
                                         Frequency,
                                         SourceMap,
                                         SourceFunctions,
                                         SpatialStep,
                                         DurationSimulation,
                                         SensorMap,
                                         Ox=np.array([1]),
                                         Oy=np.array([1]),
                                         Oz=np.array([1]),
                                         AlphaCFL=0.99,
                                         NDelta=12,
                                         ReflectionLimit=1.0000e-05,
                                         IntervalSnapshots=-1,
                                         COMPUTING_BACKEND=1,
                                         USE_SINGLE=True,
                                         SPP_ZONES=1,
                                         SPP_VolumeFraction=None,
                                         DT=None,
                                         QfactorCorrection=True,
                                         CheckOnlyParams=False,
                                         TypeSource=0,
                                         SelRMSorPeak=1,
                                         SelMapsRMSPeakList=['ALLV'],
                                         SelMapsSensorsList=['Vx','Vy','Vz'],
                                         SensorSubSampling=2,
                                         SensorStart=0,
                                         DefaultGPUDeviceName='TITAN',
                                         DefaultGPUDeviceNumber=0,
                                         SILENT=0,
                                         ManualGroupSize=np.array([-1,-1,-1]).astype(np.int32),
                                         ManualLocalSize=np.array([-1,-1,-1]).astype(np.int32)):
        '''
        Samuel Pichardo, Ph.D.
        2020

        Implementation of the 3D Sttagered-grid  Virieux scheme for the elastodynamic equation
        to calculate wave propagation in a heterogenous medium with liquid and solids
        (Virieux, Jean. "P-SV wave propagation in heterogeneous media; velocity-stress
        finite-difference method." Geophysics 51.4 (1986): 889-901.))
        The staggered grid method has the huge advantage of modeling correctly
        the propagation from liquid to solid and solid to liquid
        without the need of complicated coupled systems between acoustic and
        elastodynamic equations.

        We assume isotropic conditions (c11=c22, c12= Lamba Lame coeff, c66= Miu Lame coeff), with
        Cartesian square coordinate system.

        WE USE SI convention! meters, seconds, meters/s, Pa,kg/m^3, etc, for units.
        Version 4.0. We use now the rotated staggered grid to better model liquid-solids interfaces
        Saenger, Erik H., Norbert Gold, and Serge A. Shapiro.
        "Modeling the propagation of elastic waves using a modified finite-difference grid."
        Wave motion 31.1 (2000): 77-92.

        Version 3.0. We add true viscosity handling to account attenuation losses, following:
        Blanch, Joakim O., Johan OA Robertsson, and William W. Symes. "Modeling of a constant Q: Methodology and algorithm for an efficient and optimally inexpensive viscoelastic technique." Geophysics 60.1 (1995): 176-184.
        and
        Bohlen, Thomas. "Parallel 3-D viscoelastic finite difference seismic modelling." Computers & Geosciences 28.8 (2002): 887-899.

        version 2.0 April 2013.
        "One code to rule them all, one code to find them, one code to bring them all and in the darkness bind them". Sorry , I couln't avoid it :)
        The low-level function is based on a single code that produce  Python and Matlab modules, single or double precision, and to run either X64 or CUDA code. This will help a lot to propagate any important change in all
        implementations


        version 1.0 March 2013.
        The Virieux method is relatively easy to implement, the tricky point was
        to implement correctly, with the lowest memory footprint possible,
        the Perfect Matching  Layer (PML) method. I used the method of Collino and
        Tsogka "Application of the perfectly matched absorbing layer model to
        the linear elastodynamic problem in anisotropic
        heterogenous media" Geophysics, 66(1) pp 294-307. 2001.
        Both methods (Vireux + Collino&Tsogka) are quite referenced and validated.

        Use
        [Sensor,LastMap,SnapShots]=StaggeredFDTD_3D(MaterialMap,MaterialProperties,
        SourceMap,SourceFunctions,SpatialStep,DurationSimulation,SensorMap)

        where
        'MaterialMap' is an uint32 N1xN2xN3 3D matrix where each different
        material is labeled. Start labeling from '0'
        'MaterialProperties' is a Nx3 matrix where N is the maximum number of
        materials in MaterialMap. Since the labeling should starts in 0, N must be max(MaterialMap(:))+1
        Column 1 in MaterialProperties refers to the density of the material,
        Column 2 is the compressional speed of sound
        Column 3 is the shear speed of sound

        'SourceMap' is a uint32 N1xN2 matrix where the presence of source is
        indicate. Any value different to 0 is considered as a source. The value
        indicate which source function must be refered.

        'SourceFunctions' is a N-sources x NumberTimeSteps matrix
        where N-sources is the maximun value of SourceMap. Each row in
        SourceFuntions is the evolution on time of the stress (xx,yy) (negative
        version of the pressure)


        You can specifu optional parameters 'IntervalSnapshots', 'ReflectionLimit' (for
        PML), 'NDelta. (also for  PML), 'AlphaCFL' (for the Courant convergence criteria)
        For the default values for the PML, reead below in the code and be
        cautious when modifying them, since your simulation mat get a lot of
        dispersion or reflections.

        '''


        SzMap=MaterialMap.shape
        if TypeSource<2:
            if not(np.all(SzMap[0:3]==SourceMap.shape)  and np.all(SzMap[0:3]==SensorMap.shape)
                and np.all(SzMap[0:3]==Ox.shape) and np.all(SzMap[0:3]==Oy.shape) and
                np.all(SzMap[0:3]==Oz.shape)):
                raise ValueError('The size SourceMap, Ox, Oy, Oz, MaterialMap, SensorMap must be equal!!!')
        else:
            if not(np.all(SzMap[0:3]==SourceMap.shape)  and np.all(SzMap[0:3]==SensorMap.shape)):
                raise ValueError('The size SourceMap, MaterialMap, SensorMap must be equal!!!')
            if Ox.ndim==1:
                if not(Ox.size==1 and Oy.size==1  and Oz.size==1 and Ox[0]==1 and Oy[0]==1 and Oz[0]==1):
                    raise ValueError('When specifying a source for stress, Oy and Oz must remain equal to [1], and Ox can be either [1] or same dimensions as SourceMap')
                Ox=np.ones(MaterialMap.shape)
            else:
                if not( np.all(SzMap[0:3]==Ox.shape) and Oy.size==1  and Oz.size==1 and Oy[0]==1 and Oz[0]==1):
                    raise ValueError('When specifying a source for stress, Oy and Oz must remain equal to [1], and Ox can be either [1] or same dimensions as SourceMap')



        N1=SzMap[0]
        N2=SzMap[1]
        N3=SzMap[2]

        h=SpatialStep

        VMaxLong=np.max(MaterialProperties[:,np.int32(StaggeredConstants['ColLongSOS'])])

        ###%%%%%%


        dt,RhoMat,MiuMat, LambdaMiuMat, LambdaMat,TauLong,TauShear,TauSigma,AnalysisQFactorLong,AnalysisQFactorShear=self.CalculateMatricesForPropagation(MaterialMap,MaterialProperties,Frequency,QfactorCorrection,h,AlphaCFL)

        Omega=Frequency*2*np.pi

        PoisonRatio=LambdaMat.flatten()/(2.0*(LambdaMat.flatten()+MiuMat.flatten()))

        bValidPoisonRatio=True
        if np.max(PoisonRatio)>0.5:
            bValidPoisonRatio=False
            if CheckOnlyParams==False:
                raise ValueError('Poison ratio larger than 0.5!!!! are you sure of the values of speed of sound and density??')
        if np.min(PoisonRatio)<-1.0:
            bValidPoisonRatio=False
            if CheckOnlyParams==False:
                raise ValueError('Poison ratio smaller than -1!!!! are you sure of the values of speed of sound and density??')

        if dt < 0.0:
            raise ValueError('Invalid dt conditions!!! dt =' + str(dt))
        if CheckOnlyParams:
            return bValidPoisonRatio, dt

        if SPP_ZONES>1 and SPP_VolumeFraction is None:
            raise ValueError('SPP_VolumeFraction must contain a (N1,N2,N3) matrix with the fraction of solid material')
        if SPP_ZONES>1:
            if np.all(SzMap[0:3]==SPP_VolumeFraction.shape)  is None:
                raise ValueError('SPP_VolumeFraction must contain a matrix with the fraction of solid material of same dimensions as material map')
        if SPP_ZONES>1 and (np.any(SPP_VolumeFraction<0) or np.any(SPP_VolumeFraction)>1.0):
            raise ValueError('SPP_VolumeFraction must contain values between 0.0 and 1.0 ')

        OneOverTauSigma=1.0/TauSigma

        MaterialMap3D=np.zeros((N1+1,N2+1,N3+1),MaterialMap.dtype)
        MaterialMap3D[0:N1,0:N2,0:N3]=MaterialMap
        MaterialMap3D[-1,:,:]=MaterialMap3D[-2,:,:]
        MaterialMap3D[:,-1,:]=MaterialMap3D[:,-2,:]
        MaterialMap3D[:,:,-1]=MaterialMap3D[:,:,-2]

        #TODO: Dec 30, 2020: We have some issues with the PML if it is a solid....
        MaterialMap3D[0:NDelta,:,:]=0
        MaterialMap3D[N1-NDelta:,:,:]=0
        MaterialMap3D[:,0:NDelta,:]=0
        MaterialMap3D[:,N2-NDelta:,:]=0
        MaterialMap3D[:,:,0:NDelta]=0
        MaterialMap3D[:,:,N3-NDelta:]=0

        if DT!=None:
             if DT >dt:
                import warnings
                raise ValueError('Staggered:DT_INVALID The specified manual step is larger than the minimal optimal size, there is a risk of unstable calculation ' + str(DT) + ' ' +str(dt))

             else:
                print ('The specified manual step  is smaller than the minimal optimal size, calculations may take longer than required\n', DT,dt)
                dt=DT


        TimeVector=np.arange(0.0,DurationSimulation,dt)

        if SourceFunctions.shape[0]<np.max(SourceMap.flatten()):
             raise ValueError('The maximum identifier in SourceMap  is larger than the maximum source function (maximum row) in SourceFunctions')
        
        NumberSensorSubSampling=np.floor(TimeVector.size/SensorSubSampling)
        if (SensorStart <0) or (SensorStart>=NumberSensorSubSampling):
            raise ValueError('Invalid SensorStart value, it must be larger or equal to 0, and less than the size of the sensor length %i ' %(NumberSensorSubSampling))


        LengthSource = SourceFunctions.shape[1]

        delta=(NDelta-2.0)*h;# %% We do a trick to force the PML to act in the regions where there are calculations.
        delta=(NDelta)*h;# %% We do a trick to force the PML to act in the regions where there are calculations.

        Dx = np.zeros((NDelta+1,1))
        Dxhp = np.zeros((NDelta+1,1))

        d0 = np.log(1.0/ReflectionLimit)*3*VMaxLong/2/delta
        #%d0 = 80*8.63e-2*max(VLong)/delta;
        ddx=(d0*(np.arange(1,NDelta+1)*h/delta)**2)
        ddxhp=(d0*((np.arange(1,NDelta+1)-0.5)*h/delta)**2)

        Dx[0:NDelta,0]=np.flipud(ddx).flatten()
        Dxhp[0:NDelta,0]=np.flipud(ddxhp).flatten()

        InvDXDTplus=1.0/(1.0/dt + Dx/2)
        DXDTminus=(1.0/dt - Dx/2)
        InvDXDTplushp=1.0/(1.0/dt + Dxhp/2)
        DXDTminushp=(1.0/dt - Dxhp/2)
        #%%%%%%%%%%%%%%%% Perfect Matched Layer


        LambdaMiuMatOverH=LambdaMiuMat /h
        LambdaMatOverH=LambdaMat /h
        MiuMatOverH=MiuMat/h

        InvRhoMatH =1.0/RhoMat/h

        #%% the sensors are easy, just pass the indexes that need to be observed
        IndexSensors=np.nonzero(np.transpose(SensorMap).flatten()>0)[0]+1 #KEEP the +1, since in the low level function the index is substracted

        SnapshotsPos=[]
        SnapShots=[]
        if IntervalSnapshots>0:
            tPlot=1
            for n in range(TimeVector.size):
                if np.floor(TimeVector[n]/IntervalSnapshots)==tPlot:
                    SnapshotsPos.append(n)
                    SnapShots.append({'time':TimeVector[n]})
                    tPlot+=+1
        InputParam={}

        #We decode what maps to collect for RMS-Peak
        SelMapsRMSPeak=int(0)
        SelMapsSensors=int(0)
        curIndexRMS=0
        curIndexSensors=0
        IndexRMSMaps={}
        IndexSensorMaps={}
        curMask=int(0x0001)
        #Do not modify the order of this search without matching the low level functions!
        for pMap in ['ALLV','Vx','Vy','Vz','Sigmaxx','Sigmayy','Sigmazz','Sigmaxy','Sigmaxz','Sigmayz','Pressure']:
            if pMap in  SelMapsRMSPeakList:
                SelMapsRMSPeak=SelMapsRMSPeak | curMask
                IndexRMSMaps[pMap]=curIndexRMS
                curIndexRMS+=1
            else:
                IndexRMSMaps[pMap]=-1

            if pMap in  SelMapsSensorsList:
                SelMapsSensors=SelMapsSensors | curMask
                IndexSensorMaps[pMap]=curIndexSensors
                curIndexSensors+=1
            else:
                IndexSensorMaps[pMap]=-1
            curMask=curMask<<1


        if USE_SINGLE==False:
            InputParam['InvDXDTplus']=InvDXDTplus;
            InputParam['DXDTminus']=DXDTminus;
            InputParam['InvDXDTplushp']=InvDXDTplushp;
            InputParam['DXDTminushp']=DXDTminushp;
            InputParam['LambdaMiuMatOverH']=LambdaMiuMatOverH;
            InputParam['LambdaMatOverH']=LambdaMatOverH;
            InputParam['MiuMatOverH']=MiuMatOverH;
            InputParam['TauLong']=TauLong;
            InputParam['TauShear']=TauShear;
            InputParam['OneOverTauSigma']=OneOverTauSigma;
            InputParam['InvRhoMatH']=InvRhoMatH;
            InputParam['SourceFunctions']=SourceFunctions;
            InputParam['DT']=dt;
            InputParam['Ox']=Ox;
            InputParam['Oy']=Oy;
            InputParam['Oz']=Oz;
        else:
            InputParam['InvDXDTplus']=InvDXDTplus.astype(np.float32)
            InputParam['DXDTminus']=DXDTminus.astype(np.float32)
            InputParam['InvDXDTplushp']=InvDXDTplushp.astype(np.float32)
            InputParam['DXDTminushp']=DXDTminushp.astype(np.float32)
            InputParam['LambdaMiuMatOverH']=LambdaMiuMatOverH.astype(np.float32)
            InputParam['LambdaMatOverH']=LambdaMatOverH.astype(np.float32)
            InputParam['MiuMatOverH']=MiuMatOverH.astype(np.float32)
            InputParam['TauLong']=TauLong.astype(np.float32)
            InputParam['TauShear']=TauShear.astype(np.float32)
            InputParam['OneOverTauSigma']=OneOverTauSigma.astype(np.float32)
            InputParam['InvRhoMatH']=InvRhoMatH.astype(np.float32)
            InputParam['SourceFunctions']=SourceFunctions.astype(np.float32)
            InputParam['DT']=np.float32(dt);
            InputParam['Ox']=Ox.astype(np.float32)
            InputParam['Oy']=Oy.astype(np.float32)
            InputParam['Oz']=Oz.astype(np.float32)

        InputParam['MaterialMap']=MaterialMap3D;
        InputParam['IndexSensorMap']=np.uint32(IndexSensors)
        InputParam['N1']=np.uint32(N1)
        InputParam['N2']=np.uint32(N2)
        InputParam['N3']=np.uint32(N3)
        InputParam['SILENT']=np.uint32(SILENT)
        InputParam['TypeSource']=np.uint32(TypeSource)
        InputParam['TimeSteps']=np.uint32(TimeVector.size)
        InputParam['SourceMap']=np.uint32(SourceMap)
        InputParam['SnapshotsPos']=np.uint32(SnapshotsPos)
        InputParam['PMLThickness']=np.uint32(NDelta)
        InputParam['SelRMSorPeak']=np.uint32(SelRMSorPeak)
        InputParam['SelMapsRMSPeak']=np.uint32(SelMapsRMSPeak)
        InputParam['SensorSubSampling']=np.uint32(SensorSubSampling)
        InputParam['SensorStart']=np.uint32(SensorStart)
        InputParam['SelMapsSensors']=np.uint32(SelMapsSensors)
        InputParam['LengthSource']=np.uint32(LengthSource); #%we need now to provided a limit how much the source lasts
        InputParam['DefaultGPUDeviceName']=DefaultGPUDeviceName
        InputParam['DefaultGPUDeviceNumber']=np.uint32(DefaultGPUDeviceNumber)
        InputParam['ManualGroupSize']=ManualGroupSize
        InputParam['ManualLocalSize']=ManualLocalSize

        SolidFraction=None
        if SPP_ZONES>1:
            print('We will use SPP')
        else:
            InputParam['SPP_ZONES']=np.uint32(1)

        if SPP_VolumeFraction is not None:
            SolidFraction=np.zeros((N1+1,N2+1,N3+1))
            SolidFraction[:N1,:N2,:N3]=SPP_VolumeFraction
            SolidFraction[0:NDelta,:,:]=0.
            SolidFraction[N1-NDelta:,:,:]=0.
            SolidFraction[:,0:NDelta,:]=0.
            SolidFraction[:,N2-NDelta:,:]=0.
            SolidFraction[:,:,0:NDelta]=0.
            SolidFraction[:,:,N3-NDelta:]=0.
            
        InputParam['SPP_ZONES']=np.uint32(SPP_ZONES)
        MultiZoneMaterialMap= PrepareSuperpositionArrays(MaterialMap3D,SolidFraction,SPP_ZONES=SPP_ZONES);
        InputParam['OrigMaterialMap']=MaterialMap3D
        InputParam['MaterialMap']=MultiZoneMaterialMap
        InputParam['SolidFraction']=SolidFraction


        print ('Matrix size= %i x %i x %i , spatial resolution = %g, time steps = %i, temporal step = %g, total sonication length %g ' %(N1,N2,N3,h,TimeVector.size,dt,DurationSimulation))

        SensorOutput_orig,V,RMSValue,Snapshots_orig=self.ExecuteSimulation(InputParam,COMPUTING_BACKEND)# This will be executed either locally or remotely using Pyro4

        for n in range(len(SnapShots)):
            SnapShots[n]['V']=np.squeeze(Snapshots_orig[:,:,n])

        #SensorOutput_orig=np.sqrt(SensorOutput_orig); #Sensors captured the sum of squares of Vx, Vy and Vz

        RetValueSensors={}
        RetValueSensors['time']=TimeVector[SensorStart*SensorSubSampling:len(TimeVector):SensorSubSampling]
        for key,index in IndexSensorMaps.items():
            if index>=0:
                RetValueSensors[key]=SensorOutput_orig[:,0:len(RetValueSensors['time']),index]
        if 'ALLV' in RetValueSensors:
            #for sensor ALLV we collect the sum of squares of Vx, Vy and Vz, so we just need to calculate the sqr rootS
            RetValueSensors['ALLV'] =np.sqrt(RetValueSensors['ALLV'] )

        if (IntervalSnapshots>0):
            RetSnap=SnapShots
        else:
            RetSnap=[]

        #now time to organize this a dictionary
        RetValueRMS={}
        RetValuePeak={}

        if SelRMSorPeak==1 or SelRMSorPeak==3:
            for key,index in IndexRMSMaps.items():
                if index>=0:
                    #%in RMSValue we have the sum of square values over time, we need a
                    #%final calculation to have the real RMS
                    RetValueRMS[key]=np.sqrt(RMSValue[:,:,:,index,0]/(len(TimeVector)-SensorStart*SensorSubSampling))
        if SelRMSorPeak==2:
            for key,index in IndexRMSMaps.items():
                if index>=0:
                    RetValuePeak[key]=RMSValue[:,:,:,index,0]
        elif SelRMSorPeak==3:
            for key,index in IndexRMSMaps.items():
                if index>=0:
                    RetValuePeak[key]=RMSValue[:,:,:,index,1]

        pFactor=MaterialProperties[MaterialMap,0]*MaterialProperties[MaterialMap,1]**2/SpatialStep
        if 'Pressure' in RetValuePeak:
            #What was calculated in the low level function was the stencil gradient of each Vx, Vy, Vz 
            # We need to scale back to rho c^2 /Spatialstep
            RetValuePeak['Pressure']*=pFactor
            
        if 'Pressure' in RetValueRMS:
            RetValueRMS['Pressure']*= pFactor

        if 'Pressure' in RetValueSensors:
            ii,jj,kk=np.unravel_index(IndexSensors-1, SensorMap.shape, order='F')
            RetValueSensors['Pressure']*=np.repeat(pFactor[ii,jj,kk].reshape([RetValueSensors['Pressure'].shape[0],1]),
                                                  RetValueSensors['Pressure'].shape[1],axis=1)
        
        if 'ALLV' in RetValuePeak:
            #for peak ALLV we collect the sum of squares of Vx, Vy and Vz, so we just need to calculate the sqr rootS
            RetValuePeak['ALLV'] =np.sqrt(RetValuePeak['ALLV'] )

        if  IntervalSnapshots>0:
            if len(RetValueRMS)>0 and len(RetValuePeak)>0:
                return RetValueSensors,V,RetValueRMS,RetValuePeak,InputParam,RetSnap
            elif len(RetValueRMS)>0:
                return RetValueSensors,V,RetValueRMS,InputParam,RetSnap
            elif len(RetValuePeak)>0:
                return RetValueSensors,V,RetValuePeak,InputParam,RetSnap
            else:
                raise SystemError("How we got a condition where no RMS or Peak value was selected")
        else:
            if len(RetValueRMS)>0 and len(RetValuePeak)>0:
                return RetValueSensors,V,RetValueRMS,RetValuePeak,InputParam
            elif len(RetValueRMS)>0:
                return RetValueSensors,V,RetValueRMS,InputParam
            elif len(RetValuePeak)>0:
                return RetValueSensors,V,RetValuePeak,InputParam
            else:
                raise SystemError("How we got a condition where no RMS or Peak value was selected")



    def ExecuteSimulation(self,InputParam,COMPUTING_BACKEND):
        if COMPUTING_BACKEND == 1:
            print( "Performing Simulation with GPU CUDA")
            SensorOutput_orig,V,RMSValue,Snapshots_orig=StaggeredFDTD_3D_CUDA(InputParam)
        elif COMPUTING_BACKEND == 2:
            print ("Performing Simulation with GPU OPENCL")
            SensorOutput_orig,V,RMSValue,Snapshots_orig=StaggeredFDTD_3D_OPENCL(InputParam)
        elif COMPUTING_BACKEND == 3:
            print ("Performing Simulation with GPU METAL")
            SensorOutput_orig,V,RMSValue,Snapshots_orig=StaggeredFDTD_3D_METAL(InputParam)
        else:
            print ("Performing Simulation with CPU")
            SensorOutput_orig,V,RMSValue,Snapshots_orig=StaggeredFDTD_3D(InputParam)
        return SensorOutput_orig,V,RMSValue,Snapshots_orig

    def CalculateMatricesForPropagation(self,MaterialMap, MaterialProperties, Frequency,QfactorCorrection,h,AlphaCFL):

        rho=MaterialProperties[:,np.int32(StaggeredConstants['ColDensity'])].flatten()
        VLong=MaterialProperties[:,np.int32(StaggeredConstants['ColLongSOS'])].flatten()
        VShear=MaterialProperties[:,np.int32(StaggeredConstants['ColShearSOS'])].flatten()
        ALong=MaterialProperties[:,np.int32(StaggeredConstants['ColLongAtt'])].flatten()
        AShear=MaterialProperties[:,np.int32(StaggeredConstants['ColShearAtt'])].flatten()

        UniqueMaterial=np.unique(MaterialMap.flatten())

        if np.max(UniqueMaterial)+1 > MaterialProperties.shape[0]:
            raise ValueError('The map in MaterialMap must have as many different values as materials identified in MaterialProperties (number of rows)');


        VShearUnique=VShear[UniqueMaterial]
        VLongUnique=VLong[UniqueMaterial]
        RhoUnique=rho[UniqueMaterial]
        ALongUnique=ALong[UniqueMaterial]
        AShearUnique=AShear[UniqueMaterial]

        dt,RhoVec,MiuVec, LambdaMiuVec, LambdaVec,TauLongVec,TauShearVec,TauSigmaVec,AnalysisQFactorLong,AnalysisQFactorShear=self.CalculateLambdaMiuMatrices(VLongUnique,VShearUnique,RhoUnique,ALongUnique,AShearUnique,Frequency,QfactorCorrection,h,AlphaCFL)

        RhoMat=np.zeros(MaterialProperties.shape[0])
        RhoMat[UniqueMaterial]=RhoVec

        MiuMat=np.zeros(MaterialProperties.shape[0])
        MiuMat[UniqueMaterial]=MiuVec

        LambdaMiuMat=np.zeros(MaterialProperties.shape[0])
        LambdaMiuMat[UniqueMaterial]=LambdaMiuVec

        LambdaMat=np.zeros(MaterialProperties.shape[0])
        LambdaMat[UniqueMaterial]=LambdaVec

        TauLongMat=np.zeros(MaterialProperties.shape[0])
        TauLongMat[UniqueMaterial]=TauLongVec

        TauShearMat=np.zeros(MaterialProperties.shape[0])
        TauShearMat[UniqueMaterial]=TauShearVec

        TauSigmaMat=np.zeros(MaterialProperties.shape[0])
        TauSigmaMat[UniqueMaterial]=TauSigmaVec

        return dt,RhoMat,MiuMat, LambdaMiuMat, LambdaMat,TauLongMat,TauShearMat,TauSigmaMat,AnalysisQFactorLong,AnalysisQFactorShear

    def CalculateLambdaMiuMatrices(self,VLongInput,VShearInput,RhoMat,ALongInput,AShearInput,Frequency,QfactorCorrection,h,AlphaCFL,CheckOnlyParameters=False):
        Omega=Frequency*2*np.pi
        VMaxLong=np.max(VLongInput)


        #%% here comes the fun, calculate the relaxation coefficients,
        #%% first, we detected where attenuation is zero to avoid problems
        AttLongNonZero=ALongInput!=0.0
        AttShearNonZero=AShearInput!=0.0
        #%factor Qs, and Qp, is given by the number of wavelenghts required to
        #%attenuate the amplitude by exp(-pi), meaning alpha *Qdistance = pi;
        #% Blanch, Joakim O., Johan OA Robertsson, and William W. Symes. "Modeling of a constant Q: Methodology and algorithm for an efficient and optimally inexpensive viscoelastic technique." Geophysics 60.1 (1995): 176-184.

        #% We calculate Qs, Ql for a single relation mechanism, that is ok for
        #% single frequency relaxation :
        #%Bohlen, Thomas. "Parallel 3-D viscoelastic finite difference seismic modelling." Computers & Geosciences 28.8 (2002): 887-899.

        SubALong=ALongInput[AttLongNonZero].copy().flatten()
        SubAShear=AShearInput[AttShearNonZero].copy().flatten()

        DistQLong=np.pi/SubALong
        DistQShear=np.pi/SubAShear

        QLong=DistQLong.copy()
        if QLong.shape[0]!=0:
            QLong=QLong/(VLongInput[AttLongNonZero]/Frequency)

        QShear=DistQShear.copy()
        if QShear.shape[0]!=0:
            QShear=QShear/(VShearInput[AttShearNonZero]/Frequency)

        NoAttenuation=False
        if QShear.shape[0]==0 and QLong.shape[0]==0:
             NoAttenuation=True


        TauSigma=np.ones(ALongInput.size)/Omega


        #%We save the results of the curves of the Q factor, because if we need
        #%to analyze later that Q makes sense, this will be done postiori

        AnalysisQFactorLong,TauLong,TauSigma_l=CalculateRelaxationCoefficients(ALongInput,QLong,Frequency);
        AnalysisQFactorShear,TauShear,TauSigma_s=CalculateRelaxationCoefficients(AShearInput,QShear,Frequency);


        if QfactorCorrection: # %dispersion correction...
           Q_cw_factor_long= np.real(np.sqrt(1.0/(1.0 + (1j*Omega*TauSigma_l*TauLong)  /(1.0+1j*Omega*TauSigma_l))))
           Q_cw_factor_shear=np.real(np.sqrt(1.0/(1.0 + (1j*Omega*TauSigma_s*TauShear) /(1.0+1j*Omega*TauSigma_s))))


           print ("VLongInput,VShearInput", np.unique(VLongInput), np.unique(VShearInput))
           print ("Q_cw_factor_long,Q_cw_factor_shear", np.unique(Q_cw_factor_long), np.unique(Q_cw_factor_shear))
           VLongMat=VLongInput*Q_cw_factor_long
           VShearMat=VShearInput*Q_cw_factor_shear
           print ("VLongMat,VShearMat", np.unique(VLongMat), np.unique(VShearMat))
        else:
           VLongMat=VLongInput.copy()
           VShearMat=VShearInput.copy()

        MiuMat = VShearMat**2*RhoMat
        LambdaMiuMat =  VLongMat**2*RhoMat
        LambdaMat = LambdaMiuMat - 2*MiuMat

        if CheckOnlyParameters:
            PoisonRatio=LambdaMat.flatten()/(2.0*(LambdaMat.flatten()+MiuMat.flatten()))
            return PoisonRatio


        # %%verify the time step condition,
        #%after
        #%Sun, Chengyu, Yunfei Xiao, Xingyao Yin, and Hongchao Peng. "Stability condition of finite difference solution for viscoelastic wave equations." Earthquake Science 22, no. 5 (2009): 479-485
        #%%I thinks there is a typo in the paper that stipulates
        #%% 4*h^4./(3*vp2_Long*SumAbsWeights) as part of the calculations of dt,
        #% but letting it like that would make no sense for cases where Q-->inf
        #% , in that situation, dt must converge as in other papers to *h*6/7/sqrt(3)/VMax
        #% after reviewing the paper equations, it is clear there is a typo,
        #% the previous equation (no mumber in the paper) to (17) is correct and
        #% and should translates to 4*h^2./(3*vp2_Long*SumAbsWeights)
        if NoAttenuation==False:
            WeighCoeff=np.array([ -1.7857143e-3, 2.5396825e-2,  -0.2,  1.6 ,-2.8472222 ,1.6,  -0.2,  2.5396825e-2, -1.7857143e-3])# %these are the coefficients for order 4-th
            SumAbsWeights=np.sum(np.abs(WeighCoeff))

            vp2_Long=(np.sqrt(QLong**2+1)+QLong)*VLongMat[AttLongNonZero]**2/(2.0*QLong)
            HLongCond=np.sqrt(Omega**2*h**4/(9*QLong**2.*vp2_Long**2 * SumAbsWeights**2) + 4*h**2/(3*vp2_Long*SumAbsWeights)) -\
                Omega*h**2/(3*QLong*vp2_Long*SumAbsWeights)

            vp2_Shear=(np.sqrt(QShear**2+1)+QShear)*VShearMat[AttShearNonZero]**2/(2*QShear)
            HShearCond=np.sqrt(Omega**2*h**4/(9*QShear**2*vp2_Shear**2 * SumAbsWeights**2) + 4*h**2/(3*vp2_Shear*SumAbsWeights)) -\
                Omega*h**2/(3*QShear*vp2_Shear*SumAbsWeights)


            #%% dt using the approach from Sun is slightly smaller, but it truly does the job, before, it was getting quickly unstable results,
            #%% making smaller dt manually helped, but now we have a better tuned approach
            dt=AlphaCFL*np.min(np.hstack((HLongCond.flatten(),HShearCond.flatten())))
            print (np.min(np.hstack((HLongCond.flatten(),HShearCond.flatten()))))

        else:
            dt=AlphaCFL*h*6.0/7.0/np.sqrt(3.0)/VMaxLong #%after: Bohlen, Thomas. "Parallel 3-D viscoelastic finite difference seismic modelling." Computers & Geosciences 28.8 (2002): 887-899.

        print ("dt,VLongMat,VShearMat,TauLong,TauShear,TauSigma,VLongInput,VShearInput", dt, np.unique(VLongMat),  np.unique(VShearMat),np.unique(TauLong),np.unique(TauShear),np.unique(TauSigma),np.unique(VLongInput),np.unique(VShearInput))
        return dt,RhoMat,MiuMat, LambdaMiuMat, LambdaMat,TauLong,TauShear,TauSigma,AnalysisQFactorLong,AnalysisQFactorShear

    ###########################################################

########### AUXILIARY functions #######################

def EvalQ(x0,w,Tau_sigma):
    Tau=x0
    F=w*Tau_sigma*Tau/(1.0+w**2*Tau_sigma**2*(1.0+Tau))
    return F

def EvalQp(x0,w):
    Tau=x0[0]
    Tau_sigma=x0[1]
    F=w*Tau_sigma*Tau/(1.0+w**2*Tau_sigma**2*(1.0+Tau))
    return F

def I0l(TauSigma,w):
    F=1.0/2/TauSigma*np.log(1+w**2*TauSigma**2)
    return F


def I1l(TauSigma,w):
    F=1.0/2/TauSigma*(np.arctan(w*TauSigma)-w*TauSigma/(1+w**2*TauSigma**2))
    return F

def EvalTau(x0,w):
    LowFreq=w[0]
    HighFreq=w[1]
    TauSigma=x0;
    F=(I0l(TauSigma,HighFreq)-I0l(TauSigma,LowFreq))/(I1l(TauSigma,HighFreq)-I1l(TauSigma,LowFreq))
    return F

def OptimalTauForQFactor(QValue,CentralFreqHz):
#%%we implement at lin. sqr. root minimization of the Quality factor for the
#%%viscoelastic function with one relaxation mechanism
#%% Blanch, Joakim O., Johan OA Robertsson, and William W. Symes. "Modeling of a constant Q: Methodology and algorithm for an efficient and optimally inexpensive viscoelastic technique." Geophysics 60.1 (1995): 176-184.
#%% and
#%% Bohlen, Thomas. "Parallel 3-D viscoelastic finite difference seismic modelling." Computers & Geosciences 28.8 (2002): 887-899.

    LowFreq=CentralFreqHz-CentralFreqHz*0.2 #% we cover a bandwith of +/- 30% the central frequency
    HighFreq=CentralFreqHz+CentralFreqHz*0.2

    LowFreq=LowFreq*2*np.pi
    HighFreq=HighFreq*2*np.pi

    CentralFreq=CentralFreqHz*2*np.pi

    TauSigma=1.0/CentralFreq
    #%the formula is very good to give a initial guess
    #Tau=1.0/QValue*(I0l(TauSigma,HighFreq)-I0l(TauSigma,LowFreq))/(I1l(TauSigma,HighFreq)-I1l(TauSigma,LowFreq))
    Tau=2.0/QValue
    TauEpsilon = (Tau+1.0)*TauSigma
    #%x0=[TauSigma ,TauEpsilon ];
    x0=Tau
    SpectrumToValidate=np.linspace(LowFreq,HighFreq,num=100).flatten() #%fifty steps should be good

    QOptimal=1.0/QValue*np.ones((SpectrumToValidate.size,1))


    fh =(lambda x:np.sum((EvalQ(x,SpectrumToValidate,TauSigma)-QOptimal)**2))
    fhp =(lambda x:np.sum((EvalQp(x,SpectrumToValidate)-QOptimal)**2))


    x,fx,iuts,imode,smode = fmin_slsqp(fh,x0,bounds=[(0,np.inf)],full_output=True,iprint=0)

    xp,fxp,iutsp,imodep,smodep = fmin_slsqp(fhp,[Tau,TauSigma],bounds=[(0,1),(0,1)],full_output=True,iprint=0)

    print('2.0/QValue, x0, x, xp, TauSigma',2.0/QValue,x0,x,xp,TauSigma)
    
    Tau=x[0]
    #Tau=xp[0]
    #Tau=2.0/QValue
    TauEpsilon = (Tau+1)*TauSigma
    Qres=1.0/EvalQ(Tau,SpectrumToValidate,TauSigma)

    Error_LSQ=np.sum((Qres-QValue)**2)/Qres.size

    #%%QValueFormula=QValue-QValue*0.1;% THIS IS TRULY AD HOC, as noted in Blanch, this has be to be done
    #%% to compensate effects of linerarization, but yet,
    #%% lsqlin seems to do a good job, without having to sort out an kitchen formula... the formula is super sensitive to the range of frequencies  to be tested , which is not good at all

    fCal=[270e3,836e3,1402e3]
    Adj=[0.025,0.01,0.005]

    if CentralFreqHz<fCal[0] or CentralFreqHz>fCal[-1] :
        # warnings.warn('Central frequency (kHz) %f  outside the range of tested frequencies for adjustment of attenuation [%f,%f]' %\
        #                 (CentralFreqHz/1e3,fCal[0],fCal[-1]))
        if CentralFreqHz<fCal[0]:
            QValueFormula=QValue-QValue*Adj[0];
        else:
            QValueFormula=QValue-QValue**Adj[-1]
    else:
        intF = interp1d(fCal, Adj)
        QAdj=intF(CentralFreqHz)
        QValueFormula=QValue-QValue*QAdj
    


    TauSigmaFormula=1.0/CentralFreq;
    TauFormula=1.0/QValueFormula*(I0l(TauSigmaFormula,HighFreq)-I0l(TauSigmaFormula,LowFreq))/(I1l(TauSigmaFormula,HighFreq)-I1l(TauSigmaFormula,LowFreq));
    TauEpsilonFormula = (TauFormula+1)*TauSigmaFormula;
    QresFormula=1.0/EvalQ(TauFormula,SpectrumToValidate,TauSigmaFormula);
    Error_Formula=np.sum((QresFormula-QValue)**2)/Qres.size

    print('TauFormula, TauSigmaFormula',TauFormula, TauSigmaFormula)

    return Tau,TauSigma,Qres,SpectrumToValidate,Error_LSQ
    
def CalculateRelaxationCoefficients(AttMat,Q,Frequency):

    #Q*=1.2; #%% manual adjustment....
    AttNonZero=AttMat!=0
    IndAttNonZero=np.nonzero(AttNonZero.flatten().T)[0]
    AnalysisQFactor={}
    AnalysisQFactor['Spectrum']=None
    AnalysisQFactor['Attenuation']=AttMat[IndAttNonZero]
    AnalysisQFactor['Qres']=[]
    AnalysisQFactor['Qdesired']=Q
    AnalysisQFactor['Error_LSQ']=np.zeros((IndAttNonZero.size,1))
    TempTau=np.zeros(IndAttNonZero.size)
    TauSigma=np.zeros(IndAttNonZero.size)
    QresTemp=[]
    Error_LSQTemp=np.zeros(IndAttNonZero.size)
    SpectrumToValidate=None

    for n in range(IndAttNonZero.size):
        Tau,Ts,Qres,SpectrumToValidateTemp,Error_LSQ=OptimalTauForQFactor(Q[n],Frequency)
        #%[Tau,Ts,Qres,SpectrumToValidateTemp,Error_LSQ]=OptimalTauAndTaueEpsForQFactor(Q(n),Frequency);
        if n==0:
            SpectrumToValidate=SpectrumToValidateTemp.copy()

        TempTau[n]=Tau
        TauSigma[n]=Ts
        QresTemp.append(Qres)
        Error_LSQTemp[n]=Error_LSQ

    Tau=np.zeros(AttMat.size)
    TauSigma_l=Tau.copy()
    Tau[IndAttNonZero]=TempTau
    TauSigma_l[IndAttNonZero]=TauSigma;
    AnalysisQFactor['Spectrum']=SpectrumToValidate;    #%the spectrum is the same for every value of Q
    AnalysisQFactor['Error_LSQ']=Error_LSQTemp;
    AnalysisQFactor['Qres']=QresTemp
    return AnalysisQFactor,Tau,TauSigma_l


def PrepareSuperpositionArrays(SourceMaterialMap,SolidFraction,SPP_ZONES=1,OrderExtra=2,bRemoveIsolatedRegions=True):
        #This function will assign the id of material in the fraction elements that are the closest to the water solid interfaces
        #if SPP_ZONES==1 is False, we just create dummy arrays, as these are need to be passed to the low level function for completeness
        #bRemoveIsolatedRegions controls if disconected regions will need to be removed as this can occur if SolidFraction was
        #calculated with no perfectly interfaces crossing the voxels (or small issues with whatever CSG method to caclulate intersection)

        ZoneCount=SPP_ZONES
        if ZoneCount>1 or SolidFraction is not None:
            SolidRegion=SourceMaterialMap!=0
            MaterialMap=SourceMaterialMap.copy()

            NewMaterialMap=SourceMaterialMap.copy()
            s=SourceMaterialMap.shape
            MultiZoneMaterialMap=np.zeros((s[0],s[1],s[2],ZoneCount),dtype=np.uint32)

            ExpandaMaterial=((SolidRegion)^(SolidFraction>0))
            ExpandaMaterial=((ExpandaMaterial)&(SolidFraction>0))
            ii,jj,kk=np.where(ExpandaMaterial)

            mgrid = np.lib.index_tricks.nd_grid()
            iS,jS,kS=mgrid[-1:2,-1:2,-1:2]
            # print(ii.shape)
            for i,j,k in zip(ii,jj,kk):
                ssi=i+iS
                ssj=j+jS
                ssk=k+kS

                SubMat=MaterialMap[ssi,ssj,ssk]
                assert(np.all(SubMat==0)==False)
                assert(np.any(SubMat==0))
                sel=SubMat!=0
                ssi=iS[sel]
                ssj=jS[sel]
                ssk=kS[sel]
                QuadDistance=np.linalg.norm(np.vstack((ssi,ssj,ssk)),axis=0)
                mIn=np.argmin(QuadDistance)
                assert(QuadDistance[mIn]!=0.0)
                SubMat=SubMat[sel]
                NewMaterialMap[i,j,k]=SubMat[mIn]

            SolidRegion=NewMaterialMap!=0

            SuperpositionMap = np.zeros(SourceMaterialMap.shape,dtype=np.uint8)
            SkullRingFraction=((SolidFraction>0)&(SolidFraction<=1.0))
            ExpandedRing=ndimage.binary_dilation(SkullRingFraction,iterations=SPP_ZONES)
            ExpandedRing[SolidFraction==1.0]=True
            SuperpositionMap[ExpandedRing]=1

            ExtraLayers=ndimage.binary_dilation(ExpandedRing,iterations=OrderExtra)
            ExtraLayers=np.logical_xor(ExtraLayers,ExpandedRing)
            SuperpositionMap[ExtraLayers]=2


            SelSuperpoistion=SuperpositionMap>0
            AllIndexesLarge=np.where(SelSuperpoistion)
            AllIndexesLargeFlat=np.where(SelSuperpoistion.flatten())[0]
            AllIndexesLarge=np.vstack((AllIndexesLarge[0].astype(np.uint32),AllIndexesLarge[1].astype(np.uint32),AllIndexesLarge[2].astype(np.uint32))).T

            MatMap_zone=  np.zeros((ZoneCount,AllIndexesLargeFlat.size),dtype=MaterialMap.dtype)
            SubMat=NewMaterialMap.flatten()[AllIndexesLargeFlat]
            SelFraction=SolidFraction.flatten()[AllIndexesLargeFlat]

            for zone in range(ZoneCount):
                frac=(zone)/ZoneCount
                MatMap_zone[zone,SelFraction>frac]=SubMat[SelFraction>frac]
                assert(np.sum(SelFraction>frac)==np.sum(MatMap_zone[zone,:]>0))
                ZoneMaterialMap=NewMaterialMap*0
                ZoneMaterialMap[SolidFraction>frac]=NewMaterialMap[SolidFraction>frac]

                MultiZoneMaterialMap[:,:,:,zone]=ZoneMaterialMap

                if bRemoveIsolatedRegions:
                    lab,num_features =ndimage.label(MultiZoneMaterialMap[:,:,:,zone])
                    if num_features>1: #only if we have more than 1 big block
                        LabSize=[]
                        for k in range(1,num_features+1):
                            LabSize.append([(lab==k).sum(),k])
                        LabSize.sort() #we sort by size, the largest skull region is at the end
                        LabSize=np.array(LabSize)
                        Reg=MultiZoneMaterialMap[:,:,:,zone]
                        Reg[lab!=LabSize[-1,1]]=0 #we made background all small regions
                        print('Removing %f %% of isolated voxels' % (LabSize[:-1,0].sum()/LabSize[-1,0]*100) )
                        MultiZoneMaterialMap[:,:,:,zone]=Reg
                    else:
                        print('No isolated voxels were found')

        else:
            s=SourceMaterialMap.shape
            MultiZoneMaterialMap=np.zeros((s[0],s[1],s[2],1),dtype=np.uint32)
            MultiZoneMaterialMap[:,:,:,0]=SourceMaterialMap
        return MultiZoneMaterialMap
