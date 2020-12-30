import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

import pdb

from scipy.optimize import fmin_slsqp

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
                                         Ox,
                                         Oy,
                                         Oz,
                                         SpatialStep,
                                         DurationSimulation,
                                         SensorMap,
                                         AlphaCFL=0.99,
                                         NDelta=12,
                                         ReflectionLimit=1.0000e-05,
                                         IntervalSnapshots=-1,
                                         COMPUTING_BACKEND=1,
                                         USE_SINGLE=True,
                                         USE_SPP = False,
                                         SPP_ZONES=2,
                                         SPP_VolumeFractionFile='',
                                         DT=None,
                                         QfactorCorrection=True,
                                         CheckOnlyParams=False,
                                         TypeSource=0,
                                         SelRMSorPeak=1,
                                         SelMapsRMSPeakList=['ALLV'],
                                         DefaultGPUDeviceName='TITAN'):
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
        if not(np.all(SzMap[0:3]==SourceMap.shape)  and np.all(SzMap[0:3]==SensorMap.shape)
            and np.all(SzMap[0:3]==Ox.shape) and np.all(SzMap[0:3]==Oy.shape) and
            np.all(SzMap[0:3]==Oz.shape)):
            raise ValueError('The size SourceMap, Ox, Oy, Oz, MaterialMap, SensorMap must be equal!!!')


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

        OneOverTauSigma=1.0/TauSigma

        MaterialMap3D=np.zeros((N1+1,N2+1,N3+1),MaterialMap.dtype)
        MaterialMap3D[0:N1,0:N2,0:N3]=MaterialMap
        MaterialMap3D[-1,:,:]=MaterialMap3D[-2,:,:]
        MaterialMap3D[:,-1,:]=MaterialMap3D[:,-2,:]
        MaterialMap3D[:,:,-1]=MaterialMap3D[:,:,-2]

        #we have to put water in the PML to simplify calculation
        #
        #nExtra=8
        nExtra=0
        MaterialMap3D[0:NDelta+nExtra,:,:]=0
        MaterialMap3D[N1-NDelta-nExtra:,:,:]=0
        MaterialMap3D[:,0:NDelta+nExtra,:]=0
        MaterialMap3D[:,N2-NDelta-nExtra:,:]=0
        MaterialMap3D[:,:,0:NDelta+nExtra]=0
        MaterialMap3D[:,:,N3-NDelta-nExtra:]=0

        if DT!=None:
             if DT >dt:
                print  ('Staggered:DT_INVALID The specified manual step is larger than the minimal optimal size, there is a risk of unstable calculation ' + str(DT) + ' ' +str(dt))

             else:
                print ('The specified manual step  is smaller than the minimal optimal size, calculations may take longer than required\n', DT,dt)
                dt=DT


        TimeVector=np.arange(0.0,DurationSimulation,dt)

        if SourceFunctions.shape[0]<np.max(SourceMap.flatten()):
             raise ValueError('The maximum identifier in SourceMap  is larger than the maximum source function (maximum row) in SourceFunctions')

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

        SensorOutput={}

        SensorOutput['time']=TimeVector

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

#define MASK_ALLV				0x0000000001
#define MASK_Vx   			0x0000000002
#define MASK_Vy   			0x0000000004
#define MASK_Vz   			0x0000000008
#define MASK_Sigmaxx    0x0000000010
#define MASK_Sigmayy    0x0000000020
#define MASK_Sigmazz    0x0000000040
#define MASK_Sigmaxy    0x0000000080
#define MASK_Sigmaxz    0x0000000100
#define MASK_Sigmayz    0x0000000200
        #We decode what maps to collect for RMS-Peak
        SelMapsRMSPeak=int(0)
        curIndex=0
        IndexRMSMaps={}
        curMask=int(0x0001)
        #Do not modify the order of this search without matching the low level functions!
        for pMap in ['ALLV','Vx','Vy','Vz','Sigmaxx','Sigmayy','Sigmazz','Sigmaxy','Sigmaxz','Sigmayz']:
            if pMap in  SelMapsRMSPeakList:
                SelMapsRMSPeak=SelMapsRMSPeak | curMask
                IndexRMSMaps[pMap]=curIndex
                curIndex+=1
            else:
                IndexRMSMaps[pMap]=-1
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
        InputParam['TypeSource']=np.uint32(TypeSource)
        InputParam['TimeSteps']=np.uint32(TimeVector.size)
        InputParam['SourceMap']=np.uint32(SourceMap)
        InputParam['SnapshotsPos']=np.uint32(SnapshotsPos)
        InputParam['PMLThickness']=np.uint32(NDelta)
        InputParam['SelRMSorPeak']=np.uint32(SelRMSorPeak)
        InputParam['SelMapsRMSPeak']=np.uint32(SelMapsRMSPeak)
        InputParam['LengthSource']=np.uint32(LengthSource); #%we need now to provided a limit how much the source lasts
        InputParam['DefaultGPUDeviceName']=DefaultGPUDeviceName


        if USE_SPP:
            InputParam['USE_SPP']=np.uint32(SPP_ZONES)
            print('We will use SPP')
            print('read volume fraction from file', SPP_VolumeFractionFile)
            fVol=ReadFromH5py(SPP_VolumeFractionFile)
            SkullFraction=fVol['VolumeFraction']
            SkullRing=fVol['SkullRing']
            print('Calculating SPP matrices')
            MatMap_zone,AllIndexesSparse,AllIndexesLarge,EdgeIndexesSparse,EdgeIndexesLarge,\
                    InternalIndexesSparse, InternalIndexesLarge, InternalIndDic,MultiZoneMaterialMap= PrepareSuperpositionArrays(InputParam['MaterialMap'],SkullFraction,SkullRing,bDisplay=False,SPP_ZONES=SPP_ZONES);

        else:
            InputParam['USE_SPP']=np.uint32(1)
            #this will just create dummy matrices that are required to be passed to the low level function
            MatMap_zone,AllIndexesSparse,AllIndexesLarge,EdgeIndexesSparse,EdgeIndexesLarge,\
                    InternalIndexesSparse, InternalIndexesLarge, InternalIndDic,MultiZoneMaterialMap= PrepareSuperpositionArrays(InputParam['MaterialMap'],None,None,bDisplay=False,USE_SPP=False);

        InputParam['OrigMaterialMap']=MaterialMap3D
        InputParam['MaterialMap']=MultiZoneMaterialMap


        print ('Matrix size= %i x %i x %i , spatial resolution = %g, time steps = %i, temporal step = %g, total sonication length %g ' %(N1,N2,N3,h,TimeVector.size,dt,DurationSimulation))

        SensorOutput_orig,V,RMSValue,Snapshots_orig=self.ExecuteSimulation(InputParam,COMPUTING_BACKEND)# This will be executed either locally or remotely using Pyro4

        for n in range(len(SnapShots)):
            SnapShots[n]['V']=np.squeeze(Snapshots_orig[:,:,n])

        #SensorOutput_orig=np.sqrt(SensorOutput_orig); #Sensors captured the sum of squares of Vx, Vy and Vz
        for n in range(len(IndexSensors)):
            SensorOutput['Vx']=SensorOutput_orig[:,:,0]
            SensorOutput['Vy']=SensorOutput_orig[:,:,1]
            SensorOutput['Vz']=SensorOutput_orig[:,:,2]

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
                    RetValueRMS[key]=np.sqrt(RMSValue[:,:,:,index,0]/len(TimeVector))
        if SelRMSorPeak==2:
            for key,index in IndexRMSMaps.items():
                if index>=0:
                    RetValuePeak[key]=RMSValue[:,:,:,index,0]
        elif SelRMSorPeak==3:
            for key,index in IndexRMSMaps.items():
                if index>=0:
                    RetValuePeak[key]=RMSValue[:,:,:,index,1]
        if 'ALLV' in RetValuePeak:
            #for peak ALLV we collect the sum of squares of Vx, Vy and Vz, so we just need to calculate the sqr rootS
            RetValuePeak['ALLV'] =np.sqrt(RetValuePeak['ALLV'] )

        if  IntervalSnapshots>0:
            if len(RetValueRMS)>0 and len(RetValuePeak)>0:
                return SensorOutput,V,RetValueRMS,RetValuePeak,InputParam,RetSnap
            elif len(RetValueRMS)>0:
                return SensorOutput,V,RetValueRMS,InputParam,RetSnap
            elif len(RetValuePeak)>0:
                return SensorOutput,V,RetValuePeak,InputParam,RetSnap
            else:
                raise SystemError("How we got a condition where no RMS or Peak value was selected")
        else:
            if len(RetValueRMS)>0 and len(RetValuePeak)>0:
                return SensorOutput,V,RetValueRMS,RetValuePeak,InputParam
            elif len(RetValueRMS)>0:
                return SensorOutput,V,RetValueRMS,InputParam
            elif len(RetValuePeak)>0:
                return SensorOutput,V,RetValuePeak,InputParam
            else:
                raise SystemError("How we got a condition where no RMS or Peak value was selected")
    def ExecuteSimulation(self,InputParam,COMPUTING_BACKEND):
        if COMPUTING_BACKEND in [1,2]:
            if COMPUTING_BACKEND==1:
                print( "Performing Simulation wtih GPU CUDA")
                SensorOutput_orig,V,RMSValue,Snapshots_orig=StaggeredFDTD_3D_CUDA(InputParam)
            else:
                print ("Performing Simulation wtih GPU OPENCL")
                SensorOutput_orig,V,RMSValue,Snapshots_orig=StaggeredFDTD_3D_OPENCL(InputParam)
        else:
            print ("Performing Simulation wtih CPU")
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
            dt=AlphaCFL*np.min([np.min(HLongCond),np.min(HShearCond)])
            print ([np.min(HLongCond),np.min(HShearCond)])

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

    LowFreq=CentralFreqHz-CentralFreqHz*0.2 #% we cover a bandwith of +/- 20% the central frequency
    HighFreq=CentralFreqHz+CentralFreqHz*0.2

    LowFreq=LowFreq*2*np.pi
    HighFreq=HighFreq*2*np.pi

    CentralFreq=CentralFreqHz*2*np.pi

    TauSigma=1.0/CentralFreq
    #%the formula is very good to give a initial guess
    Tau=1.0/QValue*(I0l(TauSigma,HighFreq)-I0l(TauSigma,LowFreq))/(I1l(TauSigma,HighFreq)-I1l(TauSigma,LowFreq))
    TauEpsilon = (Tau+1.0)*TauSigma
    #%x0=[TauSigma ,TauEpsilon ];
    x0=Tau
    SpectrumToValidate=np.linspace(LowFreq,HighFreq,num=50).flatten() #%fifty steps should be good

    QOptimal=1.0/QValue*np.ones((SpectrumToValidate.size,1))


    fh =(lambda x:np.sum((EvalQ(x,SpectrumToValidate,TauSigma)-QOptimal)**2))


    x,fx,iuts,imode,smode = fmin_slsqp(fh,x0,bounds=[(0,np.inf)],full_output=True,iprint=0)

    #%TauSigma=x(1);
    #%Tau=x(2);
    Tau=x[0]
    TauEpsilon = (Tau+1)*TauSigma
    Qres=1.0/EvalQ(Tau,SpectrumToValidate,TauSigma)

    Error_LSQ=np.sum((Qres-QValue)**2)/Qres.size

    #%%QValueFormula=QValue-QValue*0.1;% THIS IS TRULY AD HOC, as noted in Blanch, this has be to be done
    #%% to compensate effects of linerarization, but yet,
    #%% lsqlin seems to do a good job, without having to sort out an kitchen formula... the formula is super sensitive to the range of frequencies  to be tested , which is not good at all

    if CentralFreqHz==270e3:
        QValueFormula=QValue-QValue*0.025;#% For 270 KHz, this works well...
    elif CentralFreqHz==836e3:
        QValueFormula=QValue-QValue*0.01;#% For 836 KHz, this works well...
    else:
        QValueFormula=QValue-QValue*0.005;#% For 1402 KHz, this works well...


    TauSigmaFormula=1.0/CentralFreq;
    TauFormula=1.0/QValueFormula*(I0l(TauSigmaFormula,HighFreq)-I0l(TauSigmaFormula,LowFreq))/(I1l(TauSigmaFormula,HighFreq)-I1l(TauSigmaFormula,LowFreq));
    TauEpsilonFormula = (TauFormula+1)*TauSigmaFormula;
    QresFormula=1.0/EvalQ(TauFormula,SpectrumToValidate,TauSigmaFormula);
    Error_Formula=np.sum((QresFormula-QValue)**2)/Qres.size

    return Tau,TauSigma,Qres,SpectrumToValidate,Error_LSQ

def CalculateRelaxationCoefficients(AttMat,Q,Frequency):

    Q=Q*1.2; #%% manual adjustment....
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


def PrepareSuperpositionArrays(SourceMaterialMap,SkullFraction,SkullRing,SPP_ZONES=5,OrderExtra=2,USE_SPP=True,bDisplay=False):
        #if USE_SPP is False, we just create dummy arrays, as these are need to be passed to the low level function for completeness
        ZoneCount=SPP_ZONES
        if USE_SPP:
            SkullRegion=SourceMaterialMap!=0


            MaterialMap=SourceMaterialMap.copy()

            NewMaterialMap=SourceMaterialMap.copy()
            s=SourceMaterialMap.shape
            MultiZoneMaterialMap=np.zeros((s[0],s[1],s[2],ZoneCount),dtype=np.uint32)

            ExpandaMaterial=((SkullRegion)^(SkullFraction>0))
            print (np.sum(ExpandaMaterial))
            ExpandaMaterial=((ExpandaMaterial)&(SkullFraction>0))
            print (np.sum(ExpandaMaterial))
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

            SkullRegion=NewMaterialMap!=0

            SuperpositionMap = np.zeros(SourceMaterialMap.shape,dtype=np.uint8)
            SkullRingFraction=((SkullFraction>0)&(SkullFraction<1.0))
            ExpandedRing=ndimage.binary_dilation(SkullRingFraction,iterations=SPP_ZONES)
            ExpandedRing[SkullFraction==1.0]=True
            SuperpositionMap[ExpandedRing]=1

            ExtraLayers=ndimage.binary_dilation(ExpandedRing,iterations=OrderExtra)
            ExtraLayers=np.logical_xor(ExtraLayers,ExpandedRing)
            SuperpositionMap[ExtraLayers]=2


            SelSuperpoistion=SuperpositionMap>0
            AllIndexesSparse=np.arange(0,np.sum(SelSuperpoistion)).astype(np.uint32)
            AllIndexesLarge=np.where(SelSuperpoistion)
            AllIndexesLargeFlat=np.where(SelSuperpoistion.flatten())[0]
            AllIndexesLarge=np.vstack((AllIndexesLarge[0].astype(np.uint32),AllIndexesLarge[1].astype(np.uint32),AllIndexesLarge[2].astype(np.uint32))).T

            AllMap=SuperpositionMap[SelSuperpoistion].flatten()
            SuperpositionIndexMap=np.zeros(SuperpositionMap.shape,dtype=np.int32)
            SuperpositionIndexMap[SelSuperpoistion]=AllIndexesSparse
            SuperpositionIndexMap[SelSuperpoistion==False]=-1

            SelEdge=SuperpositionMap==2
            EdgeIndexesSparse=np.where(AllMap==2)[0].astype(np.uint32)
            EdgeIndexesLarge=np.where(SelEdge)
            EdgeIndexesLarge=np.vstack((EdgeIndexesLarge[0].astype(np.uint32),EdgeIndexesLarge[1].astype(np.uint32),EdgeIndexesLarge[2].astype(np.uint32))).T

            SelInternal=SuperpositionMap==1
            InternalIndexesLarge=np.where(SelInternal)
            InternalIndexesLarge=np.vstack((InternalIndexesLarge[0].astype(np.uint32),InternalIndexesLarge[1].astype(np.uint32),InternalIndexesLarge[2].astype(np.uint32))).T
            InternalIndexesSparse=np.where(AllMap==1)[0].astype(np.uint32)
            InternalIndDic={}
            for inn in [-2,-1,1,2]:
                sn='InternalIndexesSparse_'
                if inn <0:
                    inl='minus'
                else:
                    inl='plus'
                for sublab in ['i','j','k']:
                    kname=sn+inl+str(abs(inn))+sublab
                    InternalIndDic[kname]=InternalIndexesSparse*0

            InternalIndDic['InternalIndexesSparse_plusij']=InternalIndexesSparse*0
            InternalIndDic['InternalIndexesSparse_plusik']=InternalIndexesSparse*0
            InternalIndDic['InternalIndexesSparse_plusjk']=InternalIndexesSparse*0

            ii,jj,kk=np.where(SelInternal)
            nc=0
            for i,j,k in zip(ii,jj,kk):
                assert(SuperpositionIndexMap[i,j,k]==InternalIndexesSparse[nc])
                for inn in range(-2,3):
                    assert(SuperpositionIndexMap[i+inn,j,k]!=-1)
                    assert(SuperpositionIndexMap[i,j+inn,k]!=-1)
                    assert(SuperpositionIndexMap[i,j,k+inn]!=-1)
                    if inn==1:
                        assert(SuperpositionIndexMap[i+inn,j+inn,k]!=-1)
                        assert(SuperpositionIndexMap[i+inn,j,k+inn]!=-1)
                        assert(SuperpositionIndexMap[i,j+inn,k+inn]!=-1)

                for inn in [-2,-1,1,2]:
                    sn='InternalIndexesSparse_'
                    if inn <0:
                        inl='minus'
                    else:
                        inl='plus'
                    for sublab in ['i','j','k']:
                        kname=sn+inl+str(abs(inn))+sublab
                        if sublab=='i':
                            index=SuperpositionIndexMap[i+inn,j,k]
                        elif sublab=='j':
                            index=SuperpositionIndexMap[i,j+inn,k]
                        else:
                            index=SuperpositionIndexMap[i,j,k+inn]
                        InternalIndDic[kname][nc]=index

                InternalIndDic['InternalIndexesSparse_plusij'][nc]=SuperpositionIndexMap[i+1,j+1,k]
                InternalIndDic['InternalIndexesSparse_plusik'][nc]=SuperpositionIndexMap[i+1,j,k+1]
                InternalIndDic['InternalIndexesSparse_plusjk'][nc]=SuperpositionIndexMap[i,j+1,k+1]

                nc+=1
            MatMap_zone=  np.zeros((ZoneCount,AllIndexesLargeFlat.size),dtype=MaterialMap.dtype)
            SubMat=NewMaterialMap.flatten()[AllIndexesLargeFlat]
            SelFraction=SkullFraction.flatten()[AllIndexesLargeFlat]

            for zone in range(ZoneCount):
                frac=(zone+1)/ZoneCount
                MatMap_zone[zone,SelFraction>=frac]=SubMat[SelFraction>=frac]
                assert(np.sum(SelFraction>=frac)==np.sum(MatMap_zone[zone,:]>0))
                ZoneMaterialMap=NewMaterialMap*0
                ZoneMaterialMap[SkullFraction>=frac]=NewMaterialMap[SkullFraction>=frac]

                MultiZoneMaterialMap[:,:,:,zone]=ZoneMaterialMap


            if bDisplay:
                plt.figure(figsize=(16,8))
                plt.subplot(1,2,1)
                #plt.imshow(SkullRegion[:,:,56],cmap=plt.cm.gray)
                mask = np.ma.masked_where(np.logical_or(SkullRegion,SkullRing)==0, SkullFraction)
                plt.imshow(mask[:,56,:],cmap=plt.cm.inferno)
                plt.colorbar()
                plt.subplot(1,2,2)


                plt.imshow(((SkullRegion)^(SkullFraction>0))[:,56,:],cmap=plt.cm.jet)


                plt.figure(figsize=(18,6))
                plt.subplot(1,3,1)
                mask = np.ma.masked_where(MaterialMap==0, MaterialMap)
                plt.imshow(mask[:,60,:],cmap=plt.cm.inferno)
                plt.xlim(40,60)
                plt.ylim(80,60)
                plt.colorbar()
                plt.subplot(1,3,2)

                mask = np.ma.masked_where(NewMaterialMap==0, NewMaterialMap)
                plt.imshow(mask[:,60,:],cmap=plt.cm.inferno)
                plt.xlim(40,60)
                plt.ylim(80,60)
                plt.colorbar()

                plt.subplot(1,3,3)
                mask = np.ma.masked_where(SkullFraction==0.0, SkullFraction)
                plt.imshow(mask[:,60,:],cmap=plt.cm.inferno)
                plt.xlim(40,60)
                plt.ylim(80,60)
                plt.colorbar()

                plt.figure(figsize=(16,8))
                plt.subplot(1,2,1)
                plt.imshow(SuperpositionMap[:,:,56])
                plt.colorbar()
                plt.subplot(1,2,2)
                plt.imshow(ExtraLayers[:,:,56])

        else:
            s=SourceMaterialMap.shape
            MultiZoneMaterialMap=np.zeros((s[0],s[1],s[2],1),dtype=np.uint32)
            MultiZoneMaterialMap[:,:,:,0]=SourceMaterialMap

            MatMap_zone=np.zeros((1,1),dtype=np.uint32)
            AllIndexesSparse=np.zeros((1),dtype=np.uint32)
            AllIndexesLarge=np.zeros((1,3),dtype=np.uint32)
            EdgeIndexesSparse=np.zeros((1),dtype=np.uint32)
            EdgeIndexesLarge=np.zeros((1,3),dtype=np.uint32)
            InternalIndexesSparse=np.zeros((1),dtype=np.uint32)
            InternalIndexesLarge=np.zeros((1,3),dtype=np.uint32)
            InternalIndDic={}
            for inn in [-2,-1,1,2]:
                    sn='InternalIndexesSparse_'
                    if inn <0:
                        inl='minus'
                    else:
                        inl='plus'
                    for sublab in ['i','j','k']:
                        kname=sn+inl+str(abs(inn))+sublab
                        InternalIndDic[kname]=np.zeros((1),dtype=np.uint32)

            InternalIndDic['InternalIndexesSparse_plusij']=np.zeros((1),dtype=np.uint32)
            InternalIndDic['InternalIndexesSparse_plusik']=np.zeros((1),dtype=np.uint32)
            InternalIndDic['InternalIndexesSparse_plusjk']=np.zeros((1),dtype=np.uint32)


        return MatMap_zone,AllIndexesSparse,AllIndexesLarge,EdgeIndexesSparse,EdgeIndexesLarge,\
                InternalIndexesSparse, InternalIndexesLarge, InternalIndDic,MultiZoneMaterialMap



#######################
#%%%%%%%%%%%%%%%%%%%% OLD EXPERIMENTAL STUFF, kept just for potential future use
ENABLE_EXPERIMENTAL = False
if ENABLE_EXPERIMENTAL:
    def CleanUpIsolatedBoneVoxelsOverLine(MaterialMap):
    #this will clean up lonely bone voxels (lost in the middle of water or having only one face in contact to other bone voxel)
        NewMaterialMap=MaterialMap.copy()
        BoneAround=np.zeros(NewMaterialMap.shape)
        TotalAccum=0

        nTarget=2
        while(True):
            IsBone=NewMaterialMap!=0
            BoneAround[:]=0

            for n in range(1,nTarget+1):
                BoneAround[n:,:,:][NewMaterialMap[:-n,:,:]!=0]+=1
                BoneAround[:-n,:,:][NewMaterialMap[n:,:,:]!=0]+=1
            ToRemove=(BoneAround<nTarget)&(IsBone)
            NewMaterialMap[ToRemove]=0 #lonely voxel
            #print "Total lonely voxels eliminated following I = ", (ToRemove).sum()
            accum=ToRemove.sum()

            IsBone=NewMaterialMap!=0
            BoneAround[:]=0
            for n in range(1,nTarget+1):
                BoneAround[:,n:,:][NewMaterialMap[:,:-n,:]!=0]+=1
                BoneAround[:,:-n,:][NewMaterialMap[:,n:,:]!=0]+=1
            ToRemove=(BoneAround<nTarget)&(IsBone)
            NewMaterialMap[ToRemove]=0 #lonely voxel
            #print "Total lonely voxels eliminated following J = ", (ToRemove).sum()
            accum+=ToRemove.sum()

            IsBone=NewMaterialMap!=0
            BoneAround[:]=0
            IsBone=NewMaterialMap!=0
            BoneAround[:]=0
            for n in range(1,nTarget+1):
                BoneAround[:,:,n:][NewMaterialMap[:,:,:-n]!=0]+=1
                BoneAround[:,:,:-n][NewMaterialMap[:,:,n:]!=0]+=1
            ToRemove=(BoneAround<nTarget)&(IsBone)
            NewMaterialMap[ToRemove]=0 #lonely voxel
            #print "Total lonely voxels eliminated following K = ", (ToRemove).sum()
            accum+=ToRemove.sum()

            TotalAccum+=accum

            if accum==0:
                break

        #print "Total lonely voxels eliminated = ", TotalAccum,  ' from an original total of ', (MaterialMap!=0).sum()
        N1h=NewMaterialMap.shape[0]/2
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow( MaterialMap[N1h,:,:]);
        plt.subplot(1,3,2)
        plt.imshow( NewMaterialMap[N1h,:,:]);
        plt.subplot(1,3,3)
        plt.imshow( (NewMaterialMap[N1h,:,:]!=0)^(MaterialMap[N1h,:,:]!=0));

        N1h=NewMaterialMap.shape[2]/2
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow( MaterialMap[:,:,N1h]);
        plt.subplot(1,3,2)
        plt.imshow( NewMaterialMap[:,:,N1h]);
        plt.subplot(1,3,3)
        plt.imshow( (NewMaterialMap[:,:,N1h]!=0)^(MaterialMap[:,:,N1h]!=0));

        plt.show()
        return NewMaterialMap


    def CleanUpSingleBoneVoxels(MaterialMap):
    #this will clean up lonely bone voxels (lost in the middle of water or having only one face in contact to other bone voxel)
        NewMaterialMap=MaterialMap.copy()

        accum=0
        while(True):
            BoneAround=np.zeros(NewMaterialMap.shape)

            IsBone=NewMaterialMap!=0;

            BoneAround[1:,:,:][NewMaterialMap[:-1,:,:]!=0]+=1
            BoneAround[:-1,:,:][NewMaterialMap[1:,:,:]!=0]+=1

            BoneAround[:,1:,:][NewMaterialMap[:,:-1,:]!=0]+=1
            BoneAround[:,:-1,:][NewMaterialMap[:,1:,:]!=0]+=1

            BoneAround[:,:,1:][NewMaterialMap[:,:,:-1]!=0]+=1
            BoneAround[:,:,:-1][NewMaterialMap[:,:,1:]!=0]+=1

            NewMaterialMap[(BoneAround==0)&(IsBone)]=0 #lonely voxel
            NewMaterialMap[(BoneAround==1)&(IsBone)]=0 #voxel with only one pal... sorry man

            thisRound=((BoneAround==0)&(IsBone)).sum()+((BoneAround==1)&(IsBone)).sum()

            accum+=thisRound

            #print "Total lonely voxels eliminated = ", thisRound,  ' from an original total of ', (MaterialMap!=0).sum(), 'with remaining ', (NewMaterialMap!=0).sum()

            if thisRound==0:
                break

        #print "Total lonely voxels eliminated = ", accum,  ' from an original total of ', (MaterialMap!=0).sum()

        N1h=NewMaterialMap.shape[0]/2
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow( MaterialMap[N1h,:,:]);
        plt.subplot(1,3,2)
        plt.imshow( NewMaterialMap[N1h,:,:]);
        plt.subplot(1,3,3)
        plt.imshow( (NewMaterialMap[N1h,:,:]!=0)^(MaterialMap[N1h,:,:]!=0));

        N1h=NewMaterialMap.shape[2]/2
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow( MaterialMap[:,:,N1h]);
        plt.subplot(1,3,2)
        plt.imshow( NewMaterialMap[:,:,N1h]);
        plt.subplot(1,3,3)
        plt.imshow( (NewMaterialMap[:,:,N1h]!=0)^(MaterialMap[:,:,N1h]!=0));

        plt.show()

        return NewMaterialMap

    def ObtainSkullSurfaceAndRing(MaterialMap,bDisplay=False):
        from mayavi import mlab
        from tvtk.api import tvtk

        SkullRegion=MaterialMap!=0
        SkullRing=np.logical_xor(ndimage.morphology.binary_dilation(SkullRegion),ndimage.morphology.binary_erosion(SkullRegion))

        data=SkullRegion.copy()
        data=ndimage.morphology.binary_dilation(data,iterations=4)*1.0
        for n in range(6):
            data=ndimage.gaussian_filter(data,0.2)

        src = mlab.pipeline.scalar_field(data)
        src.spacing = [1, 1, 1]
        src.update_image_data = True
        src.origin=[1.1,1,1]


        srcOrig = mlab.pipeline.scalar_field((SkullRing)*1.0)
        srcOrig.spacing = [1, 1, 1]
        srcOrig.update_image_data = True


        median_filter = tvtk.ImageMedian3D()
        try:
            median_filter.set_kernel_size(2, 2, 2)
        except AttributeError:
            median_filter.kernel_size = [2, 2, 2]

        median = mlab.pipeline.user_defined(src, filter=median_filter)

        diffuse_filter = tvtk.ImageAnisotropicDiffusion3D(
                                            diffusion_factor=0.5,
                                            diffusion_threshold=1,
                                            number_of_iterations=1)

        diffuse = mlab.pipeline.user_defined(median, filter=diffuse_filter)

        contour = mlab.pipeline.contour(median, )

        contour.filter.contours = [1, ]

        dec = mlab.pipeline.decimate_pro(contour)
        dec.filter.feature_angle = 90.
        dec.filter.target_reduction = 0.6

        smooth_ = tvtk.SmoothPolyDataFilter(
                            number_of_iterations=100,
                            relaxation_factor=0.1,
                            feature_angle=90,
                            feature_edge_smoothing=False,
                            boundary_smoothing=False,
                            convergence=0.,
                        )
        smooth = mlab.pipeline.user_defined(dec, filter=smooth_)

        # Get the largest connected region
        connect_ = tvtk.PolyDataConnectivityFilter(extraction_mode=4)
        connect = mlab.pipeline.user_defined(smooth, filter=connect_)

        # Compute normals for shading the surface
        compute_normals = mlab.pipeline.poly_data_normals(connect)
        compute_normals.filter.feature_angle = 80

        #origin of scalarfield in mayavi is (1,1,1), so we better adjust to have it at 0,0,0



        if bDisplay:
            print ("showing image")
            fig = mlab.figure(bgcolor=(0, 0, 0), size=(400, 500))

            # to speed things up
            fig.scene.disable_render = True

            surf = mlab.pipeline.surface(compute_normals,
                                                    color=(0.3, 0.72, 0.62),opacity=0.8)

            #----------------------------------------------------------------------
            # Display a cut plane of the raw data
            ipw = mlab.pipeline.image_plane_widget(srcOrig, colormap='bone',
                            plane_orientation='z_axes',
                            slice_index=55)

            mlab.view(-165, 32, 350, [143, 133, 73])
            mlab.roll(180)

            fig.scene.disable_render = False

            #----------------------------------------------------------------------
            # To make the link between the Mayavi pipeline and the much more
            # complex VTK pipeline, we display both:
            mlab.show_pipeline(rich_view=False)
            from tvtk.pipeline.browser import PipelineBrowser
            browser = PipelineBrowser(fig.scene)
            browser.show()
            mlab.show()

        result=compute_normals.get_output_dataset()
        normals= np.array(result.point_data.normals)
        faces=result.polys.data.to_array().reshape((result.polys.number_of_cells,4))[:,1:4]
        points=np.array(result.points)
        points[:,0]-=1
        points[:,1]-=1
        points[:,2]-=1
        return SkullRing, points, faces,normals,result,compute_normals
