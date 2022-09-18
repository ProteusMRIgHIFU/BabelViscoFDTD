//Samuel Pichardo, Sunnybrook Research Institute.
//2013
//MEX (Matlab C function) and Python C NUMPY extension (ey, one code to rule them all!!, meaning this will go straight ahead to
// Proteus) to implement the 3D Sttagered-grid  Virieux scheme for the elastodynamic equation
// to calculate wave propagation in a heterogenous medium with liquid and solids
// (Virieux, Jean. "P-SV wave propagation in heterogeneous media; velocity-stress finite-difference method." Geophysics 51.4 (1986): 889-901.))
// The staggered grid method has the huge advantage of modeling correctly the propagation from liquid to solid and solid to liquid
// without the need of complicated coupled systems between acoustic and elastodynamic equations.
//
//  We assume isotropic conditions (c11=c22, c12= Lamba Lame coeff, c66= Miu Lame coeff), with
// Cartesian square coordinate system.
//
// WE USE SI convention! meters, seconds, meters/s, Pa,kg/m^3, etc, for units.
//
// Version 3.0. We add true viscosity handling to account attenuation losses, following:
// Blanch, Joakim O., Johan OA Robertsson, and William W. Symes. "Modeling of a constant Q: Methodology and algorithm for an efficient and optimally inexpensive viscoelastic technique." Geophysics 60.1 (1995): 176-184.
// and
// Bohlen, Thomas. "Parallel 3-D viscoelastic finite difference seismic modelling." Computers & Geosciences 28.8 (2002): 887-899.
//
// version 2.0 April 2013.
// "One code to rule them all, one code to find them, one code to bring them all and in the darkness bind them". Sorry , I couln't avoid it :)
// We have now one single code to produce  Python and Matlab modules, single or double precision, and to run either X64 or CUDA code. This will help a lot to propagate any important change in all
//implementations
//
// version 1.0 March 2013.
// The Virieux method is relatively easy to implement, the tricky point was to implement correctly, with the lowest memory footprint possible, the Perfect Matching
//Layer method. I used the method of Collino and Tsogka "Application of the perfectly matched absorbing layer model to the the linear elastodynamic problem in anisotropic
//heterogenous media" Geophysics, 66(1) pp 294-307. 2001.
// Both methods (Vireux + Collino&Tsogka) are quite referenced and validated.
//
// USE THIS FUNCTION only with the Matlab function StaggeredFDTD_3D.m . Refer to that function to review how parameters are passed to this Mex function.
//The function receives one input structure parameter with all the info required for the simulation. It returns 3 outputs:
// 1 - The evolution of the Stress on X direction over time on locations designated by the user
// 2 - The last value of maps of Vx, Vy, Sigma_xx, Sigma_yy and Sigma_xy. This will be useful for future developments where we may use these maps as initial states in later simulations
// 3 - Snapshot of the particle velocity map at given time points


#include <Python.h>
#include <ndarrayobject.h>


#include <math.h>
#include <string.h>
#include <stdlib.h>
#if defined(USE_OPENMP)
#include <omp.h>
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#if defined(__MACH__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif


#define INHOST(_VarName) _VarName


unsigned int	INHOST(SILENT);

#include "commonDef.h"


#ifdef MATLAB_MEX
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
#else
static PyObject *mexFunction(PyObject *self, PyObject *args)
#endif
{
	unsigned int			   INHOST(N1),INHOST(N2),INHOST(N3),INHOST(LengthSource),INHOST(TypeSource),
								INHOST(TimeSteps),NumberSnapshots,INHOST(NumberSources),TimeStepsSource,
								INHOST(NumberSensors),INHOST(PML_Thickness), INHOST(SelRMSorPeak),
								INHOST(SelMapsRMSPeak),
								INHOST(SelMapsSensors),
								INHOST(SensorSubSampling),
								INHOST(SensorStart),
								INHOST(NumberSelRMSPeakMaps),
								INHOST(NumberSelSensorMaps),
								INHOST(IndexRMSPeak_Vx)=0,
								INHOST(IndexRMSPeak_Vy)=0,
								INHOST(IndexRMSPeak_Vz)=0,
								INHOST(IndexRMSPeak_Sigmaxx)=0,
								INHOST(IndexRMSPeak_Sigmayy)=0,
								INHOST(IndexRMSPeak_Sigmazz)=0,
								INHOST(IndexRMSPeak_Sigmaxy)=0,
								INHOST(IndexRMSPeak_Sigmaxz)=0,
								INHOST(IndexRMSPeak_Sigmayz)=0,
								INHOST(IndexRMSPeak_Pressure)=0,
								INHOST(IndexSensor_Vx)=0,
								INHOST(IndexSensor_Vy)=0,
								INHOST(IndexSensor_Vz)=0,
								INHOST(IndexSensor_Sigmaxx)=0,
								INHOST(IndexSensor_Sigmayy)=0,
								INHOST(IndexSensor_Sigmazz)=0,
								INHOST(IndexSensor_Sigmaxy)=0,
								INHOST(IndexSensor_Sigmaxz)=0,
								INHOST(IndexSensor_Sigmayz)=0,
								INHOST(IndexSensor_Pressure)=0,
								INHOST(SelK),
								INHOST(DefaultGPUDeviceNumber);
    mexType INHOST(DT);



// #ifdef WIN32
// 	typedef unsigned long DWORD;
// 	DWORD		dummyID;
// #endif
	//---------------------------------------------------------------//
	// check input parameters
	//---------------------------------------------------------------//

	#ifdef MATLAB_MEX
	  /* check for proper number of arguments */
    if (nrhs!=1)
		mexErrMsgTxt("Incorrect number of input arguments \n Syntax: [SensorOutput,LastVMap,VMax,Snapshots]=FDTDStaggered(InputParam);\n");
    if (nlhs!=4)
		mexErrMsgTxt("Incorrect number of output arguments \n Syntax: [SensorOutput,LastVMap,VMax,Snapshots]=FDTDStaggered(InputParam);\n");
	#else
	PyDictObject *py_argDict;

	if(!PyArg_ParseTuple(args, "O!", &PyDict_Type, &py_argDict))
	{
		PyErr_SetString(PyExc_Exception,"Argument was not a python dictionary... goodbye (1).\n");
		return NULL;
	}

	if(!PyDict_Check(py_argDict))
	{
		PyErr_SetString(PyExc_Exception, "Argument was not a python dictionary... goodbye (2).\n");
		return NULL;
	}
	#endif
	GET_FIELD(SILENT);
	INHOST(SILENT)=*GET_DATA_UINT32_PR(SILENT);
	PRINTF("Running with new interface conditions\n");

	//We use Macros to get pointers to the Matlab/Python objects in the input structure/dictionary
    //#pragma message(VAR_NAME_VALUE(GET_FIELD(InvDXDTplus)))
	GET_FIELD(InvDXDTplus);
	GET_FIELD(DXDTminus);
	GET_FIELD(InvDXDTplushp);
	GET_FIELD(DXDTminushp);
	GET_FIELD(LambdaMiuMatOverH);
	GET_FIELD(LambdaMatOverH);
	GET_FIELD(MiuMatOverH);
	GET_FIELD(TauLong);
	GET_FIELD(OneOverTauSigma);
	GET_FIELD(TauShear);
	GET_FIELD(InvRhoMatH);
	GET_FIELD(IndexSensorMap);
	GET_FIELD(N1);
	GET_FIELD(N2);
	GET_FIELD(N3);
	GET_FIELD(LengthSource);
	GET_FIELD(TimeSteps);
	GET_FIELD(SourceFunctions);
	GET_FIELD(SourceMap);
	GET_FIELD(SnapshotsPos);
	GET_FIELD(DT);
	GET_FIELD(MaterialMap);
	GET_FIELD(PMLThickness);
    GET_FIELD(TypeSource);
	GET_FIELD_GENERIC(DefaultGPUDeviceName);
	GET_FIELD(Ox);
	GET_FIELD(Oy);
	GET_FIELD(Oz);
	GET_FIELD(SelRMSorPeak);
	GET_FIELD(SelMapsRMSPeak);
	GET_FIELD(SelMapsSensors);
	GET_FIELD(SensorSubSampling);
	GET_FIELD(SensorStart);
	GET_FIELD(DefaultGPUDeviceNumber);
	GET_FIELD(ManualGroupSize);
	GET_FIELD(ManualLocalSize);
	GET_FIELD_GENERIC(PI_OCL_PATH);
	GET_FIELD_GENERIC(kernelfile);
	GET_FIELD_GENERIC(kernbinfile);

	GET_FIELD(SPP_ZONES);

	unsigned int INHOST(ZoneCount);

	//These macros validate that the datatype of each object is the one expected
    VALIDATE_FIELD_MEX_TYPE(InvDXDTplus);
	VALIDATE_FIELD_MEX_TYPE(DXDTminus);
	VALIDATE_FIELD_MEX_TYPE(InvDXDTplushp);
	VALIDATE_FIELD_MEX_TYPE(DXDTminushp);
	VALIDATE_FIELD_MEX_TYPE(LambdaMiuMatOverH);
	VALIDATE_FIELD_MEX_TYPE(LambdaMatOverH);
	VALIDATE_FIELD_MEX_TYPE(MiuMatOverH);
	VALIDATE_FIELD_MEX_TYPE(TauLong);
	VALIDATE_FIELD_MEX_TYPE(OneOverTauSigma);
	VALIDATE_FIELD_MEX_TYPE(TauShear);
	VALIDATE_FIELD_MEX_TYPE(InvRhoMatH);
	VALIDATE_FIELD_UINT32(IndexSensorMap);
	VALIDATE_FIELD_MEX_TYPE(DT);
	VALIDATE_FIELD_UINT32(MaterialMap);
	VALIDATE_FIELD_UINT32(PMLThickness);
	VALIDATE_FIELD_UINT32(N1);
	VALIDATE_FIELD_UINT32(N2);
	VALIDATE_FIELD_UINT32(N3);
	VALIDATE_FIELD_UINT32(SILENT);
	VALIDATE_FIELD_UINT32(TimeSteps);
	VALIDATE_FIELD_UINT32(LengthSource);
	VALIDATE_FIELD_UINT32(SnapshotsPos);
	VALIDATE_FIELD_MEX_TYPE(SourceFunctions);
	VALIDATE_FIELD_UINT32(SourceMap);
   	VALIDATE_FIELD_UINT32(TypeSource);
	VALIDATE_FIELD_STRING(DefaultGPUDeviceName);
	VALIDATE_FIELD_MEX_TYPE(Ox);
	VALIDATE_FIELD_MEX_TYPE(Oy);
	VALIDATE_FIELD_MEX_TYPE(Oz);
	VALIDATE_FIELD_UINT32(SelRMSorPeak);
	VALIDATE_FIELD_UINT32(SelMapsRMSPeak);
	VALIDATE_FIELD_UINT32(SelMapsSensors);
	VALIDATE_FIELD_UINT32(SensorSubSampling);
	VALIDATE_FIELD_UINT32(SensorStart);
	VALIDATE_FIELD_UINT32(DefaultGPUDeviceNumber);
	VALIDATE_FIELD_STRING(PI_OCL_PATH);
	VALIDATE_FIELD_STRING(kernelfile);
	VALIDATE_FIELD_STRING(kernbinfile);
	VALIDATE_FIELD_INT32(ManualGroupSize);
	VALIDATE_FIELD_INT32(ManualLocalSize);


	INHOST(TimeSteps)=*GET_DATA_UINT32_PR(TimeSteps);


	INHOST(NumberSensors)=(unsigned int) GET_NUMBER_ELEMS(IndexSensorMap);
	NumberSnapshots=(unsigned int) GET_NUMBER_ELEMS(SnapshotsPos);


//Getting pointers from the input matrices
	GET_DATA(InvDXDTplus);
	GET_DATA(DXDTminus);
	GET_DATA(InvDXDTplushp);
	GET_DATA(DXDTminushp);
	GET_DATA(LambdaMiuMatOverH);
	GET_DATA(LambdaMatOverH);
	GET_DATA(MiuMatOverH);
	GET_DATA(TauLong);
	GET_DATA(OneOverTauSigma);
	GET_DATA(TauShear);
	GET_DATA(InvRhoMatH);
	GET_DATA(SourceFunctions);
	GET_DATA(Ox);
	GET_DATA(Oy);
	GET_DATA(Oz);
	GET_DATA_UINT32(IndexSensorMap);
	GET_DATA_UINT32(SourceMap);
	GET_DATA_UINT32(SnapshotsPos);
	GET_DATA_UINT32(MaterialMap);
	GET_DATA_INT32(ManualGroupSize);
	GET_DATA_INT32(ManualLocalSize);

	INHOST(ZoneCount)=*GET_DATA_UINT32_PR(SPP_ZONES);


	TimeStepsSource=(unsigned int) GET_N(SourceFunctions);

	//The INHOST macro add a "h" in the name of the variable if compiling for CUDA, this is needed to avoid conflict between global constant in CUDA and local variables
	//in the host function
	INHOST(NumberSources)=(unsigned int) GET_M(SourceFunctions);

	//Dimension of the matrices (we could have used the size of MaterialMap)
	INHOST(N1)=*GET_DATA_UINT32_PR(N1);
	INHOST(N2)=*GET_DATA_UINT32_PR(N2);
	INHOST(N3)=*GET_DATA_UINT32_PR(N3);

	INHOST(DT) =  *GET_DATA_PR(DT);	 //Temporal step
	INHOST(LengthSource)= *GET_DATA_UINT32_PR(LengthSource);

	INHOST(TypeSource)=*GET_DATA_UINT32_PR(TypeSource);

	INHOST(SelRMSorPeak)=*GET_DATA_UINT32_PR(SelRMSorPeak);
	INHOST(SelMapsRMSPeak)=*GET_DATA_UINT32_PR(SelMapsRMSPeak);
	INHOST(SelMapsSensors)=*GET_DATA_UINT32_PR(SelMapsSensors);
	INHOST(SensorSubSampling)=*GET_DATA_UINT32_PR(SensorSubSampling);
	INHOST(SensorStart)=*GET_DATA_UINT32_PR(SensorStart);

	VALIDATE_FIELD_UINT32(SelRMSorPeak);
	VALIDATE_FIELD_UINT32(SelMapsRMSPeak);

	GET_DATA_STRING(DefaultGPUDeviceName);
	INHOST(DefaultGPUDeviceNumber)=*GET_DATA_UINT32_PR(DefaultGPUDeviceNumber);
	GET_DATA_STRING(PI_OCL_PATH);
	GET_DATA_STRING(kernelfile);
	GET_DATA_STRING(kernbinfile);

	if (TimeStepsSource!=INHOST(LengthSource))
		ERROR_STRING("The limit for time steps in source is different from N-dimension in SourceFunctions ");

	if ((INHOST(N1)+1) != GET_M(MaterialMap))
		ERROR_STRING("Material map dim 0 must be N1+1");
	if ((INHOST(N2)+1) != GET_N(MaterialMap))
			ERROR_STRING("Material map dim 1 must be N2+1");
	if ((INHOST(N3)+1) != GET_O(MaterialMap))
			ERROR_STRING("Material map dim 3 must be N3+1 ");
	if ((INHOST(ZoneCount)) != GET_P(MaterialMap))
			ERROR_STRING("Material map dim 4 must be ZoneCount ");

	if ( !((INHOST(SelRMSorPeak))& SEL_PEAK) && !((INHOST(SelRMSorPeak))&SEL_RMS))
			ERROR_STRING("SelRMSorPeak must be either 1 (RMS), 2 (Peak) or 3 (Both RMS and Peak)");

	if (GET_NUMBER_ELEMS(ManualGroupSize)!=3)
		ERROR_STRING("ManualGroupSize must be an array of 3 values");

	if (GET_NUMBER_ELEMS(ManualLocalSize)!=3)
		ERROR_STRING("ManualGroupSize must be an array of 3 values");

    COUNT_SELECTIONS(INHOST(NumberSelRMSPeakMaps),INHOST(SelMapsRMSPeak));
	if (INHOST(NumberSelRMSPeakMaps)==0)
		ERROR_STRING("SelMapsRMSPeak must select at least one type of map to track");

	COUNT_SELECTIONS(INHOST(NumberSelSensorMaps),INHOST(SelMapsSensors));
	if (INHOST(NumberSelSensorMaps)==0)
		ERROR_STRING("NumberSelSensorMaps must select at least one type of map to track");
	//We detect how many maps we need to keep track
	unsigned int curMapIndex =0;
	ACCOUNT_RMSPEAK(Vx);
	ACCOUNT_RMSPEAK(Vy);
	ACCOUNT_RMSPEAK(Vz);
	ACCOUNT_RMSPEAK(Sigmaxx);
	ACCOUNT_RMSPEAK(Sigmayy);
	ACCOUNT_RMSPEAK(Sigmazz);
	ACCOUNT_RMSPEAK(Sigmaxy);
	ACCOUNT_RMSPEAK(Sigmaxz);
	ACCOUNT_RMSPEAK(Sigmayz);
	ACCOUNT_RMSPEAK(Pressure);

	if (INHOST(NumberSelRMSPeakMaps)!=curMapIndex)
	{
			PRINTF("NumberSelRMSPeakMaps =%i, curMapIndex=%i\n",INHOST(NumberSelRMSPeakMaps),curMapIndex )
			ERROR_STRING("curMapIndex and NumberSelRMSPeakMaps should be the same.... how did this happen?");
	}

	curMapIndex =0;
	ACCOUNT_SENSOR(Vx);
	ACCOUNT_SENSOR(Vy);
	ACCOUNT_SENSOR(Vz);
	ACCOUNT_SENSOR(Sigmaxx);
	ACCOUNT_SENSOR(Sigmayy);
	ACCOUNT_SENSOR(Sigmazz);
	ACCOUNT_SENSOR(Sigmaxy);
	ACCOUNT_SENSOR(Sigmaxz);
	ACCOUNT_SENSOR(Sigmayz);
	ACCOUNT_SENSOR(Pressure);

	if (INHOST(NumberSelSensorMaps)!=curMapIndex)
	{
			PRINTF("NumberSelRMSPeakMaps =%i, curMapIndex=%i\n",INHOST(NumberSelRMSPeakMaps),curMapIndex )
			ERROR_STRING("curMapIndex and NumberSelRMSPeakMaps should be the same.... how did this happen?");
	}

	///// PML conditions, you truly do not want to modify this, the smallest error and you got a nasty field.
	INHOST(PML_Thickness)=*GET_DATA_UINT32_PR(PMLThickness);
	unsigned int INHOST(Limit_I_low_PML)=INHOST(PML_Thickness)-1;
	unsigned int INHOST(Limit_I_up_PML)=INHOST(N1)-INHOST(PML_Thickness);
	unsigned int INHOST(Limit_J_low_PML)=INHOST(PML_Thickness)-1;
	unsigned int INHOST(Limit_J_up_PML)=INHOST(N2)-INHOST(PML_Thickness);
	unsigned int INHOST(Limit_K_low_PML)=INHOST(PML_Thickness)-1;
	unsigned int INHOST(Limit_K_up_PML)=INHOST(N3)-INHOST(PML_Thickness);

	unsigned int INHOST(SizeCorrI)=INHOST(N1)-2*INHOST(PML_Thickness);
	unsigned int INHOST(SizeCorrJ)=INHOST(N2)-2*INHOST(PML_Thickness);
	unsigned int INHOST(SizeCorrK)=INHOST(N3)-2*INHOST(PML_Thickness);

	//The size of the matrices where the PML is valid depends on the size of the PML barrier
	unsigned int INHOST(SizePML) = (INHOST(N1))*(INHOST(N2))*(INHOST(N3))      - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLxp1) = (INHOST(N1)+1)*(INHOST(N2))*(INHOST(N3)) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLyp1) = (INHOST(N1))*(INHOST(N2)+1)*INHOST(N3)   - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLzp1) = (INHOST(N1))*(INHOST(N2))*(INHOST(N3)+1) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLxp1yp1zp1) = (INHOST(N1)+1)*(INHOST(N2)+1)*(INHOST(N3)+1) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;

	PRINTF("SizePML=%i\n",INHOST(SizePML));
	PRINTF("SizePMLxp1=%i\n",INHOST(SizePMLxp1));
	PRINTF("SizePMLyp1=%i\n",INHOST(SizePMLyp1));
	PRINTF("SizePMLzp1=%i\n",INHOST(SizePMLzp1));
	PRINTF("SizePMLxp1yp1zp1=%i\n",INHOST(SizePMLxp1yp1zp1));

	//for SPP
	// unsigned int SizeMatMap_zone = INHOST(TotalIndexCount)*INHOST(ZoneCount);
//We define a few variable required to create arrays depending if it is for Numpy or Mex
#ifdef MATLAB_MEX
	    mwSize ndim=1;
			mwSize dims[5];
			dims[0]=1;

	    const char *fieldNames[] = {"Vx", "Vy", "Vz","Sigma_xx", "Sigma_yy" ,"Sigma_zz", "Sigma_xy","Sigma_xz","Sigma_yz","Pressure"};
	    mxArray * LastVMap_mx=mxCreateStructArray( 1, dims, 9, fieldNames );
#else

			npy_intp dims[5];
			//int dims[3];
			int ndim;

			PyArray_Descr *__descr;

#endif

  	CREATE_ARRAY_AND_INIT(Vx_res,INHOST(N1)+1,INHOST(N2),INHOST(N3));
	CREATE_ARRAY_AND_INIT(Vy_res,INHOST(N1),INHOST(N2)+1,INHOST(N3));
	CREATE_ARRAY_AND_INIT(Vz_res,INHOST(N1),INHOST(N2),INHOST(N3)+1);
	CREATE_ARRAY_AND_INIT(Sigma_xx_res,INHOST(N1),INHOST(N2),INHOST(N3));
	CREATE_ARRAY_AND_INIT(Sigma_yy_res,INHOST(N1),INHOST(N2),INHOST(N3));
	CREATE_ARRAY_AND_INIT(Sigma_zz_res,INHOST(N1),INHOST(N2),INHOST(N3));
	CREATE_ARRAY_AND_INIT(Sigma_xy_res,INHOST(N1)+1,INHOST(N2)+1,INHOST(N3)+1);
	CREATE_ARRAY_AND_INIT(Sigma_xz_res,INHOST(N1)+1,INHOST(N2)+1,INHOST(N3)+1);
	CREATE_ARRAY_AND_INIT(Sigma_yz_res,INHOST(N1)+1,INHOST(N2)+1,INHOST(N3)+1);
	CREATE_ARRAY_AND_INIT(Pressure_res,INHOST(N1),INHOST(N2),INHOST(N3));

	ndim=5;
	dims[0]=INHOST(N1);
	dims[1]=INHOST(N2);
	dims[2]=INHOST(N3);
	dims[3]=INHOST(NumberSelRMSPeakMaps);
	if 	 ((INHOST(SelRMSorPeak) & SEL_PEAK)  && (INHOST(SelRMSorPeak) & SEL_RMS))
	 	dims[4]=2; //both peak and RMS
	else
		dims[4]=1; //just one of them
  CREATE_ARRAY(SqrAcc);
	GET_DATA(SqrAcc);
	memset(SqrAcc_pr,0,dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*sizeof(mexType));

  unsigned int INHOST(CurrSnap)=0;

	ndim=3;
	dims[0]=INHOST(N1);
	dims[1]=INHOST(N2);
	if (NumberSnapshots>0)
		dims[2]=NumberSnapshots;
	else
		dims[2]=1;

	CREATE_ARRAY(Snapshots);
	GET_DATA(Snapshots);
	memset(Snapshots_pr,0,dims[0]*dims[1]*dims[2]*sizeof(mexType));

    
	ndim=3;
	unsigned int MaxSensorSteps=INHOST(TimeSteps)/INHOST(SensorSubSampling)+1-INHOST(SensorStart);
	dims[0]=INHOST(NumberSensors);
	dims[1]=MaxSensorSteps;
	dims[2]=INHOST(NumberSelSensorMaps);
	PRINTF("For sensor, dims %li %li %li, total %li \n",dims[0],dims[1],dims[2],dims[0]*dims[1]*dims[2])
	CREATE_ARRAY(SensorOutput);
	GET_DATA(SensorOutput);
	memset(SensorOutput_pr,0,dims[0]*dims[1]*dims[2]*sizeof(mexType));

#ifdef MATLAB_MEX
	PRINTF(" Staggered FDTD - compiled at %s - %s\n",__DATE__,__TIME__);
#else
	PySys_WriteStdout(" Staggered FDTD - compiled at %s - %s\n", __DATE__, __TIME__);
#endif

    PRINTF("N1, N2,N3 , ZoneCount and DT= %i,%i,%i,%i,%g\n",INHOST(N1),INHOST(N2),INHOST(N3),INHOST(ZoneCount),INHOST(DT));
    PRINTF("Number of sensors , timesteps for sensors and total maps= %ui, %i, %i\n",INHOST(NumberSensors),	(int)dims[1],INHOST(NumberSelSensorMaps));

	time_t start_t, end_t;
	time(&start_t);
	//voila, from here the truly specific X64 or CUDA implementation starts


INHOST(SelK)=INHOST(N3)/2;

#include "FDTD3D_CPU_VERSION.h"

   ////END CPU
   time(&end_t);
   double diff_t = difftime(end_t, start_t);
   PRINTF("Execution time = %f\n", diff_t);




PRINTF("Done\n");

#ifdef MATLAB_MEX
	mxSetField( LastVMap_mx, 0, fieldNames[ 0 ], Vx_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 1 ], Vy_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 2 ], Vz_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 3 ], Sigma_xx_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 4 ], Sigma_yy_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 6 ], Sigma_xy_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 5 ], Sigma_zz_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 7 ], Sigma_xz_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 8 ], Sigma_yz_res_mx );
	mxSetField( LastVMap_mx, 0, fieldNames[ 9 ], Pressure_res_mx );
	SensorOutput_out =SensorOutput_mx;
	LastVMap_out=LastVMap_mx;
  Snapshots_out=Snapshots_mx;
  SqrAcc_out=SqrAcc_mx;

#else

	 RELEASE_STRING_OBJ(DefaultGPUDeviceName);
	 RELEASE_STRING_OBJ(PI_OCL_PATH);
	 RELEASE_STRING_OBJ(kernelfile);
	 RELEASE_STRING_OBJ(kernbinfile);

	//return Py_BuildValue("OOOOOOO", SensorOutput_mx,Vx_mx,Vy_mx,Sigma_xx_mx,Sigma_yy_mx,Sigma_xy_mx,Snapshots_mx);

    PyObject *MyResult;

		MyResult =  Py_BuildValue("N{sNsNsNsNsNsNsNsNsNsN}NN", SensorOutput_mx, "Vx",Vx_res_mx,
																        "Vy",Vy_res_mx,
																        "Vz",Vz_res_mx,
																        "Sigma_xx",Sigma_xx_res_mx,
																        "Sigma_yy",Sigma_yy_res_mx,
																        "Sigma_zz",Sigma_zz_res_mx,
																        "Sigma_xy",Sigma_xy_res_mx,
																        "Sigma_xz",Sigma_xz_res_mx,
																        "Sigma_yz",Sigma_yz_res_mx,
																		"Pressure",Pressure_res_mx,
																		SqrAcc_mx,Snapshots_mx);



   return MyResult;
#endif
}
