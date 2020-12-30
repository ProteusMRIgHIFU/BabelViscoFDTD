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

#ifdef MATLAB_MEX // The differences between Matlab mex function and a Python extension are quite minimal for our purposes
	#include "mex.h"
	//#include "matrix.h"

	//We define all the input params
	#define InputStruct prhs[0]
	//And the output params
	#define SensorOutput_out plhs[0]
	#define LastVMap_out plhs[1]
	#define SqrAcc_out plhs[2]
	#define Snapshots_out plhs[3]


#else
	#include <Python.h>
	#include <ndarrayobject.h>
#endif

#include <math.h>
#include <string.h>
#include <stdlib.h>
#if defined(USE_OPENMP)
#include <omp.h>
#endif

#include <stdio.h>
#include <time.h>
#if defined(__MACH__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#ifdef OPENCL
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#include <unistd.h>
#endif
#endif

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
								INHOST(NumberSensors),INHOST(PML_Thickness), INHOST(SelRMSorPeak), INHOST(SelMapsRMSPeak),
								INHOST(NumberSelRMSPeakMaps),
								INHOST(IndexRMSPeak_ALLV)=0,
								INHOST(IndexRMSPeak_Vx)=0,
								INHOST(IndexRMSPeak_Vy)=0,
								INHOST(IndexRMSPeak_Vz)=0,
								INHOST(IndexRMSPeak_Sigmaxx)=0,
								INHOST(IndexRMSPeak_Sigmayy)=0,
								INHOST(IndexRMSPeak_Sigmazz)=0,
								INHOST(IndexRMSPeak_Sigmaxy)=0,
								INHOST(IndexRMSPeak_Sigmaxz)=0,
								INHOST(IndexRMSPeak_Sigmayz)=0;
    mexType INHOST(DT);



#ifdef WIN32
	typedef unsigned long DWORD;
	DWORD		dummyID;
#endif
	//---------------------------------------------------------------//
	// check input parameters
	//---------------------------------------------------------------//

	PRINTF("Running with new interface conditions\n");
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

	GET_FIELD(USE_SPP);

	// unsigned int INHOST(USE_SPP), INHOST(InternalIndexCount),INHOST(EdgeIndexCount),
	// 				INHOST(TotalIndexCount),INHOST(ZoneCount);

	unsigned int INHOST(USE_SPP),INHOST(ZoneCount);

	INHOST(USE_SPP) = *GET_DATA_UINT32_PR(USE_SPP);


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
	GET_DATA_UINT32(IndexSensorMap);
	GET_DATA(SourceFunctions);
	GET_DATA(Ox);
	GET_DATA(Oy);
	GET_DATA(Oz);
	GET_DATA_UINT32(SourceMap);
	GET_DATA_UINT32(SnapshotsPos);
	GET_DATA_UINT32(MaterialMap);


	INHOST(ZoneCount)=INHOST(USE_SPP);


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

	VALIDATE_FIELD_UINT32(SelRMSorPeak);
	VALIDATE_FIELD_UINT32(SelMapsRMSPeak);

	GET_DATA_STRING(DefaultGPUDeviceName);

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



  COUNT_SELECTIONS(INHOST(NumberSelRMSPeakMaps),INHOST(SelMapsRMSPeak));
	if (INHOST(NumberSelRMSPeakMaps)==0)
		ERROR_STRING("SelMapsRMSPeak must select at least one type of map to track");

	//We detect how many maps we need to keep track
	unsigned int curMapIndex =0;
	ACCOUNT_RMSPEAK(ALLV);
	ACCOUNT_RMSPEAK(Vx);
	ACCOUNT_RMSPEAK(Vy);
	ACCOUNT_RMSPEAK(Vz);
	ACCOUNT_RMSPEAK(Sigmaxx);
	ACCOUNT_RMSPEAK(Sigmayy);
	ACCOUNT_RMSPEAK(Sigmazz);
	ACCOUNT_RMSPEAK(Sigmaxy);
	ACCOUNT_RMSPEAK(Sigmaxz);
	ACCOUNT_RMSPEAK(Sigmayz);

	if (INHOST(NumberSelRMSPeakMaps)!=curMapIndex)
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
	unsigned int INHOST(SizePML) = (INHOST(N1))*(INHOST(N2))*(INHOST(N3)) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLxp1) = (INHOST(N1)+1)*(INHOST(N2))*(INHOST(N3)) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLyp1) = (INHOST(N1))*(INHOST(N2)+1)*INHOST(N3) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLzp1) = (INHOST(N1))*(INHOST(N2))*(INHOST(N3)+1) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;
	unsigned int INHOST(SizePMLxp1yp1zp1) = (INHOST(N1)+1)*(INHOST(N2)+1)*(INHOST(N3)+1) - INHOST(SizeCorrI)*INHOST(SizeCorrJ)*INHOST(SizeCorrK)+1;

	PRINTF("SizePML=%i\n",SizePML);
	PRINTF("SizePMLxp1=%i\n",SizePMLxp1);
	PRINTF("SizePMLyp1=%i\n",SizePMLyp1);
	PRINTF("SizePMLzp1=%i\n",SizePMLzp1);
	PRINTF("SizePMLxp1yp1zp1=%i\n",SizePMLxp1yp1zp1);

	//for SPP
	// unsigned int SizeMatMap_zone = INHOST(TotalIndexCount)*INHOST(ZoneCount);
//We define a few variable required to create arrays depending if it is for Numpy or Mex
#ifdef MATLAB_MEX
	    mwSize ndim=1;
			mwSize dims[5];
			dims[0]=1;

	    const char *fieldNames[] = {"Vx", "Vy", "Vz","Sigma_xx", "Sigma_yy" ,"Sigma_zz", "Sigma_xy","Sigma_xz","Sigma_yz"};
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

  unsigned int CurrSnap=0;

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
	dims[0]=INHOST(NumberSensors);
	dims[1]=INHOST(TimeSteps);
	dims[2]=3;
	CREATE_ARRAY(SensorOutput);
	GET_DATA(SensorOutput);
	memset(SensorOutput_pr,0,dims[0]*dims[1]*dims[2]*sizeof(mexType));

#ifdef MATLAB_MEX
	PRINTF(" Staggered FDTD - compiled at %s - %s\n",__DATE__,__TIME__);
#else
	PySys_WriteStdout(" Staggered FDTD - compiled at %s - %s\n", __DATE__, __TIME__);
#endif

    PRINTF("N1, N2,N3 , ZoneCount and DT= %i,%i,%i,%g\n",INHOST(N1),INHOST(N2),INHOST(N3),INHOST(ZoneCount),INHOST(DT));
    PRINTF("Number of sensors x timesteps= %i, %i\n",INHOST(NumberSensors),INHOST(TimeSteps));

	time_t start_t, end_t;
	time(&start_t);
	//voila, from here the truly specific X64 or CUDA implementation starts

// #ifndef MATLAB_MEX
// 	Py_BEGIN_ALLOW_THREADS;
// #endif

#if defined(CUDA) || defined(OPENCL)
  #include "FDTD3D_GPU_VERSION.h"
#else
	//////////BEGIN CPU SPECIFC
  FILE * FDEBUG =fopen("DEBUG.OUT","w");
	fprintf(FDEBUG,"Starting execution\n");
	fflush(FDEBUG);
  #include "FDTD3D_CPU_VERSION.h"
#endif

   ////END CPU
   time(&end_t);
   double diff_t = difftime(end_t, start_t);
   PRINTF("Execution time = %f\n", diff_t);

 #if defined(CUDA) || defined(OPENCL)

 #else
 	  fprintf(FDEBUG,"Execution time = %f\n", diff_t);
		fflush(FDEBUG);
 #endif

 #if defined(CUDA) || defined(OPENCL)

 #else
	 fclose(FDEBUG);
 #endif

 // #ifndef MATLAB_MEX
 // 	Py_END_ALLOW_THREADS;
 // #endif

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
	SensorOutput_out =SensorOutput_mx;
	LastVMap_out=LastVMap_mx;
  Snapshots_out=Snapshots_mx;
  SqrAcc_out=SqrAcc_mx;

#else

	 RELEASE_STRING_OBJ(DefaultGPUDeviceName);

	//return Py_BuildValue("OOOOOOO", SensorOutput_mx,Vx_mx,Vy_mx,Sigma_xx_mx,Sigma_yy_mx,Sigma_xy_mx,Snapshots_mx);

    PyObject *MyResult;

		MyResult =  Py_BuildValue("N{sNsNsNsNsNsNsNsNsN}NN", SensorOutput_mx, "Vx",Vx_res_mx,
																        "Vy",Vy_res_mx,
																        "Vz",Vz_res_mx,
																        "Sigma_xx",Sigma_xx_res_mx,
																        "Sigma_yy",Sigma_yy_res_mx,
																        "Sigma_zz",Sigma_zz_res_mx,
																        "Sigma_xy",Sigma_xy_res_mx,
																        "Sigma_xz",Sigma_xz_res_mx,
																        "Sigma_yz",Sigma_yz_res_mx,SqrAcc_mx,Snapshots_mx);



   return MyResult;
#endif
}
