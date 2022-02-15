/*

*/
#ifndef __3D_Staggered_GPU
#define __3D_Staggered_GPU

#include <string.h>

enum CONSTANT_TYPES
{
	G_INT,
	G_FLOAT
};

#define MAX_SIZE_PML 101

long Find_GPU_Size(const long m)
{
	//find higher dimension that is multiple of 4
	long n =m;
	while(n%4!=0)
		n++;
	return n;
}

char* common_read_file(const char *path, long *length_out)
{
	//common_read_file is used to read the compiled binary for OPENCL
    char *buffer;
    FILE *f;
    long length;

    f = fopen(path, "r");
	if (f==NULL)
		return NULL;
    //assert(NULL != f);
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = (char *)malloc(length);
    if (fread(buffer, 1, length, f) < (size_t)length) {
        return NULL;
    }
    fclose(f);
    if (NULL != length_out) {
        *length_out = length;
    }
    return buffer;
}

#ifdef OPENCL
const char *opencl_err_code (cl_int err_in)
{
	/*
	Simple code to string for error codes
	*/
		switch (err_in) {
				case CL_SUCCESS:
						return (char*)"CL_SUCCESS";
				case CL_DEVICE_NOT_FOUND:
						return (char*)"CL_DEVICE_NOT_FOUND";
				case CL_DEVICE_NOT_AVAILABLE:
						return (char*)"CL_DEVICE_NOT_AVAILABLE";
				case CL_COMPILER_NOT_AVAILABLE:
						return (char*)"CL_COMPILER_NOT_AVAILABLE";
				case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						return (char*)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
				case CL_OUT_OF_RESOURCES:
						return (char*)"CL_OUT_OF_RESOURCES";
				case CL_OUT_OF_HOST_MEMORY:
						return (char*)"CL_OUT_OF_HOST_MEMORY";
				case CL_PROFILING_INFO_NOT_AVAILABLE:
						return (char*)"CL_PROFILING_INFO_NOT_AVAILABLE";
				case CL_MEM_COPY_OVERLAP:
						return (char*)"CL_MEM_COPY_OVERLAP";
				case CL_IMAGE_FORMAT_MISMATCH:
						return (char*)"CL_IMAGE_FORMAT_MISMATCH";
				case CL_IMAGE_FORMAT_NOT_SUPPORTED:
						return (char*)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
				case CL_BUILD_PROGRAM_FAILURE:
						return (char*)"CL_BUILD_PROGRAM_FAILURE";
				case CL_MAP_FAILURE:
						return (char*)"CL_MAP_FAILURE";
				case CL_MISALIGNED_SUB_BUFFER_OFFSET:
						return (char*)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
				case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
						return (char*)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
				case CL_INVALID_VALUE:
						return (char*)"CL_INVALID_VALUE";
				case CL_INVALID_DEVICE_TYPE:
						return (char*)"CL_INVALID_DEVICE_TYPE";
				case CL_INVALID_PLATFORM:
						return (char*)"CL_INVALID_PLATFORM";
				case CL_INVALID_DEVICE:
						return (char*)"CL_INVALID_DEVICE";
				case CL_INVALID_CONTEXT:
						return (char*)"CL_INVALID_CONTEXT";
				case CL_INVALID_QUEUE_PROPERTIES:
						return (char*)"CL_INVALID_QUEUE_PROPERTIES";
				case CL_INVALID_COMMAND_QUEUE:
						return (char*)"CL_INVALID_COMMAND_QUEUE";
				case CL_INVALID_HOST_PTR:
						return (char*)"CL_INVALID_HOST_PTR";
				case CL_INVALID_MEM_OBJECT:
						return (char*)"CL_INVALID_MEM_OBJECT";
				case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
						return (char*)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
				case CL_INVALID_IMAGE_SIZE:
						return (char*)"CL_INVALID_IMAGE_SIZE";
				case CL_INVALID_SAMPLER:
						return (char*)"CL_INVALID_SAMPLER";
				case CL_INVALID_BINARY:
						return (char*)"CL_INVALID_BINARY";
				case CL_INVALID_BUILD_OPTIONS:
						return (char*)"CL_INVALID_BUILD_OPTIONS";
				case CL_INVALID_PROGRAM:
						return (char*)"CL_INVALID_PROGRAM";
				case CL_INVALID_PROGRAM_EXECUTABLE:
						return (char*)"CL_INVALID_PROGRAM_EXECUTABLE";
				case CL_INVALID_KERNEL_NAME:
						return (char*)"CL_INVALID_KERNEL_NAME";
				case CL_INVALID_KERNEL_DEFINITION:
						return (char*)"CL_INVALID_KERNEL_DEFINITION";
				case CL_INVALID_KERNEL:
						return (char*)"CL_INVALID_KERNEL";
				case CL_INVALID_ARG_INDEX:
						return (char*)"CL_INVALID_ARG_INDEX";
				case CL_INVALID_ARG_VALUE:
						return (char*)"CL_INVALID_ARG_VALUE";
				case CL_INVALID_ARG_SIZE:
						return (char*)"CL_INVALID_ARG_SIZE";
				case CL_INVALID_KERNEL_ARGS:
						return (char*)"CL_INVALID_KERNEL_ARGS";
				case CL_INVALID_WORK_DIMENSION:
						return (char*)"CL_INVALID_WORK_DIMENSION";
				case CL_INVALID_WORK_GROUP_SIZE:
						return (char*)"CL_INVALID_WORK_GROUP_SIZE";
				case CL_INVALID_WORK_ITEM_SIZE:
						return (char*)"CL_INVALID_WORK_ITEM_SIZE";
				case CL_INVALID_GLOBAL_OFFSET:
						return (char*)"CL_INVALID_GLOBAL_OFFSET";
				case CL_INVALID_EVENT_WAIT_LIST:
						return (char*)"CL_INVALID_EVENT_WAIT_LIST";
				case CL_INVALID_EVENT:
						return (char*)"CL_INVALID_EVENT";
				case CL_INVALID_OPERATION:
						return (char*)"CL_INVALID_OPERATION";
				case CL_INVALID_GL_OBJECT:
						return (char*)"CL_INVALID_GL_OBJECT";
				case CL_INVALID_BUFFER_SIZE:
						return (char*)"CL_INVALID_BUFFER_SIZE";
				case CL_INVALID_MIP_LEVEL:
						return (char*)"CL_INVALID_MIP_LEVEL";
				case CL_INVALID_GLOBAL_WORK_SIZE:
						return (char*)"CL_INVALID_GLOBAL_WORK_SIZE";
				case CL_INVALID_PROPERTY:
						return (char*)"CL_INVALID_PROPERTY";

				default:
						return (char*)"UNKNOWN ERROR";
		}
}
#endif


int mxcheck(int result, char const *const func, const char *const file, int const line)
{
	/*
	mxcheck verifies status of GPU operation and print msg in case of error
	*/
  #if defined(CUDA)
    if (result)
    {
			char _bline[10000];\
			snprintf(_bline,10000,"CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString((cudaError_t)result), func);
        ERROR_STRING(_bline);
    }
		return 1L;
	#endif
	#if defined(OPENCL)
  if (result!= CL_SUCCESS)
  {
    PRINTF( "OPENCL error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, result, opencl_err_code(result), func);
    ERROR_STRING("Stopping execution");
  }
	return 1L;
  #endif

	#if defined(METAL)
	if (result== 0)
  {
    PRINTF( "METAL error at %s:%d  \"%s\" \n",
            file, line, func);
    ERROR_STRING("Stopping execution");
  }
	return 1L;
	#endif

}


#if defined(CUDA)
//declaration of constant values for FDTD solution
__constant__ mexType DT;
__constant__ unsigned int N1;
__constant__ unsigned int N2;
__constant__ unsigned int N3;
__constant__ unsigned int Limit_I_low_PML;
__constant__ unsigned int Limit_J_low_PML;
__constant__ unsigned int Limit_K_low_PML;
__constant__ unsigned int Limit_I_up_PML;
__constant__ unsigned int Limit_J_up_PML;
__constant__ unsigned int Limit_K_up_PML;
__constant__ unsigned int SizeCorrI;
__constant__ unsigned int SizeCorrJ;
__constant__ unsigned int SizeCorrK;
__constant__ unsigned int PML_Thickness;
__constant__ unsigned int NumberSources;
__constant__ unsigned int LengthSource;
__constant__ unsigned int NumberSensors;
__constant__ unsigned int TimeSteps;

__constant__ unsigned int SizePML;
__constant__ unsigned int SizePMLxp1;
__constant__ unsigned int SizePMLyp1;
__constant__ unsigned int SizePMLzp1;
__constant__ unsigned int SizePMLxp1yp1zp1;
__constant__ unsigned int ZoneCount;

__constant__ unsigned int SelRMSorPeak;
__constant__ unsigned int SelMapsRMSPeak;
__constant__ _PT IndexRMSPeak_ALLV;
__constant__ _PT IndexRMSPeak_Vx;
__constant__ _PT IndexRMSPeak_Vy;
__constant__ _PT IndexRMSPeak_Vz;
__constant__ _PT IndexRMSPeak_Sigmaxx;
__constant__ _PT IndexRMSPeak_Sigmayy;
__constant__ _PT IndexRMSPeak_Sigmazz;
__constant__ _PT IndexRMSPeak_Sigmaxy;
__constant__ _PT IndexRMSPeak_Sigmaxz;
__constant__ _PT IndexRMSPeak_Sigmayz;
__constant__ _PT IndexRMSPeak_Pressure;
__constant__ _PT NumberSelRMSPeakMaps;

__constant__ unsigned int SelMapsSensors;
__constant__ _PT IndexSensor_ALLV;
__constant__ _PT IndexSensor_Vx;
__constant__ _PT IndexSensor_Vy;
__constant__ _PT IndexSensor_Vz;
__constant__ _PT IndexSensor_Sigmaxx;
__constant__ _PT IndexSensor_Sigmayy;
__constant__ _PT IndexSensor_Sigmazz;
__constant__ _PT IndexSensor_Sigmaxy;
__constant__ _PT IndexSensor_Sigmaxz;
__constant__ _PT IndexSensor_Sigmayz;
__constant__ _PT IndexSensor_Pressure;
__constant__ _PT NumberSelSensorMaps;
__constant__ _PT SensorSubSampling;
__constant__ _PT SensorStart;

__constant__ mexType gpuInvDXDTpluspr[MAX_SIZE_PML];
__constant__ mexType gpuDXDTminuspr[MAX_SIZE_PML];
__constant__ mexType gpuInvDXDTplushppr[MAX_SIZE_PML];
__constant__ mexType gpuDXDTminushppr[MAX_SIZE_PML];

//Calculate the block dimensions
#define CUDA_GRID_BLOC_BASE(__KERNEL__)\
  dim3 dimBlock## __KERNEL__;  \
  dim3 dimGrid## __KERNEL__; \
  mxcheckGPUErrors(cudaOccupancyMaxPotentialBlockSize( &minGridSize, &minBlockSize,\
                                  __KERNEL__, 0, 0));\
   PRINTF("minGridSize and Blocksize from API for " #__KERNEL__ " = %i and %i\n",minGridSize,minBlockSize)\
   dimBlock## __KERNEL__.x=8;\
   dimBlock## __KERNEL__.y=8;\
   dimBlock## __KERNEL__.z=(unsigned int)floor(minBlockSize/(dimBlock ## __KERNEL__.x*dimBlock ## __KERNEL__.y));

#if defined(USE_MINI_KERNELS_CUDA)
#define REDUCTION_MAIN_KERNEL INHOST(PML_Thickness)*2
#else
#define REDUCTION_MAIN_KERNEL 0
#endif
#define CUDA_GRID_BLOC_CALC_MAIN(__KERNEL__)\
  CUDA_GRID_BLOC_BASE(__KERNEL__)\
   dimGrid## __KERNEL__.x  = (unsigned int)ceil((float)(INHOST(N1) - REDUCTION_MAIN_KERNEL) /  dimBlock ## __KERNEL__.x);\
   dimGrid## __KERNEL__.y  = (unsigned int)ceil((float)(INHOST(N2) - REDUCTION_MAIN_KERNEL) /  dimBlock ## __KERNEL__.y);\
   dimGrid## __KERNEL__.z  = (unsigned int)ceil((float)(INHOST(N3) - REDUCTION_MAIN_KERNEL) /  dimBlock ## __KERNEL__.z);\
  PRINTF(#__KERNEL__ " block size to %dx%dx%d\n", dimBlock ## __KERNEL__.x, dimBlock ## __KERNEL__.y,dimBlock## __KERNEL__.z);\
  PRINTF(#__KERNEL__ " Stress grid size to %dx%dx%d\n", dimGrid ## __KERNEL__.x, dimGrid ## __KERNEL__.y,dimGrid## __KERNEL__.z);
 

#define CUDA_GRID_BLOC_CALC_PML(__TYPE__)\
  CUDA_GRID_BLOC_BASE(PML_1_ ##__TYPE__ ##Kernel);\
  CUDA_GRID_BLOC_BASE(PML_2_ ##__TYPE__ ##Kernel);\
  CUDA_GRID_BLOC_BASE(PML_3_ ##__TYPE__ ##Kernel);\
  CUDA_GRID_BLOC_BASE(PML_4_ ##__TYPE__ ##Kernel);\
  CUDA_GRID_BLOC_BASE(PML_5_ ##__TYPE__ ##Kernel);\
  CUDA_GRID_BLOC_BASE(PML_6_ ##__TYPE__ ##Kernel);\
  dimGridPML_1_## __TYPE__ ##Kernel.x =(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_1_## __TYPE__ ##Kernel.x);\
  dimGridPML_1_## __TYPE__ ##Kernel.y=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_1_## __TYPE__ ##Kernel.y);\
  dimGridPML_1_## __TYPE__ ##Kernel.z=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_1_## __TYPE__ ##Kernel.z);\
  \
  dimGridPML_2_## __TYPE__ ##Kernel.x=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_2_## __TYPE__ ##Kernel.x);\
  dimGridPML_2_## __TYPE__ ##Kernel.y=(unsigned int)ceil((float)(INHOST(N2)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_2_## __TYPE__ ##Kernel.y);\
  dimGridPML_2_## __TYPE__ ##Kernel.z=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_2_## __TYPE__ ##Kernel.z);\
  \
  dimGridPML_3_## __TYPE__ ##Kernel.x=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_3_## __TYPE__ ##Kernel.x);\
  dimGridPML_3_## __TYPE__ ##Kernel.y=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_3_## __TYPE__ ##Kernel.y);\
  dimGridPML_3_## __TYPE__ ##Kernel.z=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_3_## __TYPE__ ##Kernel.z);\
  \
  dimGridPML_4_## __TYPE__ ##Kernel.x=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_4_## __TYPE__ ##Kernel.x);\
  dimGridPML_4_## __TYPE__ ##Kernel.y=(unsigned int)ceil((float)(INHOST(N2)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_4_## __TYPE__ ##Kernel.y);\
  dimGridPML_4_## __TYPE__ ##Kernel.z=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_4_## __TYPE__ ##Kernel.z);\
  \
  dimGridPML_5_## __TYPE__ ##Kernel.x=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_5_## __TYPE__ ##Kernel.x);\
  dimGridPML_5_## __TYPE__ ##Kernel.y=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_5_## __TYPE__ ##Kernel.y);\
  dimGridPML_5_## __TYPE__ ##Kernel.z=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_5_## __TYPE__ ##Kernel.z);\
  \
  dimGridPML_6_## __TYPE__ ##Kernel.x=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) dimBlockPML_6_## __TYPE__ ##Kernel.x);\
  dimGridPML_6_## __TYPE__ ##Kernel.y=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_6_## __TYPE__ ##Kernel.y);\
  dimGridPML_6_## __TYPE__ ##Kernel.z=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) dimBlockPML_6_## __TYPE__ ##Kernel.z);\
  \
  PRINTF("PML_1_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",dimGridPML_1_## __TYPE__ ##Kernel.x,dimGridPML_1_## __TYPE__ ##Kernel.y,dimGridPML_1_## __TYPE__ ##Kernel.z,\
  dimGridPML_1_## __TYPE__ ##Kernel.x*dimBlockPML_1_## __TYPE__ ##Kernel.x,dimGridPML_1_## __TYPE__ ##Kernel.y*dimBlockPML_1_## __TYPE__ ##Kernel.y,dimGridPML_1_## __TYPE__ ##Kernel.z*dimBlockPML_1_## __TYPE__ ##Kernel.z);\
  PRINTF("PML_2_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",dimGridPML_2_## __TYPE__ ##Kernel.x,dimGridPML_2_## __TYPE__ ##Kernel.y,dimGridPML_2_## __TYPE__ ##Kernel.z,\
  dimGridPML_2_## __TYPE__ ##Kernel.x*dimBlockPML_2_## __TYPE__ ##Kernel.x,dimGridPML_2_## __TYPE__ ##Kernel.y*dimBlockPML_2_## __TYPE__ ##Kernel.y,dimGridPML_2_## __TYPE__ ##Kernel.z*dimBlockPML_2_## __TYPE__ ##Kernel.z);\
  PRINTF("PML_3_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",dimGridPML_3_## __TYPE__ ##Kernel.x,dimGridPML_3_## __TYPE__ ##Kernel.y,dimGridPML_3_## __TYPE__ ##Kernel.z,\
  dimGridPML_3_## __TYPE__ ##Kernel.x*dimBlockPML_3_## __TYPE__ ##Kernel.x,dimGridPML_3_## __TYPE__ ##Kernel.y*dimBlockPML_3_## __TYPE__ ##Kernel.y,dimGridPML_3_## __TYPE__ ##Kernel.z*dimBlockPML_3_## __TYPE__ ##Kernel.z);\
  PRINTF("PML_4_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",dimGridPML_4_## __TYPE__ ##Kernel.x,dimGridPML_4_## __TYPE__ ##Kernel.y,dimGridPML_4_## __TYPE__ ##Kernel.z,\
  dimGridPML_4_## __TYPE__ ##Kernel.x*dimBlockPML_4_## __TYPE__ ##Kernel.x,dimGridPML_4_## __TYPE__ ##Kernel.y*dimBlockPML_4_## __TYPE__ ##Kernel.y,dimGridPML_4_## __TYPE__ ##Kernel.z*dimBlockPML_4_## __TYPE__ ##Kernel.z);\
  PRINTF("PML_5_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",dimGridPML_5_## __TYPE__ ##Kernel.x,dimGridPML_5_## __TYPE__ ##Kernel.y,dimGridPML_5_## __TYPE__ ##Kernel.z,\
  dimGridPML_5_## __TYPE__ ##Kernel.x*dimBlockPML_5_## __TYPE__ ##Kernel.x,dimGridPML_5_## __TYPE__ ##Kernel.y*dimBlockPML_5_## __TYPE__ ##Kernel.y,dimGridPML_5_## __TYPE__ ##Kernel.z*dimBlockPML_5_## __TYPE__ ##Kernel.z);\
   PRINTF("PML_6_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",dimGridPML_6_## __TYPE__ ##Kernel.x,dimGridPML_6_## __TYPE__ ##Kernel.y,dimGridPML_6_## __TYPE__ ##Kernel.z,\
  dimGridPML_6_## __TYPE__ ##Kernel.x*dimBlockPML_6_## __TYPE__ ##Kernel.x,dimGridPML_6_## __TYPE__ ##Kernel.y*dimBlockPML_6_## __TYPE__ ##Kernel.y,dimGridPML_6_## __TYPE__ ##Kernel.z*dimBlockPML_6_## __TYPE__ ##Kernel.z);
 
#endif
//---------------------------------------------
#if defined(OPENCL) || defined(METAL)
//OPENCL
#define MAXP_BUFFER_GPU_CODE 2000000
char BUFFER_FOR_GPU_CODE[MAXP_BUFFER_GPU_CODE];
int __InitBuffer =0;
#endif
//--------------------------


#define k_blockDimX    8
#define k_blockDimY    8
#define k_blockDimZ    8

#define k_blockDimMaxY  32

#if defined(CUDA)

#define mxcheckGPUErrors(val)           mxcheck ( (val), #val, __FILE__, __LINE__ )


#define InitSymbol(_NameVar,_datatype,_gtype) mxcheckGPUErrors(cudaMemcpyToSymbol(_NameVar,&INHOST(_NameVar),sizeof(_datatype)));

#define ownGpuCalloc(_NameVar,_dataType,_size) _dataType * gpu_ ## _NameVar ## _pr; \
									PRINTF("Allocating in GPU for " #_NameVar " %lu elem. (nZones=%i)\n",(_PT)_size*INHOST(ZoneCount),INHOST(ZoneCount));\
									mxcheckGPUErrors(cudaMalloc((void **)&gpu_ ## _NameVar ##_pr,_size*sizeof(_dataType)*INHOST(ZoneCount))); \
									NumberAlloc++;\
									{ \
										_dataType * temp_zeros_pr = (_dataType *) calloc(_size,sizeof(_dataType)*INHOST(ZoneCount)); \
										mxcheckGPUErrors(cudaMemcpy(gpu_ ## _NameVar ## _pr, temp_zeros_pr, _size*sizeof(_dataType)*INHOST(ZoneCount), cudaMemcpyHostToDevice));\
										free(temp_zeros_pr);\
									};

#define CreateAndCopyFromMXVarOnGPU2(_NameVar,_dataType) SizeCopy =GET_NUMBER_ELEMS(_NameVar); \
										 PRINTF("Allocating in GPU for " #_NameVar " %lu elem.\n",(_PT)SizeCopy);\
										 mxcheckGPUErrors(cudaMalloc((void **)&gpu_ ## _NameVar ##_pr,SizeCopy*sizeof(_dataType))); \
										 NumberAlloc++;\
										 mxcheckGPUErrors(cudaMemcpy(gpu_ ## _NameVar ## _pr, _NameVar ## _pr, SizeCopy*sizeof(_dataType), cudaMemcpyHostToDevice));

#define CreateAndCopyFromMXVarOnGPU(_NameVar,_dataType) _dataType * gpu_ ## _NameVar ## _pr; \
										 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
										 PRINTF("Allocating in GPU for " #_NameVar " %lu elem.\n",(_PT)SizeCopy);\
										 mxcheckGPUErrors(cudaMalloc((void **)&gpu_ ## _NameVar ##_pr,SizeCopy*sizeof(_dataType))); \
										 NumberAlloc++;\
										 mxcheckGPUErrors(cudaMemcpy(gpu_ ## _NameVar ## _pr, _NameVar ## _pr, SizeCopy*sizeof(_dataType), cudaMemcpyHostToDevice));

#define CopyFromGPUToMX(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar ## _res)*INHOST(ZoneCount); \
										 mxcheckGPUErrors(cudaMemcpy( _NameVar ## _pr, gpu_ ## _NameVar ## _pr, SizeCopy*sizeof(_dataType), cudaMemcpyDeviceToHost));

#define CopyFromGPUToMXAsync(_NameVar,_dataType,_stream) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar ## _res)*INHOST(ZoneCount); \
										 mxcheckGPUErrors(cudaMemcpyAsync( _NameVar ## _pr, gpu_ ## _NameVar ## _pr, SizeCopy*sizeof(_dataType), cudaMemcpyDeviceToHost,_stream));

#define CopyFromGPUToMX3(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
										 mxcheckGPUErrors(cudaMemcpy( _NameVar ## _pr, gpu_ ## _NameVar ## _pr, SizeCopy*sizeof(_dataType), cudaMemcpyDeviceToHost));

#define ownGPUFree(_NameVar) 	PRINTF("Releasing GPU memory for " #_NameVar "\n"); \
								 mxcheckGPUErrors(cudaFree(gpu_ ## _NameVar ##_pr)); 	\
								 NumberAlloc--;

#define InParamP(_NameVar) pHost._NameVar ## _pr =  gpu_ ## _NameVar ## _pr;

  struct InputDataKernel
  {
  	mexType
  	*V_x_x_pr,
      *V_y_x_pr,
      *V_z_x_pr,
      *V_x_y_pr,
      *V_y_y_pr,
      *V_z_y_pr,
      *V_x_z_pr,
      *V_y_z_pr,
      *V_z_z_pr,
      *Vx_pr,
      *Vy_pr,
      *Vz_pr,
      *Rxx_pr,
      *Ryy_pr,
      *Rzz_pr,
      *Rxy_pr,
      *Rxz_pr,
      *Ryz_pr,
	  *Sigma_x_xx_pr,
      *Sigma_y_xx_pr,
      *Sigma_z_xx_pr,
      *Sigma_x_yy_pr,
      *Sigma_y_yy_pr,
      *Sigma_z_yy_pr,
      *Sigma_x_zz_pr,
      *Sigma_y_zz_pr,
      *Sigma_z_zz_pr,
      *Sigma_x_xy_pr,
      *Sigma_y_xy_pr,
      *Sigma_x_xz_pr,
      *Sigma_z_xz_pr,
      *Sigma_y_yz_pr,
      *Sigma_z_yz_pr,
      *Sigma_xy_pr,
      *Sigma_xz_pr,
      *Sigma_yz_pr,
      *Sigma_xx_pr,
      *Sigma_yy_pr,
      *Sigma_zz_pr,
	  *Pressure_pr,
	  *SourceFunctions_pr,
      * LambdaMiuMatOverH_pr,
      * LambdaMatOverH_pr,
      * MiuMatOverH_pr,
      * TauLong_pr,
      * OneOverTauSigma_pr,
      * TauShear_pr,
      * InvRhoMatH_pr,
      * SqrAcc_pr,
	  * SensorOutput_pr;
	unsigned int * MaterialMap_pr,
      * SourceMap_pr;
	mexType * Ox_pr,
		* Oy_pr,
		* Oz_pr;
  };

#endif
//-------------------
#ifdef OPENCL
//OPENCL
#define mxcheckGPUErrors(val)           mxcheck ( (val), #val, __FILE__, __LINE__ )

#define InitSymbol(_NameVar,_datatype,_gtype) \
			{ }

#define InitSymbolArray(_NameVar,_gtype,__Limit)\
			{}


#define ownGpuCalloc(_NameVar,_dataType,_size) cl_mem  gpu_ ## _NameVar ## _pr; \
			PRINTF("Allocating in GPU for " #_NameVar " %lu elem. (nZones=%i)\n",(_PT)_size,INHOST(ZoneCount));\
      { \
    		_dataType * temp_zeros_pr = (_dataType *) calloc(_size*INHOST(ZoneCount),sizeof(_dataType)); \
        gpu_ ## _NameVar ##_pr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,\
                  _size*sizeof(_dataType)*INHOST(ZoneCount), temp_zeros_pr, &err);\
        mxcheckGPUErrors(err);\
				free(temp_zeros_pr);\
			};\
      NumberAlloc++;


#define CreateAndCopyFromMXVarOnGPU2(_NameVar,_dataType) SizeCopy =GET_NUMBER_ELEMS(_NameVar); \
			 PRINTF("Allocating in GPU for " #_NameVar " %lu elem.\n",(_PT)SizeCopy);\
       gpu_ ## _NameVar ##_pr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,\
                 SizeCopy*sizeof(_dataType), _NameVar ## _pr, &err);\
       mxcheckGPUErrors(err);\
			 NumberAlloc++;

#define CreateAndCopyFromMXVarOnGPU(_NameVar,_dataType) cl_mem gpu_ ## _NameVar ## _pr; \
			 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
			 PRINTF("Allocating in GPU for " #_NameVar " %lu elem.\n",(_PT)SizeCopy);\
       gpu_ ## _NameVar ##_pr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,\
                 SizeCopy*sizeof(_dataType), _NameVar ## _pr, &err);\
       mxcheckGPUErrors(err);\
       NumberAlloc++;

#define CopyFromGPUToMX(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar ##_res)*INHOST(ZoneCount); \
       mxcheckGPUErrors(clEnqueueReadBuffer( commands, gpu_ ## _NameVar ## _pr, CL_TRUE, 0, SizeCopy*sizeof(_dataType),  _NameVar ## _pr, 0, NULL, NULL ));

#define CopyFromGPUToMX3(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
			        mxcheckGPUErrors(clEnqueueReadBuffer( commands, gpu_ ## _NameVar ## _pr, CL_TRUE, 0, SizeCopy*sizeof(_dataType),  _NameVar ## _pr, 0, NULL, NULL ));


#define ownGPUFree(_NameVar) 	PRINTF("Releasing GPU memory for " #_NameVar "\n"); \
        mxcheckGPUErrors(clReleaseMemObject(gpu_ ## _NameVar ##_pr));\
        NumberAlloc--;

//#this has to contain the same # of entries as in InputDataKernel when using CUDA
    #define InputDataKernel(_NameVar) _IndexDataKernel("" #_NameVar "")
     int  _IndexDataKernel(const char * NameVar)
        {
			if (strcmp(NameVar,"V_x_x")==0) return 0;
			if (strcmp(NameVar,"V_y_x")==0) return 1;
			if (strcmp(NameVar,"V_z_x")==0) return 2;
			if (strcmp(NameVar,"V_x_y")==0) return 3;
			if (strcmp(NameVar,"V_y_y")==0) return 4;
			if (strcmp(NameVar,"V_z_y")==0) return 5;
			if (strcmp(NameVar,"V_x_z")==0) return 6;
			if (strcmp(NameVar,"V_y_z")==0) return 7;
			if (strcmp(NameVar,"V_z_z")==0) return 8;
			if (strcmp(NameVar,"Vx")==0)    return 9;
			if (strcmp(NameVar,"Vy")==0)	return 10;
			if (strcmp(NameVar,"Vz")==0)	return 11;
			if (strcmp(NameVar,"Rxx")==0)	return 12;
			if (strcmp(NameVar,"Ryy")==0)	return 13;
			if (strcmp(NameVar,"Rzz")==0)	return 14;
			if (strcmp(NameVar,"Rxy")==0)	return 15;
			if (strcmp(NameVar,"Rxz")==0)	return 16;
			if (strcmp(NameVar,"Ryz")==0)	return 17;
			if (strcmp(NameVar,"Sigma_x_xx")==0) return 18;
			if (strcmp(NameVar,"Sigma_y_xx")==0)	return 19;
			if (strcmp(NameVar,"Sigma_z_xx")==0)	return 20;
			if (strcmp(NameVar,"Sigma_x_yy")==0)	return 21;
			if (strcmp(NameVar,"Sigma_y_yy")==0)	return 22;
			if (strcmp(NameVar,"Sigma_z_yy")==0)	return 23;
			if (strcmp(NameVar,"Sigma_x_zz")==0)	return 24;
			if (strcmp(NameVar,"Sigma_y_zz")==0)	return 25;
			if (strcmp(NameVar,"Sigma_z_zz")==0)	return 26;
			if (strcmp(NameVar,"Sigma_x_xy")==0)	return 27;
			if (strcmp(NameVar,"Sigma_y_xy")==0)	return 28;
			if (strcmp(NameVar,"Sigma_x_xz")==0)	return 29;
			if (strcmp(NameVar,"Sigma_z_xz")==0)	return 30;
			if (strcmp(NameVar,"Sigma_y_yz")==0)	return 31;
			if (strcmp(NameVar,"Sigma_z_yz")==0)	return 32;
			if (strcmp(NameVar,"Sigma_xy")==0)		return 33;
			if (strcmp(NameVar,"Sigma_xz")==0)		return 34;
			if (strcmp(NameVar,"Sigma_yz")==0)		return 35;
			if (strcmp(NameVar,"Sigma_xx")==0)		return 36;
			if (strcmp(NameVar,"Sigma_yy")==0)		return 37;
			if (strcmp(NameVar,"Sigma_zz")==0)		return 38;
			if (strcmp(NameVar,"SourceFunctions")==0) return 39;
			if (strcmp(NameVar,"LambdaMiuMatOverH")==0) return 40;
			if (strcmp(NameVar,"LambdaMatOverH")==0)	return 41;
			if (strcmp(NameVar,"MiuMatOverH")==0)	return 42;
			if (strcmp(NameVar,"TauLong")==0)	return 43;
			if (strcmp(NameVar,"OneOverTauSigma")==0)	return 44;
			if (strcmp(NameVar,"TauShear")==0)	return 45;
			if (strcmp(NameVar,"InvRhoMatH")==0)	return 46;
			if (strcmp(NameVar,"SqrAcc")==0)	return 47;
			if (strcmp(NameVar,"MaterialMap")==0) return 48;
			if (strcmp(NameVar,"SourceMap")==0)	return 49;
			if (strcmp(NameVar,"Ox")==0) return 50;
			if (strcmp(NameVar,"Oy")==0) return 51;
			if (strcmp(NameVar,"Oz")==0) return 52;
			if (strcmp(NameVar,"Pressure")==0) return 53;
			ERROR_STRING("Unknown parameter");
				return -1;
        };

#define InParamP(_NameVar) {int __NParam = _IndexDataKernel(#_NameVar);\
			mxcheckGPUErrors(clSetKernelArg(StressKernel, __NParam, sizeof(cl_mem), &gpu_ ## _NameVar ## _pr));\
			mxcheckGPUErrors(clSetKernelArg(ParticleKernel, __NParam, sizeof(cl_mem), &gpu_ ## _NameVar ## _pr));\
			mxcheckGPUErrors(clSetKernelArg(SensorsKernel, __NParam, sizeof(cl_mem), &gpu_ ## _NameVar ## _pr));}

int output_device_info(cl_device_id device_id)
			{
			                             // error code returned from OpenCL calls
			    cl_device_type device_type;         // Parameter defining the type of the compute device
			    cl_uint comp_units;                 // the max number of compute units on a device
			    cl_char vendor_name[1024] = {0};    // string to hold vendor name for compute device
			    cl_char device_name[1024] = {0};    // string to hold name of compute device

			    cl_uint          max_work_itm_dims;
			    size_t           max_wrkgrp_size;
			    size_t          *max_loc_size;

			    mxcheckGPUErrors(clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL));
			    PRINTF(" \n Device is  %s ",device_name);

			    mxcheckGPUErrors(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL));
			    if(device_type  == CL_DEVICE_TYPE_GPU)
					{
			       PRINTF(" GPU from ");
					}
			    else if (device_type == CL_DEVICE_TYPE_GPU)
			    {
						PRINTF("\n CPU from ");
					}
			    else
					{
			       PRINTF("\n non  CPU or GPU processor from ");
					}

			    mxcheckGPUErrors(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL));
			    PRINTF(" %s ",vendor_name);

			    mxcheckGPUErrors(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL));
			    PRINTF(" with a max of %d compute units \n",comp_units);


			//
			// Optionally print information about work group sizes
			//
			    mxcheckGPUErrors(clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
			                               &max_work_itm_dims, NULL));
			    max_loc_size = (size_t*)malloc(max_work_itm_dims * sizeof(size_t));
			    if(max_loc_size == NULL){
			       ERROR_STRING("malloc failed");
			    }
			    mxcheckGPUErrors(clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_itm_dims* sizeof(size_t),
			                               max_loc_size, NULL));
			    mxcheckGPUErrors(clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
			                               &max_wrkgrp_size, NULL));
			   PRINTF("work group, work item information");
			   PRINTF("\n max loc dim ");
			   for(unsigned int i=0; i< max_work_itm_dims; i++)
				 {
			     PRINTF(" %d ",(int)(*(max_loc_size+i)));
				 }
			   PRINTF("\n");
			   PRINTF(" Max work group size = %d\n",(int)max_wrkgrp_size);
			   return CL_SUCCESS;

			}
#endif

#if defined(OPENCL) || defined(METAL)
char* load_file(char const* path)
{
		char* buffer = 0;
		long length;
		FILE * f = fopen (path, "rb"); //was "rb"

		if (f)
		{
			fseek (f, 0, SEEK_END);
			length = ftell (f);
			fseek (f, 0, SEEK_SET);
			buffer = (char*)malloc ((length+1)*sizeof(char));
			if (buffer)
			{
				fread (buffer, sizeof(char), length, f);
			}
			fclose (f);
			buffer[length] = '\0';
		}
		return buffer;
}
#endif

#ifdef METAL
 
#define GET_KERNEL_STRESS_FUNCTION(__ID__)\
    mtlpp::Function __ID__ ##_StressKernelFunc = library.NewFunction(#__ID__ "_StressKernel");\
    mxcheckGPUErrors(((int)__ID__ ##_StressKernelFunc));\
    mtlpp::ComputePipelineState __ID__ ##_computePipelineStateStress = device.NewComputePipelineState(__ID__ ##_StressKernelFunc, nullptr);\
    mxcheckGPUErrors(((int)__ID__ ##_computePipelineStateStress));

#define GET_KERNEL_PARTICLE_FUNCTION(__ID__)\
    mtlpp::Function __ID__ ##_ParticleKernelFunc = library.NewFunction(#__ID__ "_ParticleKernel");\
    mxcheckGPUErrors(((int)__ID__ ##_ParticleKernelFunc));\
    mtlpp::ComputePipelineState __ID__ ##_computePipelineStateParticle = device.NewComputePipelineState(__ID__ ##_ParticleKernelFunc, nullptr);\
    mxcheckGPUErrors(((int)__ID__ ##_computePipelineStateParticle));


#define SET_USER_LOCAL_STRESS(__ID__)\
      __ID__ ##_local_stress[0]=(size_t)ManualLocalSize_pr[0];\
      __ID__ ##_local_stress[1]=(size_t)ManualLocalSize_pr[1];\
      __ID__ ##_local_stress[2]=(size_t)ManualLocalSize_pr[2];

#define CALC_USER_LOCAL_STRESS(__ID__)\
{\
      unsigned int w = __ID__ ##_computePipelineStateStress.GetThreadExecutionWidth();\
      unsigned int h = __ID__ ##_computePipelineStateStress.GetMaxTotalThreadsPerThreadgroup() / w;\
      unsigned int z =1;\
      if (h%2==0)\
      {\
        h=h/2;\
        z=2;\
      }\
      __ID__ ##_local_stress[0]=w;\
      __ID__ ##_local_stress[1]=h;\
      __ID__ ##_local_stress[2]=z;\
      PRINTF(#__ID__ "_local_stress =[%i,%i,%i]\n",__ID__ ##_local_stress[0],__ID__ ##_local_stress[1],__ID__ ##_local_stress[2]);\
}

#define SET_USER_LOCAL_PARTICLE(__ID__)\
      __ID__ ##_local_particle[0]=(size_t)ManualLocalSize_pr[0];\
      __ID__ ##_local_particle[1]=(size_t)ManualLocalSize_pr[1];\
      __ID__ ##_local_particle[2]=(size_t)ManualLocalSize_pr[2];

#define CALC_USER_LOCAL_PARTICLE(__ID__)\
{\
      unsigned int w = __ID__ ##_computePipelineStateParticle.GetThreadExecutionWidth();\
      unsigned int h = __ID__ ##_computePipelineStateParticle.GetMaxTotalThreadsPerThreadgroup() / w;\
      unsigned int z =1;\
      if (h%2==0)\
      {\
        h=h/2;\
        z=2;\
      }\
      __ID__ ##_local_particle[0]=w;\
      __ID__ ##_local_particle[1]=h;\
      __ID__ ##_local_particle[2]=z;\
      PRINTF(#__ID__ "_local_particle =[%i,%i,%i]\n",__ID__ ##_local_particle[0],__ID__ ##_local_particle[1],__ID__ ##_local_particle[2]);\
}

#define SET_USER_GROUP_STRESS(__ID__)\
      __ID__ ##_global_stress[0]=(size_t)ManualGroupSize_pr[0];\
      __ID__ ##_global_stress[1]=(size_t)ManualGroupSize_pr[1];\
      __ID__ ##_global_stress[2]=(size_t)ManualGroupSize_pr[2];

#define SET_USER_GROUP_PARTICLE(__ID__)\
      __ID__ ##_global_particle[0]=(size_t)ManualGroupSize_pr[0];\
      __ID__ ##_global_particle[1]=(size_t)ManualGroupSize_pr[1];\
      __ID__ ##_global_particle[2]=(size_t)ManualGroupSize_pr[2];

#define CALC_USER_GROUP_STRESS_MAIN(__ID__)\
      __ID__ ##_global_stress[0]=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) __ID__ ##_local_stress[0]);\
      __ID__ ##_global_stress[1]=(unsigned int)ceil((float)(INHOST(N2)-INHOST(PML_Thickness)*2) / (float) __ID__ ##_local_stress[1]);\
      __ID__ ##_global_stress[2]=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) __ID__ ##_local_stress[2]);\
      PRINTF(#__ID__ "_global_stress =[%i,%i,%i]\n",__ID__ ##_global_stress[0],__ID__ ##_global_stress[1],__ID__ ##_global_stress[2]);


#define CALC_USER_GROUP_PARTICLE_MAIN(__ID__)\
      __ID__ ##_global_particle[0]=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) __ID__ ##_local_particle[0]);\
      __ID__ ##_global_particle[1]=(unsigned int)ceil((float)(INHOST(N2)-INHOST(PML_Thickness)*2) / (float) __ID__ ##_local_particle[1]);\
      __ID__ ##_global_particle[2]=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) __ID__ ##_local_particle[2]);\
      PRINTF(#__ID__ "_global_particle =[%i,%i,%i]\n",__ID__ ##_global_particle[0],__ID__ ##_global_particle[1],__ID__ ##_global_particle[2]);
#define CALC_USER_GROUP_PML(__TYPE__)\
  PML_1_global_## __TYPE__[0]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_1_local_## __TYPE__[0]);\
  PML_1_global_## __TYPE__[1]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_1_local_## __TYPE__[1]);\
  PML_1_global_## __TYPE__[2]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_1_local_## __TYPE__[2]);\
  \
  PML_2_global_## __TYPE__[0]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_2_local_## __TYPE__[0]);\
  PML_2_global_## __TYPE__[1]=(unsigned int)ceil((float)(INHOST(N2)-INHOST(PML_Thickness)*2) / (float) PML_2_local_## __TYPE__[1]);\
  PML_2_global_## __TYPE__[2]=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) PML_2_local_## __TYPE__[2]);\
  \
  PML_3_global_## __TYPE__[0]=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) PML_3_local_## __TYPE__[0]);\
  PML_3_global_## __TYPE__[1]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_3_local_## __TYPE__[1]);\
  PML_3_global_## __TYPE__[2]=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) PML_3_local_## __TYPE__[2]);\
  \
  PML_4_global_## __TYPE__[0]=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) PML_4_local_## __TYPE__[0]);\
  PML_4_global_## __TYPE__[1]=(unsigned int)ceil((float)(INHOST(N2)-INHOST(PML_Thickness)*2) / (float) PML_4_local_## __TYPE__[1]);\
  PML_4_global_## __TYPE__[2]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_4_local_## __TYPE__[2]);\
  \
  PML_5_global_## __TYPE__[0]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_5_local_## __TYPE__[0]);\
  PML_5_global_## __TYPE__[1]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_5_local_## __TYPE__[1]);\
  PML_5_global_## __TYPE__[2]=(unsigned int)ceil((float)(INHOST(N3)-INHOST(PML_Thickness)*2) / (float) PML_5_local_## __TYPE__[2]);\
  \
  PML_6_global_## __TYPE__[0]=(unsigned int)ceil((float)(INHOST(N1)-INHOST(PML_Thickness)*2) / (float) PML_6_local_## __TYPE__[0]);\
  PML_6_global_## __TYPE__[1]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_6_local_## __TYPE__[1]);\
  PML_6_global_## __TYPE__[2]=(unsigned int)ceil((float)(INHOST(PML_Thickness)*2) / (float) PML_6_local_## __TYPE__[2]);\
  \
  PRINTF("PML_1_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",PML_1_global_## __TYPE__[0],PML_1_global_## __TYPE__[1],PML_1_global_## __TYPE__[2],\
    PML_1_global_## __TYPE__[0]*PML_1_local_## __TYPE__[0],PML_1_global_## __TYPE__[1]*PML_1_local_## __TYPE__[1],PML_1_global_## __TYPE__[2]*PML_1_local_## __TYPE__[2]);\
  PRINTF("PML_2_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",PML_2_global_## __TYPE__[0],PML_2_global_## __TYPE__[1],PML_2_global_## __TYPE__[2],\
    PML_2_global_## __TYPE__[0]*PML_2_local_## __TYPE__[0],PML_2_global_## __TYPE__[1]*PML_2_local_## __TYPE__[1],PML_2_global_## __TYPE__[2]*PML_2_local_## __TYPE__[2]);\
  PRINTF("PML_3_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",PML_3_global_## __TYPE__[0],PML_3_global_## __TYPE__[1],PML_3_global_## __TYPE__[2],\
    PML_3_global_## __TYPE__[0]*PML_3_local_## __TYPE__[0],PML_3_global_## __TYPE__[1]*PML_3_local_## __TYPE__[1],PML_3_global_## __TYPE__[2]*PML_3_local_## __TYPE__[2]);\
  PRINTF("PML_4_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",PML_4_global_## __TYPE__[0],PML_4_global_## __TYPE__[1],PML_4_global_## __TYPE__[2],\
    PML_4_global_## __TYPE__[0]*PML_4_local_## __TYPE__[0],PML_4_global_## __TYPE__[1]*PML_4_local_## __TYPE__[1],PML_4_global_## __TYPE__[2]*PML_4_local_## __TYPE__[2]);\
  PRINTF("PML_5_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",PML_5_global_## __TYPE__[0],PML_5_global_## __TYPE__[1],PML_5_global_## __TYPE__[2],\
    PML_5_global_## __TYPE__[0]*PML_5_local_## __TYPE__[0],PML_5_global_## __TYPE__[1]*PML_5_local_## __TYPE__[1],PML_5_global_## __TYPE__[2]*PML_5_local_## __TYPE__[2]);\
  PRINTF("PML_6_global_" #__TYPE__ "=[%i,%i,%i],[%i,%i,%i]\n",PML_6_global_## __TYPE__[0],PML_6_global_## __TYPE__[1],PML_6_global_## __TYPE__[2],\
    PML_6_global_## __TYPE__[0]*PML_6_local_## __TYPE__[0],PML_6_global_## __TYPE__[1]*PML_6_local_## __TYPE__[1],PML_6_global_## __TYPE__[2]*PML_6_local_## __TYPE__[2]);
 
  #define ENCODE_STRESS(__ID__)\
        mtlpp::ComputeCommandEncoder __ID__ ##StressEncoder = StresscommandBuffer.ComputeCommandEncoder();\
        mxcheckGPUErrors(((int)__ID__ ##StressEncoder));\
        __ID__ ##StressEncoder.SetBuffer(_CONSTANT_BUFFER_UINT, 0, 0);\
        __ID__ ##StressEncoder.SetBuffer(_CONSTANT_BUFFER_MEX, 0, 1);\
        __ID__ ##StressEncoder.SetBuffer(_INDEX_MEX, 0, 2);\
        __ID__ ##StressEncoder.SetBuffer(_INDEX_UINT, 0, 3);\
        __ID__ ##StressEncoder.SetBuffer(_UINT_BUFFER, 0, 4);\
        for (_PT ii=0;ii<12;ii++)\
            __ID__ ##StressEncoder.SetBuffer(_MEX_BUFFER[ii], 0, 5+ii);\
        __ID__ ##StressEncoder.SetComputePipelineState(__ID__ ##_computePipelineStateStress);\
        __ID__ ##StressEncoder.DispatchThreadgroups(\
            mtlpp::Size(\
              __ID__ ##_global_stress[0],\
              __ID__ ##_global_stress[1],\
              __ID__ ##_global_stress[2]),\
            mtlpp::Size(\
              __ID__ ##_local_stress[0],\
              __ID__ ##_local_stress[1],\
              __ID__ ##_local_stress[2]));\
        __ID__ ##StressEncoder.EndEncoding();

#define ENCODE_PARTICLE(__ID__)\
        mtlpp::ComputeCommandEncoder __ID__ ##ParticleEncoder = StresscommandBuffer.ComputeCommandEncoder();\
        mxcheckGPUErrors(((int)__ID__ ##ParticleEncoder));\
        __ID__ ##ParticleEncoder.SetBuffer(_CONSTANT_BUFFER_UINT, 0, 0);\
        __ID__ ##ParticleEncoder.SetBuffer(_CONSTANT_BUFFER_MEX, 0, 1);\
        __ID__ ##ParticleEncoder.SetBuffer(_INDEX_MEX, 0, 2);\
        __ID__ ##ParticleEncoder.SetBuffer(_INDEX_UINT, 0, 3);\
        __ID__ ##ParticleEncoder.SetBuffer(_UINT_BUFFER, 0, 4);\
        for (_PT ii=0;ii<12;ii++)\
            __ID__ ##ParticleEncoder.SetBuffer(_MEX_BUFFER[ii], 0, 5+ii);\
        __ID__ ##ParticleEncoder.SetComputePipelineState(__ID__ ##_computePipelineStateParticle);\
        __ID__ ##ParticleEncoder.DispatchThreadgroups(\
            mtlpp::Size(\
              __ID__ ##_global_particle[0],\
              __ID__ ##_global_particle[1],\
              __ID__ ##_global_particle[2]),\
            mtlpp::Size(\
              __ID__ ##_local_particle[0],\
              __ID__ ##_local_particle[1],\
              __ID__ ##_local_particle[2]));\
        __ID__ ##ParticleEncoder.EndEncoding();


#define mxcheckGPUErrors(val)           mxcheck ( (val), #val, __FILE__, __LINE__ )
//We define first the indexes for uint const values

#define InitSymbol(_NameVar,_datatype,_gtype)\
{\
	if (_gtype==G_INT)\
	{\
			_datatype * inData = static_cast<_datatype *>(_CONSTANT_BUFFER_UINT.GetContents());\
			inData[CInd_ ## _NameVar] = INHOST(_NameVar);\
			_CONSTANT_BUFFER_UINT.DidModify(ns::Range((CInd_ ## _NameVar)*sizeof(_datatype), sizeof(_datatype)));\
	}\
	else\
	{\
	 	_datatype * inData = static_cast<_datatype *>(_CONSTANT_BUFFER_MEX.GetContents());\
		inData[CInd_ ## _NameVar] = INHOST(_NameVar);\
		_CONSTANT_BUFFER_MEX.DidModify(ns::Range((CInd_ ## _NameVar)*sizeof(_datatype), sizeof(_datatype)));\
	}\
}

#define InitSymbolArray(_NameVar,_gtype,__Limit)\
if (_gtype==G_INT)\
{\
		unsigned int * inData = static_cast<unsigned int *>(_CONSTANT_BUFFER_UINT.GetContents());\
		for (unsigned int _n=0;_n<__Limit;_n++)\
		{\
			inData[_n + CInd_ ## _NameVar] = _NameVar ## _pr[_n];\
		}\
}\
else\
{\
	mexType * inData = static_cast<mexType *>(_CONSTANT_BUFFER_MEX.GetContents());\
	for (unsigned int _n=0;_n<__Limit;_n++)\
	{\
		inData[_n + CInd_ ## _NameVar] = _NameVar ## _pr[_n];\
	}\
}\

const _PT  _IndexDataMetal(const char * NameVar)
        {
			if (strcmp(NameVar,"V_x_x")==0) return 0;
			if (strcmp(NameVar,"V_y_x")==0) return 0;
			if (strcmp(NameVar,"V_z_x")==0) return 0;
			if (strcmp(NameVar,"V_x_y")==0) return 0;
			if (strcmp(NameVar,"V_y_y")==0) return 0;
			if (strcmp(NameVar,"V_z_y")==0) return 0;
			if (strcmp(NameVar,"V_x_z")==0) return 0;
			if (strcmp(NameVar,"V_y_z")==0) return 0;
			if (strcmp(NameVar,"V_z_z")==0) return 0;

			if (strcmp(NameVar,"Vx")==0)    return 1;
			if (strcmp(NameVar,"Vy")==0)	return 1;
			if (strcmp(NameVar,"Vz")==0)	return 1;

			if (strcmp(NameVar,"Rxx")==0)	return 2;
			if (strcmp(NameVar,"Ryy")==0)	return 2;
			if (strcmp(NameVar,"Rzz")==0)	return 2;

			if (strcmp(NameVar,"Rxy")==0)	return 3;
			if (strcmp(NameVar,"Rxz")==0)	return 3;
			if (strcmp(NameVar,"Ryz")==0)	return 3;

			if (strcmp(NameVar,"Sigma_x_xx")==0) return 4;
			if (strcmp(NameVar,"Sigma_y_xx")==0)	return 4;
			if (strcmp(NameVar,"Sigma_z_xx")==0)	return 4;
			if (strcmp(NameVar,"Sigma_x_yy")==0)	return 4;
			if (strcmp(NameVar,"Sigma_y_yy")==0)	return 4;
			if (strcmp(NameVar,"Sigma_z_yy")==0)	return 4;
			if (strcmp(NameVar,"Sigma_x_zz")==0)	return 4;
			if (strcmp(NameVar,"Sigma_y_zz")==0)	return 4;

			if (strcmp(NameVar,"Sigma_z_zz")==0)	return 5;
			if (strcmp(NameVar,"Sigma_x_xy")==0)	return 5;
			if (strcmp(NameVar,"Sigma_y_xy")==0)	return 5;
			if (strcmp(NameVar,"Sigma_x_xz")==0)	return 5;
			if (strcmp(NameVar,"Sigma_z_xz")==0)	return 5;
			if (strcmp(NameVar,"Sigma_y_yz")==0)	return 5;
			if (strcmp(NameVar,"Sigma_z_yz")==0)	return 5;

			if (strcmp(NameVar,"Sigma_xy")==0)		return 6;
			if (strcmp(NameVar,"Sigma_xz")==0)		return 6;
			if (strcmp(NameVar,"Sigma_yz")==0)		return 6;
			
			if (strcmp(NameVar,"Sigma_xx")==0)		return 7;
			if (strcmp(NameVar,"Sigma_yy")==0)		return 7;
			if (strcmp(NameVar,"Sigma_zz")==0)		return 7;

			if (strcmp(NameVar,"SourceFunctions")==0) return 8;

			if (strcmp(NameVar,"LambdaMiuMatOverH")==0) return 9;
			if (strcmp(NameVar,"LambdaMatOverH")==0)	return 9;
			if (strcmp(NameVar,"MiuMatOverH")==0)	return 9;
			if (strcmp(NameVar,"TauLong")==0)	return 9;
			if (strcmp(NameVar,"OneOverTauSigma")==0)	return 9;
			if (strcmp(NameVar,"TauShear")==0)	return 9;
			if (strcmp(NameVar,"InvRhoMatH")==0)	return 9;
			if (strcmp(NameVar,"Ox")==0) return 9;
			if (strcmp(NameVar,"Oy")==0) return 9;
			if (strcmp(NameVar,"Oz")==0) return 9;
			if (strcmp(NameVar,"Pressure")==0) return 9;

			if (strcmp(NameVar,"SqrAcc")==0)	return 10;

			if (strcmp(NameVar,"SensorOutput")==0)	return 11;
					
			
			ERROR_STRING("Unknown parameter");
				return -1;
        };

#define ownGpuCalloc(_NameVar,_dataType,_size)\
	PRINTF("Allocating in GPU for " #_NameVar " %lu elem. (nZones=%i)\n",(_PT)_size*INHOST(ZoneCount),(int)INHOST(ZoneCount));\
	if (NULL!=strstr(#_dataType,"mexType"))\
	{	\
		_PT subArray = _IndexDataMetal(#_NameVar);\
		HOST_INDEX_MEX[CInd_ ## _NameVar][0]=_c_mex_type[subArray];\
		HOST_INDEX_MEX[CInd_ ## _NameVar][1]=_size*INHOST(ZoneCount);\
		_c_mex_type[subArray]+=_size*INHOST(ZoneCount);\
	} \
	else\
	{\
		HOST_INDEX_UINT[CInd_ ## _NameVar][0]=_c_uint_type;\
		HOST_INDEX_UINT[CInd_ ## _NameVar][1]=_size*INHOST(ZoneCount);\
	 _c_uint_type+=_size*INHOST(ZoneCount);\
 }

 #define CreateAndCopyFromMXVarOnGPU(_NameVar,_dataType) \
 				 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
				 PRINTF("Allocating in GPU for " #_NameVar " %lu elem.\n",(_PT)SizeCopy);\
				 if (NULL!=strstr(#_dataType,"mexType"))\
			 	{	\
				 	_PT subArray = _IndexDataMetal(#_NameVar);\
			 		HOST_INDEX_MEX[CInd_ ## _NameVar][0]=_c_mex_type[subArray];\
			 		HOST_INDEX_MEX[CInd_ ## _NameVar][1]=SizeCopy;\
			 		_c_mex_type[subArray]+=SizeCopy;\
			 	} \
			 	else\
			 	{\
			 		HOST_INDEX_UINT[CInd_ ## _NameVar][0]=_c_uint_type;\
			 		HOST_INDEX_UINT[CInd_ ## _NameVar][1]=SizeCopy;\
			 	 _c_uint_type+=SizeCopy;\
			 	}

#define CreateAndCopyFromMXVarOnGPU2(_NameVar,_dataType) SizeCopy =GET_NUMBER_ELEMS(_NameVar); \
					 PRINTF("Allocating in GPU for " #_NameVar " %lu elem.\n",(_PT)SizeCopy);\
					 gpu_ ## _NameVar ##_pr = device.NewBuffer(sizeof(_dataType) * \
				              SizeCopy,\
				             mtlpp::ResourceOptions::StorageModeManaged);\
				   mxcheckGPUErrors(((int)gpu_ ## _NameVar ##_pr));\
					 {\
					      _dataType * inData = static_cast<_dataType*>(gpu_ ## _NameVar ##_pr.GetContents());\
					      memcpy(inData,_NameVar ## _pr ,sizeof(_dataType) * SizeCopy);\
					      gpu_ ## _NameVar ##_pr.DidModify(ns::Range( 0, sizeof(_dataType) *SizeCopy));\
					  }

#define CopyFromGPUToMX(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar ##_res)*INHOST(ZoneCount); \
		if (NULL!=strstr(#_dataType,"mexType"))\
	 {	\
	 	 _PT subArray = _IndexDataMetal(#_NameVar);\
		 _dataType * inData = static_cast<_dataType*>(_MEX_BUFFER[subArray].GetContents());\
		 memcpy(_NameVar ## _pr,&inData[HOST_INDEX_MEX[CInd_ ##_NameVar][0]],sizeof(_dataType) *SizeCopy );\
	 } \
	 else\
	 {\
		 _dataType * inData = static_cast<_dataType*>(_UINT_BUFFER.GetContents());\
		 memcpy(_NameVar ## _pr,&inData[HOST_INDEX_UINT[CInd_ ##_NameVar][0]],sizeof(_dataType) *SizeCopy );\
	 }

#define CopyFromGPUToMX3(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
		if (NULL!=strstr(#_dataType,"mexType"))\
	 {	\
		 _dataType * inData = static_cast<_dataType*>(gpu_ ## _NameVar ## _pr.GetContents());\
		 memcpy(_NameVar ## _pr,inData,sizeof(_dataType) *SizeCopy );\
	 } \
	 else\
	 {\
		 _dataType * inData = static_cast<_dataType*>(gpu_ ## _NameVar ## _pr.GetContents());\
		 memcpy(_NameVar ## _pr,inData,sizeof(_dataType) *SizeCopy );\
	 }


 #define CopyFromGPUToMX4(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
	 		if (NULL!=strstr(#_dataType,"mexType"))\
	 	 {	\
		     _PT subArray = _IndexDataMetal(#_NameVar);\
	 		 _dataType * inData = static_cast<_dataType*>(_MEX_BUFFER[subArray].GetContents());\
			 memcpy(_NameVar ## _pr,&inData[HOST_INDEX_MEX[CInd_ ##_NameVar][0]],sizeof(_dataType) *SizeCopy );\
			  } \
	 	 else\
	 	 {\
	 		 _dataType * inData = static_cast<_dataType*>(_UINT_BUFFER.GetContents());\
			 memcpy(_NameVar ## _pr,&inData[HOST_INDEX_UINT[CInd_ ##_NameVar][0]],sizeof(_dataType) *SizeCopy );\
			 }

	 // METAL is c++ based and their own clasess release the memory

		#define ownGPUFree(_NameVar) { }

		#define InParamP(_NameVar) { }

		#define CompleteCopyToGpu(_NameVar,_dataType) 	 SizeCopy = GET_NUMBER_ELEMS(_NameVar); \
		if (NULL!=strstr(#_dataType,"mexType"))\
	 {	\
	 	_PT subArray = _IndexDataMetal(#_NameVar);\
		 _dataType * inData = static_cast<_dataType*>(_MEX_BUFFER[subArray].GetContents());\
		 memcpy(&inData[HOST_INDEX_MEX[CInd_ ##_NameVar][0]],_NameVar ## _pr,sizeof(_dataType) *SizeCopy );\
	 } \
	 else\
	 {\
		 _dataType * inData = static_cast<_dataType*>(_UINT_BUFFER.GetContents());\
		 memcpy(&inData[HOST_INDEX_UINT[CInd_ ##_NameVar][0]],_NameVar ## _pr,sizeof(_dataType) *SizeCopy );\
	 	 }

#endif

#define CHOOSE_INDEX(_Type) switch(Type ## _Type)\
{\
  case 0:\
    index ## _Type=IndN1N2N3(i,j,k);\
    break;\
  case 1:\
    index ## _Type=IndN1p1N2N3(i,j,k);\
    break;\
  case 2:\
    index ## _Type=IndN1N2p1N3(i,j,k);\
    break;\
  case 3:\
    index ## _Type=IndN1N2N3p1(i,j,k);\
    break;\
  case 4:\
    index ## _Type=IndN1p1N2p1N3p1(i,j,k);\
    break;\
}

#define CHOOSE_INDEX_PML(_Type) switch(Type ## _Type)\
{\
  case 0:\
    index ## _Type=IndexPML(i,j,k);\
    break;\
  case 1:\
    index ## _Type=IndexPMLxp1(i,j,k);\
    break;\
  case 2:\
    index ## _Type=IndexPMLyp1(i,j,k);\
    break;\
  case 3:\
    index ## _Type=IndexPMLzp1(i,j,k);\
    break;\
  case 4:\
    index ## _Type=IndexPMLxp1yp1zp1(i,j,k);\
    break;\
}

#if defined(CUDA)
#include "GPU_KERNELS.h"
#endif

#endif
