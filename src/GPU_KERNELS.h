#ifdef METAL
#include"kernelparamsMetal.h"
#endif

#if defined(CUDA)
__global__ void StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
#endif
#ifdef OPENCL
__kernel void StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const unsigned int i = get_global_id(0);
  const unsigned int j = get_global_id(1);
  const unsigned int k = get_global_id(2);
#endif
#ifdef METAL
kernel void StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER [[ buffer(5) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const unsigned int i = gid.x;
  const unsigned int j = gid.y;
  const unsigned int k = gid.z;
#endif


    if (i>N1 || j >N2  || k>N3)
		return;

    #include "StressKernel.h"
}

#if defined(CUDA)
__global__ void ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
#endif
#ifdef OPENCL
__kernel void ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
		const unsigned int i = get_global_id(0);
	  const unsigned int j = get_global_id(1);
	  const unsigned int k = get_global_id(2);
#endif
#ifdef METAL
kernel void ParticleKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER [[ buffer(5) ]],
	uint3 gid[[thread_position_in_grid]])

{
	const unsigned int i = gid.x;
	const unsigned int j = gid.y;
	const unsigned int k = gid.z;
#endif

    if (i>N1 || j >N2  || k>N3)
		return;


	#include "ParticleKernel.h"
}


#if defined(CUDA)
__global__ void SnapShot(unsigned int SelK,mexType * Snapshots_pr,mexType * Sigma_xx_pr,mexType * Sigma_yy_pr,mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
#endif
#ifdef OPENCL
__kernel void SnapShot(unsigned int SelK,__global mexType * Snapshots_pr,__global mexType * Sigma_xx_pr,__global mexType * Sigma_yy_pr,__global mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
  const unsigned int i = get_global_id(0);
  const unsigned int j = get_global_id(1);
#endif
#ifdef METAL
#define Sigma_xx_pr k_Sigma_xx_pr
#define Sigma_yy_pr k_Sigma_yy_pr
#define Sigma_zz_pr k_Sigma_zz_pr

kernel void SnapShot(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER [[ buffer(5) ]],
	device mexType * Snapshots_pr [[ buffer(6) ]],
	uint2 gid[[thread_position_in_grid]])

	{
	const unsigned int i = gid.x;
	const unsigned int j = gid.y;
#endif

    if (i>=N1 || j >=N2)
		return;
	mexType accum=0.0;
	for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
		{
			unsigned int index=Ind_Sigma_xx(i,j,SelK);
			accum+=(Sigma_xx_pr[index]+Sigma_yy_pr[index]+Sigma_zz_pr[index])/3.0;

		}

		Snapshots_pr[IndN1N2Snap(i,j)+CurrSnap*N1*N2]=accum/ZoneCount;
}

#if defined(CUDA)
__global__ void SensorsKernel(InputDataKernel * p,
													  unsigned int * IndexSensorMap_pr,
														unsigned int nStep)
{
	unsigned int sj =blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef OPENCL
__kernel void SensorsKernel(
		#include "kernelparamsOpenCL.h"
		, __global mexType * SensorOutput_pr,
			__global unsigned int * IndexSensorMap_pr,
			unsigned int nStep)
{
	unsigned int sj =get_global_id(0);
#endif
#ifdef METAL

#define IndexSensorMap_pr k_IndexSensorMap_pr

kernel void SensorsKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER [[ buffer(5) ]],
	uint gid[[thread_position_in_grid]])
{
	unsigned int sj = gid;
#endif

	if (sj>=	NumberSensors)
		return;
	#include"SensorsKernel.h"

}