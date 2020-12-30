#if defined(CUDA)
__global__ void StressKernel(InputDataKernel *p,unsigned int nStep)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
__kernel void StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep)
{
  const unsigned int i = get_global_id(0);
  const unsigned int j = get_global_id(1);
  const unsigned int k = get_global_id(2);
#endif

    if (i>N1 || j >N2  || k>N3)
		return;

    #include "StressKernel.h"
}

#if defined(CUDA)
__global__ void ParticleKernel(InputDataKernel * p, unsigned int nStep,unsigned int TypeSource)
{
	  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
__kernel void ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
		const unsigned int i = get_global_id(0);
	  const unsigned int j = get_global_id(1);
	  const unsigned int k = get_global_id(2);
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
#else
__kernel void SnapShot(unsigned int SelK,__global mexType * Snapshots_pr,__global mexType * Sigma_xx_pr,__global mexType * Sigma_yy_pr,__global mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
  const unsigned int i = get_global_id(0);
  const unsigned int j = get_global_id(1);
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
__global__ void SensorsKernel(mexType * SensorOutput_pr,mexType * Vx_pr,
														mexType * Vy_pr,mexType * Vz_pr,
														unsigned int * IndexSensorMap_pr,
														unsigned int nStep)
{
	unsigned int sj =blockIdx.x * blockDim.x + threadIdx.x;
#else
__kernel void SensorsKernel(__global mexType * SensorOutput_pr,__global mexType * Vx_pr,
													__global mexType * Vy_pr,__global mexType * Vz_pr,
													__global unsigned int * IndexSensorMap_pr,
													unsigned int nStep)
{
	unsigned int sj =get_global_id(0);
#endif
	if (sj>=	NumberSensors)
		return;

	#include"SensorsParticleKernel.h"

}
