#ifdef METAL
#include"kernelparamsMetal.h"
#endif

#define _ST_PML_1
#ifdef CUDA
__global__ void PML_1_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_1_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_1_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_PML_1

#define _ST_PML_2
#ifdef CUDA
__global__ void PML_2_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_2_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_2_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_PML_2

#define _ST_PML_3
#ifdef CUDA
__global__ void PML_3_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_3_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_3_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_PML_3

#define _ST_PML_4
#ifdef CUDA
__global__ void PML_4_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_4_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_4_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_PML_4

#define _ST_PML_5
#ifdef CUDA
__global__ void PML_5_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_5_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_5_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_PML_5

#define _ST_PML_6
#ifdef CUDA
__global__ void PML_6_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_6_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_6_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_PML_6

#define _ST_MAIN_1
#ifdef CUDA
__global__ void MAIN_1_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_1_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_1_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_MAIN_1

#define _ST_MAIN_2
#ifdef CUDA
__global__ void MAIN_2_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_2_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_2_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_MAIN_2

#define _ST_MAIN_3
#ifdef CUDA
__global__ void MAIN_3_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_3_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_3_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_MAIN_3

#define _ST_MAIN_4
#ifdef CUDA
__global__ void MAIN_4_StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_4_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_4_StressKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])
{
  const _PT i = (_PT) gid.x;
  const _PT j = (_PT) gid.y;
  const _PT k = (_PT) gid.z;
#endif
    if (i>N1 || j >N2  || k>N3)
		return;
    #include "StressKernel.h" 
}
#undef _ST_MAIN_4

#define _PR_PML_1
#if defined(CUDA)
__global__ void PML_1_ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_1_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
	const _PT i = (_PT) get_global_id(0);
	const _PT j = (_PT) get_global_id(1);
	const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_1_ParticleKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])

{
	const _PT i = (_PT) gid.x;
	const _PT j = (_PT) gid.y;
	const _PT k = (_PT) gid.z;
#endif

    if (i>N1 || j >N2  || k>N3)
		return;


	#include "ParticleKernel.h"
}
#undef _PR_PML_1

#define _PR_PML_2
#if defined(CUDA)
__global__ void PML_2_ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_2_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
	const _PT i = (_PT) get_global_id(0);
	const _PT j = (_PT) get_global_id(1);
	const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_2_ParticleKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])

{
	const _PT i = (_PT) gid.x;
	const _PT j = (_PT) gid.y;
	const _PT k = (_PT) gid.z;
#endif

    if (i>N1 || j >N2  || k>N3)
		return;


	#include "ParticleKernel.h"
}
#undef _PR_PML_2

#define _PR_PML_3
#if defined(CUDA)
__global__ void PML_3_ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_3_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
	const _PT i = (_PT) get_global_id(0);
	const _PT j = (_PT) get_global_id(1);
	const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void PML_3_ParticleKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])

{
	const _PT i = (_PT) gid.x;
	const _PT j = (_PT) gid.y;
	const _PT k = (_PT) gid.z;
#endif

    if (i>N1 || j >N2  || k>N3)
		return;


	#include "ParticleKernel.h"
}
#undef _PR_PML_3

#define _PR_MAIN_1
#if defined(CUDA)
__global__ void MAIN_1_ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_1_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
	const _PT i = (_PT) get_global_id(0);
	const _PT j = (_PT) get_global_id(1);
	const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_1_ParticleKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])

{
	const _PT i = (_PT) gid.x;
	const _PT j = (_PT) gid.y;
	const _PT k = (_PT) gid.z;
#endif

    if (i>N1 || j >N2  || k>N3)
		return;


	#include "ParticleKernel.h"
}
#undef _PR_MAIN_1

#define _PR_MAIN_2
#if defined(CUDA)
__global__ void MAIN_2_ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_2_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
	const _PT i = (_PT) get_global_id(0);
	const _PT j = (_PT) get_global_id(1);
	const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_2_ParticleKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])

{
	const _PT i = (_PT) gid.x;
	const _PT j = (_PT) gid.y;
	const _PT k = (_PT) gid.z;
#endif

    if (i>N1 || j >N2  || k>N3)
		return;


	#include "ParticleKernel.h"
}
#undef _PR_MAIN_2

#define _PR_MAIN_3
#if defined(CUDA)
__global__ void MAIN_3_ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_3_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
	const _PT i = (_PT) get_global_id(0);
	const _PT j = (_PT) get_global_id(1);
	const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_3_ParticleKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint3 gid[[thread_position_in_grid]])

{
	const _PT i = (_PT) gid.x;
	const _PT j = (_PT) gid.y;
	const _PT k = (_PT) gid.z;
#endif

    if (i>N1 || j >N2  || k>N3)
		return;


	#include "ParticleKernel.h"
}
#undef _PR_MAIN_3

#if defined(CUDA)
__global__ void SnapShot(unsigned int SelK,mexType * Snapshots_pr,mexType * Sigma_xx_pr,mexType * Sigma_yy_pr,mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
  const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void SnapShot(unsigned int SelK,__global mexType * Snapshots_pr,__global mexType * Sigma_xx_pr,__global mexType * Sigma_yy_pr,__global mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
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
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	device mexType * Snapshots_pr [[ buffer(17) ]],
	uint2 gid[[thread_position_in_grid]])

	{
	const _PT i = (_PT) gid.x;
	const _PT j = (_PT) gid.y;
#endif

    if (i>=N1 || j >=N2)
		return;
	mexType accum=0.0;
	for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
		{
			_PT index=Ind_Sigma_xx(i,j,(_PT)SelK);
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
	_PT sj =(_PT) get_global_id(0);
#endif
#ifdef METAL

#define IndexSensorMap_pr k_IndexSensorMap_pr

kernel void SensorsKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],
	uint gid[[thread_position_in_grid]])
{
	_PT sj = (_PT) gid;
#endif

	if (sj>=(_PT) NumberSensors)
		return;
	#include"SensorsKernel.h"

}