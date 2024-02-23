#ifdef METAL
#include"kernelparamsMetal.h"

#ifdef METALCOMPUTE
#define CGID uint
#else
#define CGID uint3
#endif
#ifndef METALCOMPUTE
#define METAL_PARAMS\
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],\
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],\
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],\
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],\
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],\
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],\
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],\
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],\
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],\
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],\
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],\
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],\
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],\
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],\
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],\
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],\
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],\
	CGID gid[[thread_position_in_grid]])\
{
#else
#define METAL_PARAMS\
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],\
	const device unsigned int *p_INDEX_MEX [[ buffer(1) ]],\
	const device unsigned int *p_INDEX_UINT [[ buffer(2) ]],\
	const device unsigned int *p_UINT_BUFFER [[ buffer(3) ]],\
	device mexType * p_MEX_BUFFER_0 [[ buffer(4) ]],\
	device mexType * p_MEX_BUFFER_1 [[ buffer(5) ]],\
	device mexType * p_MEX_BUFFER_2 [[ buffer(6) ]],\
	device mexType * p_MEX_BUFFER_3 [[ buffer(7) ]],\
	device mexType * p_MEX_BUFFER_4 [[ buffer(8) ]],\
	device mexType * p_MEX_BUFFER_5 [[ buffer(9) ]],\
	device mexType * p_MEX_BUFFER_6 [[ buffer(10) ]],\
	device mexType * p_MEX_BUFFER_7 [[ buffer(11) ]],\
	device mexType * p_MEX_BUFFER_8 [[ buffer(12) ]],\
	device mexType * p_MEX_BUFFER_9 [[ buffer(13) ]],\
	device mexType * p_MEX_BUFFER_10 [[ buffer(14) ]],\
	device mexType * p_MEX_BUFFER_11 [[ buffer(15) ]],\
	CGID gid[[thread_position_in_grid]])\
{
#endif
#endif
/// PMLS
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL)) || defined(USE_MINI_KERNELS_CUDA)
#define _ST_PML
#define _PML_KERNEL_I_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_1_StressKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_1_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_1_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else
	#define nN1 (PML_Thickness)
	#define nN2 (N2)
	#define nN3 (N3)
    _PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "StressKernel.h" 
}
#undef _PML_KERNEL_I_BOTTOM

#define _PML_KERNEL_I_TOP
#ifdef CUDA
extern "C" __global__ void PML_2_StressKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_2_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_2_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
	_PT i = (_PT) gid.x;
	_PT j = (_PT) gid.y;
	_PT k = (_PT) gid.z;
	#else
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (PML_Thickness)
	#define nN2 (N2)
	#define nN3 (N3)
    _PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "StressKernel.h" 
}
#undef _PML_KERNEL_I_TOP

#define _PML_KERNEL_J_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_3_StressKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_3_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_3_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
	_PT i = (_PT) gid.x;
	_PT j = (_PT) gid.y;
	_PT k = (_PT) gid.z;
	#else
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (PML_Thickness)
	#define nN3 (N3)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "StressKernel.h" 
}
#undef _PML_KERNEL_J_BOTTOM

#define _PML_KERNEL_J_TOP
#ifdef CUDA
extern "C" __global__ void PML_4_StressKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_4_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_4_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (PML_Thickness)
	#define nN3 (N3)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "StressKernel.h" 
}
#undef _PML_KERNEL_J_TOP

#define _PML_KERNEL_K_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_5_StressKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_5_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_5_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
	#define nN3 (PML_Thickness)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "StressKernel.h" 
}
#undef _PML_KERNEL_K_BOTTOM

#define _PML_KERNEL_K_TOP
#ifdef CUDA
extern "C" __global__ void PML_6_StressKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_6_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_6_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
	#define nN3 (PML_Thickness)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "StressKernel.h" 
}
#undef _PML_KERNEL_K_TOP

#undef _ST_PML
#endif

#define _ST_MAIN
#define _MAIN_KERNEL
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA)) || (defined(METAL) && defined(METAL_SINGLE_KERNEL))
#define _ST_PML
#endif
#ifdef CUDA
extern "C" __global__ void MAIN_1_StressKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_1_StressKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_1_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#ifdef METAL_SINGLE_KERNEL
	#define nN1 (N1)
	#define nN2 (N2)
	#define nN3 (N3)
	#else
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
	#define nN3 (N3-PML_Thickness*2)
	#endif
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "StressKernel.h" 
}
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA)) || (defined(METAL) && defined(METAL_SINGLE_KERNEL))
#undef _ST_PML
#endif
#undef _MAIN_KERNEL
#undef _ST_MAIN

// PML
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL)) || defined(USE_MINI_KERNELS_CUDA)
#define _PR_PML
#define _PML_KERNEL_I_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_1_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_1_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_1_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (PML_Thickness)
	#define nN2 (N2)
	#define nN3 (N3)
    _PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "ParticleKernel.h" 
}
#undef _PML_KERNEL_I_BOTTOM

#define _PML_KERNEL_I_TOP
#ifdef CUDA
extern "C" __global__ void PML_2_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_2_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_2_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (PML_Thickness)
	#define nN2 (N2)
	#define nN3 (N3)
    _PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "ParticleKernel.h" 
}
#undef _PML_KERNEL_I_TOP

#define _PML_KERNEL_J_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_3_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_3_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_3_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (PML_Thickness)
	#define nN3 (N3)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "ParticleKernel.h" 
}
#undef _PML_KERNEL_J_BOTTOM

#define _PML_KERNEL_J_TOP
#ifdef CUDA
extern "C" __global__ void PML_4_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_4_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_4_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (PML_Thickness)
	#define nN3 (N3)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "ParticleKernel.h" 
}
#undef _PML_KERNEL_J_TOP

#define _PML_KERNEL_K_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_5_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_5_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_5_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
	#define nN3 (PML_Thickness)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "ParticleKernel.h" 
}
#undef _PML_KERNEL_K_BOTTOM

#define _PML_KERNEL_K_TOP
#ifdef CUDA
extern "C" __global__ void PML_6_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void PML_6_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
  _PT k = (_PT) get_global_id(2);
#endif
#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL))
kernel void PML_6_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else	
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
	#define nN3 (PML_Thickness)
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
    #include "ParticleKernel.h" 
}
#undef _PML_KERNEL_K_TOP

#undef _PR_PML
#endif

#define _PR_MAIN
#define _MAIN_KERNEL
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA)) || (defined(METAL) && defined(METAL_SINGLE_KERNEL))
#define _PR_PML
#endif
#if defined(CUDA)
extern "C" __global__ void MAIN_1_ParticleKernel(
			#include "kernelparamsOpenCL.h"
			,unsigned int nStep,unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void MAIN_1_ParticleKernel(
	#include "kernelparamsOpenCL.h"
	, unsigned int nStep,
	unsigned int TypeSource)
{
	_PT i = (_PT) get_global_id(0);
	_PT j = (_PT) get_global_id(1);
	_PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void MAIN_1_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	_PT k = (_PT) gid.z;
	#else
	#ifdef nN1
	#undef nN1
	#endif
	#ifdef nN2
	#undef nN2
	#endif
	#ifdef nN3
	#undef nN3
	#endif
	#ifdef METAL_SINGLE_KERNEL
	#define nN1 (N1)
	#define nN2 (N2)
	#define nN3 (N3)
	#else
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
	#define nN3 (N3-PML_Thickness*2)
	#endif
	_PT k = (_PT) (gid/(nN1*nN2));
  	_PT j = (_PT) ((gid - k*nN1*nN2)/nN1);
  	_PT i = (_PT) (gid - k*nN1*nN2-j*nN1);
	#endif
#endif
	#include "ParticleKernel.h"
}
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA)) || (defined(METAL) && defined(METAL_SINGLE_KERNEL))
#undef _PR_PML
#endif
#undef _PR_MAIN
#undef _MAIN_KERNEL

#if defined(CUDA)
extern "C" __global__ void SnapShot(unsigned int SelK,mexType * Snapshots_pr,mexType * Sigma_xx_pr,mexType * Sigma_yy_pr,mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
  _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void SnapShot(unsigned int SelK,__global mexType * Snapshots_pr,__global mexType * Sigma_xx_pr,__global mexType * Sigma_yy_pr,__global mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
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
	_PT i = (_PT) gid.x;
	_PT j = (_PT) gid.y;
#endif

    if (i>=N1 || j >=N2)
		return;
	// mexType accum=0.0;
	// for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
	// 	{
	// 		_PT index=Ind_Sigma_xx(i,j,(_PT)SelK);
	// 		accum+=(Sigma_xx_pr[index]+Sigma_yy_pr[index]+Sigma_zz_pr[index])/3.0;

	// 	}

	// 	Snapshots_pr[IndN1N2Snap(i,j)+CurrSnap*N1*N2]=accum/ZoneCount;
}

#if defined(CUDA)
extern "C" __global__ void SensorsKernel(
	#include "kernelparamsOpenCL.h"
	,mexType * SensorOutput_pr,
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

#ifndef METALCOMPUTE
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
#else
kernel void SensorsKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(2) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(3) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(15) ]],
	uint gid[[thread_position_in_grid]])
#endif
{
	_PT sj = (_PT) gid;
#endif

	if (sj>=(_PT) NumberSensors)
		return;
	#include"SensorsKernel.h"

}