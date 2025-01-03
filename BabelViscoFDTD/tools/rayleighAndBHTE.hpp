#if defined(_METAL) || defined(_MLX)
#include <metal_stdlib>
using namespace metal;
#define pi M_PI_F
#endif

#if !defined(_METAL) && !defined(_MLX)
#define pi 3.141592653589793
#define ppCos &pCos
#endif

#ifdef _CUDA
#include <cupy/complex.cuh>
#define MAX_ELEMS_IN_CONSTANT  2730 // the total constant memory can't be greater than 64k bytes

__device__ __forceinline__ complex<float> cuexpf (complex<float> z)
{
    float res_i,res_r;
    sincosf(z.imag(), &res_i, &res_r);
    return expf (z.real())*complex<float> (res_r,res_i);;
}
#endif

typedef float FloatingType;
