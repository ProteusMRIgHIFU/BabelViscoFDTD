#if defined(_METAL) || defined(_MLX)
#include <metal_stdlib>
using namespace metal;
#define pi M_PI_F
#endif

#if !defined(_METAL) && !defined(_MLX)
#define pi 3.141592653589793
#define ppCos &pCos
#endif

typedef float FloatingType;
