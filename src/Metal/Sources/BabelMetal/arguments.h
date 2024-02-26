#include <simd/simd.h>

#ifndef MyArguments_h
#define MyArguments_h

struct RayleighArguments
{
    float c_wvnb_real;
    float c_wvnb_imag;
    float MaxDistance;
    uint32_t mr1step;
    uint32_t mr1;
    uint32_t mr2;
    uint32_t n2BaseSteps;   
};
#endif