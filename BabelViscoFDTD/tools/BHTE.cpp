// Bioheat Transfer Equation

#ifdef _CUDA
extern "C" __global__   void BHTEFDTDKernel(float *d_output, 
                                            float *d_output2,
                                            const float *d_input, 
                                            const float *d_input2,
                                            const float *d_bhArr,
                                            const float *d_perfArr, 
                                            const unsigned int *d_labels,
                                            const float *d_Qarr,
                                            const unsigned int *d_pointsMonitoring,
                                            const float CoreTemp,
                                            const int sonication,
                                            const int outerDimx, 
                                            const int outerDimy, 
                                            const int outerDimz,
                                            const float dt,
                                            float *d_MonitorSlice,
                                            float *d_Temppoints,
                                            const int TotalStepsMonitoring,
                                            const int nFactorMonitoring,
                                            const int n_Step,
                                            const int SelJ,
                                            const unsigned int StartIndexQ,
                                            const unsigned TotalSteps)	
{
    const int gtidx = (blockIdx.x * blockDim.x + threadIdx.x);
    const int gtidy = (blockIdx.y * blockDim.y + threadIdx.y);
    const int gtidz = (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef _OPENCL
__kernel  void BHTEFDTDKernel(__global float *d_output, 
                              __global float *d_output2,
                              __global const float *d_input, 
                              __global const float *d_input2,
                              __global const float *d_bhArr,
                              __global const float *d_perfArr, 
                              __global const unsigned int *d_labels,
                              __global const float *d_Qarr,
                              __global const unsigned int *d_pointsMonitoring,
                              const float CoreTemp,
                              const  unsigned int sonication,
                              const  unsigned int outerDimx, 
                              const  unsigned int outerDimy, 
                              const  unsigned int outerDimz,
                              const float dt,
                              __global float *d_MonitorSlice,
                              __global float *d_Temppoints,
                              const unsigned int TotalStepsMonitoring,
                              const unsigned int nFactorMonitoring,
                              const unsigned int n_Step,
                              const unsigned int SelJ,
                              const unsigned int StartIndexQ,
                              const unsigned TotalSteps)	
{
    const int gtidx = get_global_id(0);
    const int gtidy = get_global_id(1);
    const int gtidz = get_global_id(2);
#endif
#ifdef _METAL
kernel  void BHTEFDTDKernel(device float *d_output [[ buffer(0) ]], 
                            device float *d_output2 [[ buffer(1) ]],
                            device const float *d_input [[ buffer(2) ]], 
                            device const float *d_input2 [[ buffer(3) ]],
                            device const float *d_bhArr [[ buffer(4) ]],
                            device const float *d_perfArr [[ buffer(5) ]], 
                            device const unsigned int *d_labels [[ buffer(6) ]],
                            device const float *d_Qarr [[ buffer(7) ]],
                            device const unsigned int *d_pointsMonitoring [[ buffer(8) ]],
                            device float *d_MonitorSlice [[ buffer(9) ]],
                            device float *d_Temppoints [[ buffer(10) ]],
                            constant float * floatParams [[ buffer(11) ]],
                            constant unsigned int * intparams [[ buffer(12) ]],
                            uint gid[[thread_position_in_grid]])	
{
#endif
#if defined(_METAL) || defined(_MLX)
    #ifdef _MLX
    uint gid = thread_position_in_grid.x;
    #endif

    #define CoreTemp floatParams[0]
    #define dt floatParams[1]
    #define sonication intparams[0]
    #define outerDimx intparams[1]
    #define outerDimy intparams[2]
    #define outerDimz intparams[3]
    #define TotalStepsMonitoring intparams[4]
    #define nFactorMonitoring intparams[5]
    #define n_Step intparams[6]
    #define SelJ intparams[7]
    #define StartIndexQ intparams[8]
    #define TotalSteps intparams[9]
    const int gtidx =  gid/(outerDimy*outerDimz);
    const int gtidy =  (gid - gtidx*outerDimy*outerDimz)/outerDimz;
    const int gtidz =  gid - gtidx*outerDimy*outerDimz - gtidy*outerDimz;
#endif

    #define Tref 43.0
    unsigned int DzDy = outerDimz*outerDimy;
    unsigned int coord = gtidx * DzDy + gtidy * outerDimz + gtidz;
    
    float R1,R2,dtp;
    if(gtidx > 0 && gtidx < outerDimx-1 && gtidy > 0 && gtidy < outerDimy-1 && gtidz > 0 && gtidz < outerDimz-1)
    {

        const unsigned int label = d_labels[coord];

        d_output[coord] = d_input[coord] + d_bhArr[label] * 
                          (d_input[coord + 1] + d_input[coord - 1] + d_input[coord + outerDimz] + d_input[coord - outerDimz] +
                           d_input[coord + DzDy] + d_input[coord - DzDy] - 6.0 * d_input[coord]) +
                          + d_perfArr[label] * (CoreTemp - d_input[coord]);
        if (sonication)
        {
            d_output[coord] += d_Qarr[coord+StartIndexQ];
        }
        
        R2 = (d_output[coord] >= Tref)?0.5:0.25; 
        R1 = (d_input[coord] >= Tref)?0.5:0.25;

        if(fabs(d_output[coord]-d_input[coord])<0.0001)
        {
            d_output2[coord] = d_input2[coord] + dt * pow((float)R1,(float)(Tref-d_input[coord]));
        }
        else
        {
            if(R1 == R2)
            {
                d_output2[coord] = d_input2[coord] + (pow((float)R2,(float)(Tref-d_output[coord])) - pow((float)R1,(float)(Tref-d_input[coord]))) / 
                                ( -(d_output[coord]-d_input[coord])/ dt * log(R1));
            }
            else
            {
                dtp = dt * (Tref - d_input[coord])/(d_output[coord] - d_input[coord]);

                d_output2[coord] = d_input2[coord] + (1 - pow((float)R1,(float)(Tref-d_input[coord]))) / (- (Tref - d_input[coord])/ dtp * log(R1)) + 
                                (pow((float)R2,(float)(Tref-d_output[coord])) - 1) / (-(d_output[coord] - Tref)/(dt - dtp) * log(R2));
            }
        }

        if (gtidy==SelJ && (n_Step % nFactorMonitoring ==0))
        {
            d_MonitorSlice[gtidx*outerDimz*TotalStepsMonitoring+gtidz*TotalStepsMonitoring+ n_Step/nFactorMonitoring] =d_output[coord];
        }

        if (d_pointsMonitoring[coord]>0)
        {
            d_Temppoints[TotalSteps*(d_pointsMonitoring[coord]-1)+n_Step]=d_output[coord];
        }
    }
    else if(gtidx < outerDimx && gtidy < outerDimy && gtidz < outerDimz)
    {
        d_output[coord] = d_input[coord];
        d_output2[coord] = d_input2[coord];
    }
}