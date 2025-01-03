#ifdef _CUDA
extern "C" __global__ void ForwardPropagationKernel(int mr2,
                                                    complex<float> c_wvnb,
                                                    FloatingType MaxDistance, 
                                                    FloatingType *r2pr, 
                                                    FloatingType *r1pr, 
                                                    FloatingType *a1pr,
                                                    complex<float> * u1complex,
                                                    complex<float> *py_data_u2, 
                                                    int mr1,
                                                    int mr1step)
{
    const int si2 = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x ;		// Grid is a "flatten" 1D, thread blocks are 1D
#endif
#ifdef _OPENCL
__kernel  void ForwardPropagationKernel(const int mr2,
                                        const FloatingType c_wvnb_real,
                                        const FloatingType c_wvnb_imag,
                                        const FloatingType MaxDistance,
                                        const int mr1,
                                        __global const FloatingType *r2pr, 
                                        __global const FloatingType *r1pr, 
                                        __global const FloatingType *a1pr, 
                                        __global const FloatingType *u1_real, 
                                        __global const FloatingType *u1_imag,
                                        __global  FloatingType  *py_data_u2_real,
                                        __global  FloatingType  *py_data_u2_imag,
                                        const int mr1step
                                        )
{
    int si2 = get_global_id(0);		// Grid is a "flatten" 1D, thread blocks are 1D
#endif
    #ifdef _MLX
    int si2 = thread_position_in_grid.x;
    #endif

    FloatingType dx,dy,dz,R,r2x,r2y,r2z;
    #ifdef _CUDA
    complex<float> cj=complex<float>(0.0,1);
    complex<float> temp,temp2;
    #else
    FloatingType temp_r,tr ;
    FloatingType temp_i,ti,pCos,pSin ;
    #endif

    int offset = mr1step*si2;

    // Ensure index is less than number of detection points
    if (si2 < mr2)  
    {
        // Temp variables for real and imag values
        #ifdef _CUDA
        temp*=0;
        #else
        temp_r = 0;
        temp_i = 0;
        #endif

        // Detection point x, y, and z coordinates for specific index
        r2x=r2pr[si2*3];
        r2y=r2pr[si2*3+1];
        r2z=r2pr[si2*3+2];

        // loop through each tx element/source point
        for (int si1=0; si1<mr1; si1++)
        {
            // In matlab we have a Fortran convention, in Python-numpy, we have the C-convention for matrixes (hoorray!!!)
            // get x, y, and z distances between source and detection points
            dx=r1pr[si1*3]-r2x;
            dy=r1pr[si1*3+1]-r2y;
            dz=r1pr[si1*3+2]-r2z;

            // Get actual distance between source and detection points
            R = sqrt(dx*dx+dy*dy+dz*dz);

            // If distance is greater than supplied max distance, ignore calculation
            if (MaxDistance > 0.0 && R > MaxDistance) continue;

            #ifdef _CUDA
            temp2=cj*c_wvnb;
            temp2=temp2*(-R);
            temp2=cuexpf(temp2);
            temp2=temp2*u1complex[si1+mr1step*si2];
            temp2=temp2*a1pr[si1]/R;
            temp=temp+temp2;
            #else
            // Start of Rayleigh Integral calculation
            ti=(exp(R*c_wvnb_imag)*a1pr[si1]/R);
            tr=ti;

            // Calculate sin and cosine values of distance * real sound speed
            #if defined(_METAL) || defined(_MLX)
            pSin=sincos(R*c_wvnb_real,pCos);
            #else
            pSin=sincos(R*c_wvnb_real,ppCos);
            #endif

            // Real and imaginary terms of rayleigh integral
            tr*=(u1_real[si1+offset]*pCos+u1_imag[si1+offset]*pSin);
            ti*=(u1_imag[si1+offset]*pCos-u1_real[si1+offset]*pSin);

            // Summate real and imaginary terms
            temp_r += tr;
            temp_i += ti;
            #endif
        }
        
        #ifdef _CUDA
        temp2=cj*c_wvnb;
        temp=temp*temp2;

        py_data_u2[si2]=temp/((float)(2*pi));
        #else
        // Final cumulative real and imaginary pressure at detection point
        R = temp_r;

        temp_r = -temp_r*c_wvnb_imag-temp_i*c_wvnb_real;
        temp_i = R*c_wvnb_real-temp_i*c_wvnb_imag;

        py_data_u2_real[si2]=temp_r/(2*pi);
        py_data_u2_imag[si2]=temp_i/(2*pi);
        #endif
    }
}