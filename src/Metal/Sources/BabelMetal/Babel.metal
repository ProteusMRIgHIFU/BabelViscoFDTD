
#include <metal_stdlib>
#include <metal_math>
using namespace metal;


//METAL kernel for Rayleigh Integral

#define pi M_PI_F
#define mr1step (*mr1step_pr)
#define mr1 (*mr1_pr)
#define mr2 (*mr2_pr)
#define n2BaseSteps (*n2BaseSteps_pr)
kernel void ForwardSimpleMetal(const device float *c_wvnb_real [[ buffer(0) ]],
                               const device float *c_wvnb_imag [[ buffer(1) ]],
                               const device int *mr1_pr        [[ buffer(2) ]],
                               const device int *mr2_pr        [[ buffer(3) ]],
                               const device float *r2pr        [[ buffer(4) ]],
                               const device float *r1pr        [[ buffer(5) ]],
                               const device float *a1pr        [[ buffer(6) ]],
                               const device float *u1_real     [[ buffer(7) ]],
                               const device float *u1_imag     [[ buffer(8) ]],
                               device float *py_data_u2_real   [[ buffer(9) ]],
                               device float *py_data_u2_imag   [[ buffer(10) ]],
                               const device int *mr1step_pr    [[ buffer(11) ]],
                               const device int *n2BaseSteps_pr [[ buffer(12) ]],
                               uint si2 [[ thread_position_in_grid ]]) {
    
   
        float dx,dy,dz,R,r2x,r2y,r2z;
        float temp_r,tr ;
        float temp_i,ti,pCos,pSin ;
        int offset=mr1step*si2+n2BaseSteps;
        if (si2<uint(mr2))
        {

            temp_r = 0;
            temp_i = 0;
            r2x=r2pr[si2*3];
            r2y=r2pr[si2*3+1]; 
            r2z=r2pr[si2*3+2];

            for (int si1=0; si1<mr1; si1++)
            {
                // In matlab we have a Fortran convention, in Python-numpy, we have the C-convention for matrixes (hoorray!!!)
                dx=r1pr[si1*3]-r2x;
                dy=r1pr[si1*3+1]-r2y;
                dz=r1pr[si1*3+2]-r2z;


                R=sqrt(dx*dx+dy*dy+dz*dz);
                ti=(exp(R*c_wvnb_imag[0])*a1pr[si1]/R);

                tr=ti;
                pSin=sincos(R*c_wvnb_real[0],pCos);

                tr*=(u1_real[si1+offset]*pCos+u1_imag[si1+offset]*pSin);
                ti*=(u1_imag[si1+offset]*pCos-u1_real[si1+offset]*pSin);

                temp_r +=tr;
                temp_i +=ti;
            }

            R=temp_r;

            temp_r = -temp_r*c_wvnb_imag[0]-temp_i*c_wvnb_real[0];
            temp_i = R*c_wvnb_real[0]-temp_i*c_wvnb_imag[0];

            py_data_u2_real[si2]=temp_r/(2*pi);
            py_data_u2_imag[si2]=temp_i/(2*pi);
        }
}
