
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

#define Tref 43.0
#define CoreTemp (*pCoreTemp)
#define sonication (*psonication)
#define outerDimx (*pouterDimx)
#define outerDimy (*pouterDimy)
#define outerDimz (*pouterDimz)
#define dt (*pdt)
#define TotalStepsMonitoring (*pTotalStepsMonitoring)
#define nFactorMonitoring (*pnFactorMonitoring)
#define n_Step (*pn_Step)
#define SelJ (*pSelJ)
#define StartIndexQ (*pStartIndexQ)
#define gtidx  gid.x
#define gtidy  gid.y
#define gtidz  gid.z

kernel  void BHTEFDTDMetal( device float *d_output               [[ buffer(0) ]], 
                             device float *d_output2              [[ buffer(1) ]], 
                            const device float			*d_input  [[ buffer(2) ]], 
                            const device float			*d_input2 [[ buffer(3) ]], 
                            const device float 			*d_bhArr  [[ buffer(4) ]], 
                            const device float 			*d_perfArr [[ buffer(5) ]], 
                            const device  unsigned int	*d_labels  [[ buffer(6) ]], 
                            device float 		        *d_Qarr   [[ buffer(7) ]], 
                            const device float 			*pCoreTemp [[ buffer(8) ]],
                            const device int			*psonication [[ buffer(9) ]],
                            const  device int			*pouterDimx [[ buffer(10) ]], 
                            const  device int           *pouterDimy [[ buffer(11) ]],
                            const  device int           *pouterDimz [[ buffer(12) ]],
                            const  device float 		*pdt        [[ buffer(13) ]],
                            device float 	            *d_MonitorSlice [[ buffer(14) ]],
                            const  device int           *pTotalStepsMonitoring [[ buffer(15) ]],
                            const  device int           *pnFactorMonitoring [[ buffer(16) ]],
                            const  device int           *pn_Step        [[ buffer(17) ]],
                            const  device int           *pSelJ          [[ buffer(18) ]],
                            const  device unsigned int  *pStartIndexQ   [[ buffer(19) ]],
                            uint3 gid[[thread_position_in_grid]])

{

    int DzDy=outerDimz*outerDimy;
    int coord = gtidx*DzDy + gtidy*outerDimz + gtidz;
    
    float R1,R2,dtp;
    if(gtidx > 0 && gtidx < uint(outerDimx-1) && gtidy > 0 && gtidy < uint(outerDimy-1) && gtidz > 0 && gtidz < uint(outerDimz-1))
    {

            const int label = d_labels[coord];

            d_output[coord] = d_input[coord] + d_bhArr[label] * ( 
                     d_input[coord + 1] + d_input[coord - 1] + d_input[coord + outerDimz] + d_input[coord - outerDimz] +
                      d_input[coord + DzDy] + d_input[coord - DzDy] - 6.0 * d_input[coord]) +
                    + d_perfArr[label] * (CoreTemp - d_input[coord]) ;
            if (sonication)
            {
                d_output[coord]+=d_Qarr[coord+StartIndexQ];
            }
            
            R2 = (d_output[coord] >= Tref)?0.5:0.25; 
            R1 = (d_input[coord] >= Tref)?0.5:0.25;

            if(fabs(d_output[coord]-d_input[coord])<0.0001)
            {
                d_output2[coord] = d_input2[coord] + dt * pow(float(R1),float((Tref-d_input[coord])));
            }
            else
            {
                if(R1 == R2)
                {
                    d_output2[coord] = d_input2[coord] + (pow(float(R2),float((Tref-d_output[coord]))) - pow(float(R1),float((Tref-d_input[coord])))) / 
                                   ( -(d_output[coord]-d_input[coord])/ dt * log(R1));
                }
                else
                {
                    dtp = dt * (Tref - d_input[coord])/(d_output[coord] - d_input[coord]);

                    d_output2[coord] = d_input2[coord] + (1 - pow(float(R1),float((Tref-d_input[coord]))))     / (- (Tref - d_input[coord])/ dtp * log(R1)) + 
                                   (pow(float(R2),float((Tref-d_output[coord]))) - 1) / (-(d_output[coord] - Tref)/(dt - dtp) * log(R2));
                }
            }

            if (gtidy==uint(SelJ) && (n_Step % nFactorMonitoring ==0))
            {
                 d_MonitorSlice[gtidx*outerDimz*TotalStepsMonitoring+gtidz*TotalStepsMonitoring+ n_Step/nFactorMonitoring] =d_output[coord];
            }
        }
        else if(gtidx < uint(outerDimx) && gtidy < uint(outerDimy) && gtidz < uint(outerDimz)){
            d_output[coord] = d_input[coord];
            d_output2[coord] = d_input2[coord];

        }

}

#define mexType float
#define METAL
#define MAX_SIZE_PML 101
#include "Indexing.h"
#include "GPU_KERNELS.h"