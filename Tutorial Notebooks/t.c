#define mexType float
#define OPENCL
#ifndef INDEXING_DEF
#define INDEXING_DEF


typedef unsigned long  _PT;


typedef unsigned char interface_t;
typedef _PT tIndex ;
#define inside 0x00
#define frontLine 0x01
#define frontLinep1 0x02
#define frontLinep2 0x04
#define backLine   0x08
#define backLinem1 0x10
#define backLinem2 0x20

//#define USE_2ND_ORDER_EDGES 1

#ifdef USE_2ND_ORDER_EDGES

//#define REQUIRES_2ND_ORDER_P(__I) ((interface ## __I & frontLinep1)  || (interface ## __I & frontLine) || (interface ## __I & backLine) )
#define REQUIRES_2ND_ORDER_P(__I) (interface ## __I & frontLine)

//#define REQUIRES_2ND_ORDER_M(__I) ((interface ## __I & backLinem1) || (interface ## __I & frontLine) || (interface ## __I & backLine))
#define REQUIRES_2ND_ORDER_M(__I) (interface ## __I & frontLine)
#else

#define REQUIRES_2ND_ORDER_P(__I) (0)

#define REQUIRES_2ND_ORDER_M(__I) (0)
#endif

#define XOR(_a,_b) ((!(_a) && (_b)) || ((_a) && !(_b)))
//these are the coefficients for the 4th order FDTD
//CA = 9/8
//CB = 1/24
#define CA (1.1250)
#define CB (0.0416666666666666643537020320309)

#if defined(METAL)
#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)
#endif

#if defined(CUDA)
#define __PRE_MAT p->
#elif defined(METAL)
#define __PRE_MAT k_
#else
#define __PRE_MAT
#endif

#if defined(METAL)
#define EL(_Mat,_i,_j,_k) CONCAT(__PRE_MAT,_Mat ## _pr[Ind_##_Mat(_i,_j,_k)])
#define ELD(_Mat,_index) CONCAT(__PRE_MAT,_Mat ## _pr[_index])
#else
#define EL(_Mat,_i,_j,_k) __PRE_MAT _Mat##_pr[Ind_##_Mat(_i,_j,_k)]
#define ELD(_Mat,_index) __PRE_MAT _Mat##_pr[_index]
#endif

#define ELO(_Mat,_i,_j,_k)  _Mat##_pr[Ind_##_Mat(_i,_j,_k)]
#define ELDO(_Mat,_index)  _Mat##_pr[_index]

#define hELO(_Mat,_i,_j,_k)  _Mat##_pr[hInd_##_Mat(_i,_j,_k)]


//////////////////////////////////////////////
#define hInd_Source(a,b)((b)*INHOST(N2)+a)

#define hIndN1N2Snap(a,b) ((b)*INHOST(N1)+a)
#define hIndN1N2N3(a,b,c,_ZoneSize)   ((c)*(INHOST(N1)*INHOST(N2))      +(b)*INHOST(N1)    +a+(CurZone*(_ZoneSize)))
#define hIndN1p1N2N3(a,b,c,_ZoneSize) ((c)*((INHOST(N1)+1)*INHOST(N2))  +(b)*(INHOST(N1)+1)+a+(CurZone*(_ZoneSize)))
#define hIndN1N2p1N3(a,b,c,_ZoneSize) ((c)*((INHOST(N1))*(INHOST(N2)+1))+(b)*(INHOST(N1))  +a+(CurZone*(_ZoneSize)))
#define hIndN1N2N3p1(a,b,c,_ZoneSize) ((c)*(INHOST(N1)*INHOST(N2))      +(b)*INHOST(N1)    +a+(CurZone*(_ZoneSize)))
#define hIndN1p1N2p1N3p1(a,b,c,_ZoneSize) ((c)*((INHOST(N1)+1)*(INHOST(N2)+1))+(b)*(INHOST(N1)+1)+a +(CurZone*(_ZoneSize)))

#define hCorrecI(_i,_j,_k) ((_j)>hLimit_J_low_PML && (_k)>hLimit_K_low_PML && (_j)<hLimit_J_up_PML && (_k)<hLimit_K_up_PML && (_i)> hLimit_I_low_PML ?hSizeCorrI :0)
#define hCorrecJ(_j,_k) ((_k)>hLimit_K_low_PML && (_k)<hLimit_K_up_PML && (_j)>hLimit_J_low_PML+1 ?((_j)<hLimit_J_up_PML?((_j)-hLimit_J_low_PML-1)*(hSizeCorrI):hSizeCorrI*hSizeCorrJ):0)
#define hCorrecK(_k) ( (_k) <= hLimit_K_low_PML+1  ? 0 : ((_k)<hLimit_K_up_PML ? ((_k)-hLimit_K_low_PML-1)*(hSizeCorrI*hSizeCorrJ): (hSizeCorrI*hSizeCorrJ*hSizeCorrK)))

#define hIndexPML(_i,_j,_k,_ZoneSize)  (hIndN1N2N3(_i,_j,_k,_ZoneSize) - hCorrecI(_i,_j,_k) - hCorrecJ(_j,_k) -hCorrecK(_k))

#define hIndexPMLxp1(_i,_j,_k,_ZoneSize) (hIndN1p1N2N3(_i,_j,_k,_ZoneSize) - hCorrecI(_i,_j,_k) - hCorrecJ(_j,_k) -hCorrecK(_k))
#define hIndexPMLyp1(_i,_j,_k,_ZoneSize) (hIndN1N2p1N3(_i,_j,_k,_ZoneSize) - hCorrecI(_i,_j,_k) - hCorrecJ(_j,_k) -hCorrecK(_k))
#define hIndexPMLzp1(_i,_j,_k,_ZoneSize) hIndexPML(_i,_j,_k,_ZoneSize)
#define hIndexPMLxp1yp1zp1(_i,_j,_k,_ZoneSize) (hIndN1p1N2p1N3p1(_i,_j,_k,_ZoneSize) - hCorrecI(_i,_j,_k) - hCorrecJ(_j,_k) -hCorrecK(_k))

#define hInd_MaterialMap(_i,_j,_k) (hIndN1p1N2p1N3p1(_i,_j,_k,(INHOST(N1)+1)*(INHOST(N2)+1)*(INHOST(N3)+1)))

#define hInd_V_x(_i,_j,_k) (hIndN1p1N2N3(_i,_j,_k,(INHOST(N1)+1)*INHOST(N2)*INHOST(N3)))
#define hInd_V_y(_i,_j,_k) (hIndN1N2p1N3(_i,_j,_k,INHOST(N1)*(INHOST(N2)+1)*INHOST(N3)))
#define hInd_V_z(_i,_j,_k) (hIndN1N2N3p1(_i,_j,_k,INHOST(N1)*INHOST(N2)*(INHOST(N3)+1)))

#define hInd_Vx(_i,_j,_k) (hIndN1p1N2N3(_i,_j,_k,(INHOST(N1)+1)*INHOST(N2)*INHOST(N3)))
#define hInd_Vy(_i,_j,_k) (hIndN1N2p1N3(_i,_j,_k,INHOST(N1)*(INHOST(N2)+1)*INHOST(N3)))
#define hInd_Vz(_i,_j,_k) (hIndN1N2N3p1(_i,_j,_k,INHOST(N1)*INHOST(N2)*(INHOST(N3)+1)))

#define hInd_Sigma_xx(_i,_j,_k) (hIndN1N2N3(_i,_j,_k,INHOST(N1)*INHOST(N2)*INHOST(N3)))
#define hInd_Sigma_yy(_i,_j,_k) (hIndN1N2N3(_i,_j,_k,INHOST(N1)*INHOST(N2)*INHOST(N3)))
#define hInd_Sigma_zz(_i,_j,_k) (hIndN1N2N3(_i,_j,_k,INHOST(N1)*INHOST(N2)*INHOST(N3)))

#define hInd_Pressure(_i,_j,_k) (hIndN1N2N3(_i,_j,_k,INHOST(N1)*INHOST(N2)*INHOST(N3)))
#define hInd_Pressure_old(_i,_j,_k) (hIndN1N2N3(_i,_j,_k,INHOST(N1)*INHOST(N2)*INHOST(N3)))

#define hInd_Sigma_xy(_i,_j,_k) (hIndN1p1N2p1N3p1(_i,_j,_k,(INHOST(N1)+1)*(INHOST(N2)+1)*(INHOST(N3)+1)))
#define hInd_Sigma_xz(_i,_j,_k) (hIndN1p1N2p1N3p1(_i,_j,_k,(INHOST(N1)+1)*(INHOST(N2)+1)*(INHOST(N3)+1)))
#define hInd_Sigma_yz(_i,_j,_k) (hIndN1p1N2p1N3p1(_i,_j,_k,(INHOST(N1)+1)*(INHOST(N2)+1)*(INHOST(N3)+1)))

#define hInd_SqrAcc(_i,_j,_k) (hIndN1N2N3(_i,_j,_k,INHOST(N1)*INHOST(N2)*INHOST(N3)))

#define hInd_V_x_x(_i,_j,_k) (hIndexPMLxp1(_i,_j,_k,INHOST(SizePMLxp1)))
#define hInd_V_y_x(_i,_j,_k) (hIndexPMLxp1(_i,_j,_k,INHOST(SizePMLxp1)))
#define hInd_V_z_x(_i,_j,_k) (hIndexPMLxp1(_i,_j,_k,INHOST(SizePMLxp1)))

#define hInd_V_x_y(_i,_j,_k) (hIndexPMLyp1(_i,_j,_k,INHOST(SizePMLyp1)))
#define hInd_V_y_y(_i,_j,_k) (hIndexPMLyp1(_i,_j,_k,INHOST(SizePMLyp1)))
#define hInd_V_z_y(_i,_j,_k) (hIndexPMLyp1(_i,_j,_k,INHOST(SizePMLyp1)))

#define hInd_V_x_z(_i,_j,_k) (hIndexPMLzp1(_i,_j,_k,INHOST(SizePMLzp1)))
#define hInd_V_y_z(_i,_j,_k) (hIndexPMLzp1(_i,_j,_k,INHOST(SizePMLzp1)))
#define hInd_V_z_z(_i,_j,_k) (hIndexPMLzp1(_i,_j,_k,INHOST(SizePMLzp1)))


#define hInd_Sigma_x_xx(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )
#define hInd_Sigma_y_xx(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )
#define hInd_Sigma_z_xx(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )

#define hInd_Sigma_x_yy(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )
#define hInd_Sigma_y_yy(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )
#define hInd_Sigma_z_yy(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )

#define hInd_Sigma_x_zz(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )
#define hInd_Sigma_y_zz(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )
#define hInd_Sigma_z_zz(_i,_j,_k) (hIndexPML(_i,_j,_k,INHOST(SizePML)) )


#define hInd_Sigma_x_xy(_i,_j,_k)(hIndexPMLxp1yp1zp1(_i,_j,_k,INHOST(SizePMLxp1yp1zp1)) )
#define hInd_Sigma_y_xy(_i,_j,_k)(hIndexPMLxp1yp1zp1(_i,_j,_k,INHOST(SizePMLxp1yp1zp1)) )

#define hInd_Sigma_x_xz(_i,_j,_k)(hIndexPMLxp1yp1zp1(_i,_j,_k,INHOST(SizePMLxp1yp1zp1)) )
#define hInd_Sigma_z_xz(_i,_j,_k)(hIndexPMLxp1yp1zp1(_i,_j,_k,INHOST(SizePMLxp1yp1zp1)) )

#define hInd_Sigma_y_yz(_i,_j,_k)(hIndexPMLxp1yp1zp1(_i,_j,_k,INHOST(SizePMLxp1yp1zp1)) )
#define hInd_Sigma_z_yz(_i,_j,_k)(hIndexPMLxp1yp1zp1(_i,_j,_k,INHOST(SizePMLxp1yp1zp1)) )

#define IsOnPML_I(_i) ((_i) <=Limit_I_low_PML || (_i)>=Limit_I_up_PML ? 1:0)
#define IsOnPML_J(_j) ((_j) <=Limit_J_low_PML || (_j)>=Limit_J_up_PML ? 1:0)
#define IsOnPML_K(_k) ((_k) <=Limit_K_low_PML || (_k)>=Limit_K_up_PML ? 1:0)

#define IsOnLowPML_I(_i) (_i) <=Limit_I_low_PML
#define IsOnLowPML_J(_j) (_j) <=Limit_J_low_PML
#define IsOnLowPML_K(_k) (_k) <=Limit_K_low_PML

////////////////////////////////////////
#define Ind_Source(a,b)((b)*N2+a)

#define IndN1N2Snap(a,b) ((b)*N1+a)

#define IndN1N2N3(a,b,c,_ZoneSize)   ((c)*(N1*N2)      +(b)*N1    +a+(CurZone*(_ZoneSize)))
#define IndN1p1N2N3(a,b,c,_ZoneSize) ((c)*((N1+1)*N2)  +(b)*(N1+1)+a+(CurZone*(_ZoneSize)))
#define IndN1N2p1N3(a,b,c,_ZoneSize) ((c)*((N1)*(N2+1))+(b)*(N1)  +a+(CurZone*(_ZoneSize)))
#define IndN1N2N3p1(a,b,c,_ZoneSize) ((c)*(N1*N2)      +(b)*N1    +a+(CurZone*(_ZoneSize)))
#define IndN1p1N2p1N3p1(a,b,c,_ZoneSize) ((c)*((N1+1)*(N2+1))+(b)*(N1+1)+a +(CurZone*(_ZoneSize)))

#define CorrecI(_i,_j,_k) ((_j)>Limit_J_low_PML && (_k)>Limit_K_low_PML && (_j)<Limit_J_up_PML && (_k)<Limit_K_up_PML && (_i)> Limit_I_low_PML ?SizeCorrI :0)
#define CorrecJ(_j,_k) ((_k)>Limit_K_low_PML && (_k)<Limit_K_up_PML && (_j)>Limit_J_low_PML+1 ?((_j)<Limit_J_up_PML?((_j)-Limit_J_low_PML-1)*(SizeCorrI):SizeCorrI*SizeCorrJ):0)
#define CorrecK(_k) ( (_k) <= Limit_K_low_PML+1  ? 0 : ((_k)<Limit_K_up_PML ? ((_k)-Limit_K_low_PML-1)*(SizeCorrI*SizeCorrJ): (SizeCorrI*SizeCorrJ*SizeCorrK)))

#define IndexPML(_i,_j,_k,_ZoneSize)  (IndN1N2N3(_i,_j,_k,_ZoneSize) - CorrecI(_i,_j,_k) - CorrecJ(_j,_k) -CorrecK(_k))

#define IndexPMLxp1(_i,_j,_k,_ZoneSize) (IndN1p1N2N3(_i,_j,_k,_ZoneSize) - CorrecI(_i,_j,_k) - CorrecJ(_j,_k) -CorrecK(_k))
#define IndexPMLyp1(_i,_j,_k,_ZoneSize) (IndN1N2p1N3(_i,_j,_k,_ZoneSize) - CorrecI(_i,_j,_k) - CorrecJ(_j,_k) -CorrecK(_k))
#define IndexPMLzp1(_i,_j,_k,_ZoneSize) IndexPML(_i,_j,_k,_ZoneSize)
#define IndexPMLxp1yp1zp1(_i,_j,_k,_ZoneSize) (IndN1p1N2p1N3p1(_i,_j,_k,_ZoneSize) - CorrecI(_i,_j,_k) - CorrecJ(_j,_k) -CorrecK(_k))

#define Ind_MaterialMap(_i,_j,_k) (IndN1p1N2p1N3p1(_i,_j,_k,(N1+1)*(N2+1)*(N3+1)))

#define Ind_V_x(_i,_j,_k) (IndN1p1N2N3(_i,_j,_k,(N1+1)*N2*N3))
#define Ind_V_y(_i,_j,_k) (IndN1N2p1N3(_i,_j,_k,N1*(N2+1)*N3))
#define Ind_V_z(_i,_j,_k) (IndN1N2N3p1(_i,_j,_k,N1*N2*(N3+1)))

#define Ind_Vx(_i,_j,_k) (IndN1p1N2N3(_i,_j,_k,(N1+1)*N2*N3))
#define Ind_Vy(_i,_j,_k) (IndN1N2p1N3(_i,_j,_k,N1*(N2+1)*N3))
#define Ind_Vz(_i,_j,_k) (IndN1N2N3p1(_i,_j,_k,N1*N2*(N3+1)))

#define Ind_Sigma_xx(_i,_j,_k) (IndN1N2N3(_i,_j,_k,N1*N2*N3))
#define Ind_Sigma_yy(_i,_j,_k) (IndN1N2N3(_i,_j,_k,N1*N2*N3))
#define Ind_Sigma_zz(_i,_j,_k) (IndN1N2N3(_i,_j,_k,N1*N2*N3))

#define Ind_Pressure(_i,_j,_k) (IndN1N2N3(_i,_j,_k,N1*N2*N3))

#define Ind_Sigma_xy(_i,_j,_k) (IndN1p1N2p1N3p1(_i,_j,_k,(N1+1)*(N2+1)*(N3+1)))
#define Ind_Sigma_xz(_i,_j,_k) (IndN1p1N2p1N3p1(_i,_j,_k,(N1+1)*(N2+1)*(N3+1)))
#define Ind_Sigma_yz(_i,_j,_k) (IndN1p1N2p1N3p1(_i,_j,_k,(N1+1)*(N2+1)*(N3+1)))

#define Ind_SqrAcc(_i,_j,_k) (IndN1N2N3(_i,_j,_k,N1*N2*N3))

#define Ind_V_x_x(_i,_j,_k) (IndexPMLxp1(_i,_j,_k,SizePMLxp1))
#define Ind_V_y_x(_i,_j,_k) (IndexPMLxp1(_i,_j,_k,SizePMLxp1))
#define Ind_V_z_x(_i,_j,_k) (IndexPMLxp1(_i,_j,_k,SizePMLxp1))

#define Ind_V_x_y(_i,_j,_k) (IndexPMLyp1(_i,_j,_k,SizePMLyp1))
#define Ind_V_y_y(_i,_j,_k) (IndexPMLyp1(_i,_j,_k,SizePMLyp1))
#define Ind_V_z_y(_i,_j,_k) (IndexPMLyp1(_i,_j,_k,SizePMLyp1))

#define Ind_V_x_z(_i,_j,_k) (IndexPMLzp1(_i,_j,_k,SizePMLzp1))
#define Ind_V_y_z(_i,_j,_k) (IndexPMLzp1(_i,_j,_k,SizePMLzp1))
#define Ind_V_z_z(_i,_j,_k) (IndexPMLzp1(_i,_j,_k,SizePMLzp1))


#define Ind_Sigma_x_xx(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )
#define Ind_Sigma_y_xx(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )
#define Ind_Sigma_z_xx(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )

#define Ind_Sigma_x_yy(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )
#define Ind_Sigma_y_yy(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )
#define Ind_Sigma_z_yy(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )

#define Ind_Sigma_x_zz(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )
#define Ind_Sigma_y_zz(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )
#define Ind_Sigma_z_zz(_i,_j,_k) (IndexPML(_i,_j,_k,SizePML) )


#define Ind_Sigma_x_xy(_i,_j,_k)(IndexPMLxp1yp1zp1(_i,_j,_k,SizePMLxp1yp1zp1) )
#define Ind_Sigma_y_xy(_i,_j,_k)(IndexPMLxp1yp1zp1(_i,_j,_k,SizePMLxp1yp1zp1) )

#define Ind_Sigma_x_xz(_i,_j,_k)(IndexPMLxp1yp1zp1(_i,_j,_k,SizePMLxp1yp1zp1) )
#define Ind_Sigma_z_xz(_i,_j,_k)(IndexPMLxp1yp1zp1(_i,_j,_k,SizePMLxp1yp1zp1) )

#define Ind_Sigma_y_yz(_i,_j,_k)(IndexPMLxp1yp1zp1(_i,_j,_k,SizePMLxp1yp1zp1) )
#define Ind_Sigma_z_yz(_i,_j,_k)(IndexPMLxp1yp1zp1(_i,_j,_k,SizePMLxp1yp1zp1) )


#define iPML(_i) ((_i) <=Limit_I_low_PML ? (_i) : ((_i)<Limit_I_up_PML ? PML_Thickness : (_i)<N1 ? PML_Thickness-1-(_i)+Limit_I_up_PML:0))
#define jPML(_j) ((_j) <=Limit_J_low_PML ? (_j) : ((_j)<Limit_J_up_PML ? PML_Thickness : (_j)<N2 ? PML_Thickness-1-(_j)+Limit_J_up_PML:0))
#define kPML(_k) ((_k) <=Limit_K_low_PML ? (_k) : ((_k)<Limit_K_up_PML ? PML_Thickness : (_k)<N3 ? PML_Thickness-1-(_k)+Limit_K_up_PML:0))


#if defined(CUDA) || defined(OPENCL)
#define InvDXDT_I 	(IsOnLowPML_I(i) ? gpuInvDXDTpluspr[iPML(i)] : gpuInvDXDTplushppr[iPML(i)] )
#define DXDT_I 		(IsOnLowPML_I(i) ? gpuDXDTminuspr[iPML(i)] : gpuDXDTminushppr[iPML(i)] )
#define InvDXDT_J 	(IsOnLowPML_J(j) ? gpuInvDXDTpluspr[jPML(j)] : gpuInvDXDTplushppr[jPML(j)] )
#define DXDT_J 		(IsOnLowPML_J(j) ? gpuDXDTminuspr[jPML(j)] : gpuDXDTminushppr[jPML(j)] )
#define InvDXDT_K 	(IsOnLowPML_K(k) ? gpuInvDXDTpluspr[kPML(k)] : gpuInvDXDTplushppr[kPML(k)] )
#define DXDT_K 		(IsOnLowPML_I(k) ? gpuDXDTminuspr[kPML(k)] : gpuDXDTminushppr[kPML(k)] )

#define InvDXDThp_I 	(IsOnLowPML_I(i) ? gpuInvDXDTplushppr[iPML(i)] : gpuInvDXDTpluspr[iPML(i)] )
#define DXDThp_I 		(IsOnLowPML_I(i) ? gpuDXDTminushppr[iPML(i)] : gpuDXDTminuspr[iPML(i)] )
#define InvDXDThp_J 	(IsOnLowPML_J(j) ? gpuInvDXDTplushppr[jPML(j)] : gpuInvDXDTpluspr[jPML(j)] )
#define DXDThp_J 		(IsOnLowPML_J(j) ? gpuDXDTminushppr[jPML(j)] : gpuDXDTminuspr[jPML(j)] )
#define InvDXDThp_K 	(IsOnLowPML_K(k) ? gpuInvDXDTplushppr[kPML(k)] : gpuInvDXDTpluspr[kPML(k)] )
#define DXDThp_K 		(IsOnLowPML_I(k) ? gpuDXDTminushppr[kPML(k)] : gpuDXDTminuspr[kPML(k)])
#else
#define InvDXDT_I 	(IsOnLowPML_I(i) ? InvDXDTplus_pr[iPML(i)] : InvDXDTplushp_pr[iPML(i)] )
#define DXDT_I 		(IsOnLowPML_I(i) ? DXDTminus_pr[iPML(i)] : DXDTminushp_pr[iPML(i)] )
#define InvDXDT_J 	(IsOnLowPML_J(j) ? InvDXDTplus_pr[jPML(j)] : InvDXDTplushp_pr[jPML(j)] )
#define DXDT_J 		(IsOnLowPML_J(j) ? DXDTminus_pr[jPML(j)] : DXDTminushp_pr[jPML(j)] )
#define InvDXDT_K 	(IsOnLowPML_K(k) ? InvDXDTplus_pr[kPML(k)] : InvDXDTplushp_pr[kPML(k)] )
#define DXDT_K 		(IsOnLowPML_I(k) ? DXDTminus_pr[kPML(k)] : DXDTminushp_pr[kPML(k)] )

#define InvDXDThp_I 	(IsOnLowPML_I(i) ? InvDXDTplushp_pr[iPML(i)] : InvDXDTplus_pr[iPML(i)] )
#define DXDThp_I 		(IsOnLowPML_I(i) ? DXDTminushp_pr[iPML(i)] : DXDTminus_pr[iPML(i)] )
#define InvDXDThp_J 	(IsOnLowPML_J(j) ? InvDXDTplushp_pr[jPML(j)] : InvDXDTplus_pr[jPML(j)] )
#define DXDThp_J 		(IsOnLowPML_J(j) ? DXDTminushp_pr[jPML(j)] : DXDTminus_pr[jPML(j)] )
#define InvDXDThp_K 	(IsOnLowPML_K(k) ? InvDXDTplushp_pr[kPML(k)] : InvDXDTplus_pr[kPML(k)] )
#define DXDThp_K 		(IsOnLowPML_I(k) ? DXDTminushp_pr[kPML(k)] : DXDTminus_pr[kPML(k)])
#endif


#define MASK_ALLV				0x0000000001
#define MASK_Vx   			0x0000000002
#define MASK_Vy   			0x0000000004
#define MASK_Vz   			0x0000000008
#define MASK_Sigmaxx    0x0000000010
#define MASK_Sigmayy    0x0000000020
#define MASK_Sigmazz    0x0000000040
#define MASK_Sigmaxy    0x0000000080
#define MASK_Sigmaxz    0x0000000100
#define MASK_Sigmayz    0x0000000200
#define MASK_Pressure   0x0000000400

#define IS_ALLV_SELECTED(_Value) 					(_Value &MASK_ALLV)
#define IS_Vx_SELECTED(_Value) 						(_Value &MASK_Vx)
#define IS_Vy_SELECTED(_Value) 						(_Value &MASK_Vy)
#define IS_Vz_SELECTED(_Value) 						(_Value &MASK_Vz)
#define IS_Sigmaxx_SELECTED(_Value) 			(_Value &MASK_Sigmaxx)
#define IS_Sigmayy_SELECTED(_Value) 			(_Value &MASK_Sigmayy)
#define IS_Sigmazz_SELECTED(_Value) 			(_Value &MASK_Sigmazz)
#define IS_Sigmaxy_SELECTED(_Value) 			(_Value &MASK_Sigmaxy)
#define IS_Sigmaxz_SELECTED(_Value) 			(_Value &MASK_Sigmaxz)
#define IS_Sigmayz_SELECTED(_Value) 			(_Value &MASK_Sigmayz)
#define IS_Pressure_SELECTED(_Value) 			(_Value &MASK_Pressure)

#define COUNT_SELECTIONS(_VarName,_Value) \
				{ _VarName =0;\
					_VarName += IS_ALLV_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Vx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Vy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Vz_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmayy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmazz_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxz_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmayz_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Pressure_SELECTED(_Value) ? 1 : 0;}

#define SEL_RMS			  	0x0000000001
#define SEL_PEAK   			0x0000000002

#define ACCOUNT_RMSPEAK(_VarName)\
if IS_ ## _VarName ## _SELECTED(INHOST(SelMapsRMSPeak)) \
{\
	 INHOST(IndexRMSPeak_ ## _VarName)=curMapIndex;\
	 curMapIndex++; }

 #define ACCOUNT_SENSOR(_VarName)\
 if IS_ ## _VarName ## _SELECTED(INHOST(SelMapsSensors)) \
 {\
 	 INHOST(IndexSensor_ ## _VarName)=curMapIndex;\
 	 curMapIndex++; }

#if defined(METAL)
#define CInd_N1 0
#define CInd_N2 1
#define CInd_N3 2
#define CInd_Limit_I_low_PML 3
#define CInd_Limit_J_low_PML 4
#define CInd_Limit_K_low_PML 5
#define CInd_Limit_I_up_PML 6
#define CInd_Limit_J_up_PML 7
#define CInd_Limit_K_up_PML 8
#define CInd_SizeCorrI 9
#define CInd_SizeCorrJ 10
#define CInd_SizeCorrK 11
#define CInd_PML_Thickness 12
#define CInd_NumberSources 13
#define CInd_NumberSensors 14
#define CInd_TimeSteps 15
#define CInd_SizePML 16
#define CInd_SizePMLxp1 17
#define CInd_SizePMLyp1 18
#define CInd_SizePMLzp1 19
#define CInd_SizePMLxp1yp1zp1 20
#define CInd_ZoneCount 21
#define CInd_SelRMSorPeak 22
#define CInd_SelMapsRMSPeak 23
#define CInd_IndexRMSPeak_ALLV 24
#define CInd_IndexRMSPeak_Vx 25
#define CInd_IndexRMSPeak_Vy 26
#define CInd_IndexRMSPeak_Vz 27
#define CInd_IndexRMSPeak_Sigmaxx 28
#define CInd_IndexRMSPeak_Sigmayy 29
#define CInd_IndexRMSPeak_Sigmazz 30
#define CInd_IndexRMSPeak_Sigmaxy 31
#define CInd_IndexRMSPeak_Sigmaxz 32
#define CInd_IndexRMSPeak_Sigmayz 33
#define CInd_NumberSelRMSPeakMaps 34
#define CInd_SelMapsSensors 35
#define CInd_IndexSensor_ALLV 36
#define CInd_IndexSensor_Vx 37
#define CInd_IndexSensor_Vy 38
#define CInd_IndexSensor_Vz 39
#define CInd_IndexSensor_Sigmaxx 40
#define CInd_IndexSensor_Sigmayy 41
#define CInd_IndexSensor_Sigmazz 42
#define CInd_IndexSensor_Sigmaxy 43
#define CInd_IndexSensor_Sigmaxz 44
#define CInd_IndexSensor_Sigmayz 45
#define CInd_NumberSelSensorMaps 46
#define CInd_SensorSubSampling 47
#define CInd_nStep 48
#define CInd_TypeSource 49
#define CInd_CurrSnap 50
#define CInd_LengthSource 51
#define CInd_SelK 52
#define CInd_IndexRMSPeak_Pressure 53
#define CInd_IndexSensor_Pressure 54
#define CInd_SensorStart 55

//Make LENGTH_CONST_UINT one value larger than the last index
#define LENGTH_CONST_UINT 56

//Indexes for float
#define CInd_DT 0
#define CInd_InvDXDTplus 1
#define CInd_DXDTminus (1+MAX_SIZE_PML)
#define CInd_InvDXDTplushp (1+MAX_SIZE_PML*2)
#define CInd_DXDTminushp (1+MAX_SIZE_PML*3)
//Make LENGTH_CONST_MEX one value larger than the last index
#define LENGTH_CONST_MEX (1+MAX_SIZE_PML*4)

#define CInd_V_x_x 0
#define CInd_V_y_x 1
#define CInd_V_z_x 2
#define CInd_V_x_y 3
#define CInd_V_y_y 4
#define CInd_V_z_y 5
#define CInd_V_x_z 6
#define CInd_V_y_z 7
#define CInd_V_z_z 8

#define CInd_Vx 9
#define CInd_Vy 10
#define CInd_Vz 11

#define CInd_Rxx 12
#define CInd_Ryy 13
#define CInd_Rzz 14

#define CInd_Rxy 15
#define CInd_Rxz 16
#define CInd_Ryz 17

#define CInd_Sigma_x_xx 18
#define CInd_Sigma_y_xx 19
#define CInd_Sigma_z_xx 20
#define CInd_Sigma_x_yy 21
#define CInd_Sigma_y_yy 22
#define CInd_Sigma_z_yy 23
#define CInd_Sigma_x_zz 24
#define CInd_Sigma_y_zz 25

#define CInd_Sigma_z_zz 26
#define CInd_Sigma_x_xy 27
#define CInd_Sigma_y_xy 28
#define CInd_Sigma_x_xz 29
#define CInd_Sigma_z_xz 30
#define CInd_Sigma_y_yz 31
#define CInd_Sigma_z_yz 32

#define CInd_Sigma_xy 33
#define CInd_Sigma_xz 34
#define CInd_Sigma_yz 35

#define CInd_Sigma_xx 36
#define CInd_Sigma_yy 37
#define CInd_Sigma_zz 38

#define CInd_SourceFunctions 39

#define CInd_LambdaMiuMatOverH  40
#define CInd_LambdaMatOverH	 41
#define CInd_MiuMatOverH 42
#define CInd_TauLong 43
#define CInd_OneOverTauSigma	44
#define CInd_TauShear 45
#define CInd_InvRhoMatH	 46
#define CInd_Ox 47
#define CInd_Oy 48
#define CInd_Oz 49
#define CInd_Pressure 50

#define CInd_SqrAcc 51

#define CInd_SensorOutput 52

#define LENGTH_INDEX_MEX 53

#define CInd_IndexSensorMap  0
#define CInd_SourceMap	1
#define CInd_MaterialMap 2

#define LENGTH_INDEX_UINT 3

#endif

#endif
__constant float DT = 1.50000005e-07;
__constant  unsigned int N1 = 129;
__constant  unsigned int N2 = 129;
__constant  unsigned int N3 = 234;
__constant  unsigned int Limit_I_low_PML = 11;
__constant  unsigned int Limit_J_low_PML = 11;
__constant  unsigned int Limit_K_low_PML = 11;
__constant  unsigned int Limit_I_up_PML = 117;
__constant  unsigned int Limit_J_up_PML = 117;
__constant  unsigned int Limit_K_up_PML = 222;
__constant  unsigned int SizeCorrI = 105;
__constant  unsigned int SizeCorrJ = 105;
__constant  unsigned int SizeCorrK = 210;
__constant  unsigned int PML_Thickness = 12;
__constant  unsigned int NumberSources = 1;
__constant  unsigned int LengthSource = 78;
__constant  unsigned int ZoneCount = 1;
__constant  unsigned int SizePMLxp1 = 1608931;
__constant  unsigned int SizePMLyp1 = 1608931;
__constant  unsigned int SizePMLzp1 = 1595386;
__constant  unsigned int SizePML = 1578745;
__constant  unsigned int SizePMLxp1yp1zp1 = 1656251;
__constant  unsigned int NumberSensors = 22050;
__constant  unsigned int TimeSteps = 545;
__constant  unsigned int SelRMSorPeak = 2;
__constant  unsigned int SelMapsRMSPeak = 1024;
__constant  _PT IndexRMSPeak_ALLV = 0;
__constant  _PT IndexRMSPeak_Vx = 0;
__constant  _PT IndexRMSPeak_Vy = 0;
__constant  _PT IndexRMSPeak_Vz = 0;
__constant  _PT IndexRMSPeak_Sigmaxx = 0;
__constant  _PT IndexRMSPeak_Sigmayy = 0;
__constant  _PT IndexRMSPeak_Sigmazz = 0;
__constant  _PT IndexRMSPeak_Sigmaxy = 0;
__constant  _PT IndexRMSPeak_Sigmaxz = 0;
__constant  _PT IndexRMSPeak_Sigmayz = 0;
__constant  _PT IndexRMSPeak_Pressure = 0;
__constant  unsigned int NumberSelRMSPeakMaps = 1;
__constant  unsigned int SelMapsSensors = 14;
__constant  _PT IndexSensor_ALLV = 0;
__constant  _PT IndexSensor_Vx = 0;
__constant  _PT IndexSensor_Vy = 1;
__constant  _PT IndexSensor_Vz = 2;
__constant  _PT IndexSensor_Sigmaxx = 0;
__constant  _PT IndexSensor_Sigmayy = 0;
__constant  _PT IndexSensor_Sigmazz = 0;
__constant  _PT IndexSensor_Sigmaxy = 0;
__constant  _PT IndexSensor_Sigmaxz = 0;
__constant  _PT IndexSensor_Sigmayz = 0;
__constant  _PT IndexSensor_Pressure = 0;
__constant  unsigned int NumberSelSensorMaps = 3;
__constant  unsigned int SensorSubSampling = 2;
__constant  unsigned int SensorStart = 0;
__constant float gpuInvDXDTpluspr[13] ={
1.11941041e-07,
1.16669149e-07,
1.21348918e-07,
1.25918689e-07,
1.30309331e-07,
1.34445784e-07,
1.38249135e-07,
1.41639546e-07,
1.44539754e-07,
1.46878904e-07,
1.48596627e-07,
1.49646681e-07,
1.50000005e-07};
__constant float gpuDXDTminuspr[13] ={
4400059.5,
4762087,
5092634,
5391700,
5659285.5,
5895390.5,
6100015,
6273158.5,
6414821.5,
6525003.5,
6603705.5,
6650926.5,
6666666.5};
__constant float gpuInvDXDTplushppr[13] ={
1.14307596e-07,
1.19018743e-07,
1.23651716e-07,
1.28140968e-07,
1.324142e-07,
1.36394092e-07,
1.40000893e-07,
1.4315556e-07,
1.45783531e-07,
1.4781871e-07,
1.49207352e-07,
1.49911514e-07,
1.50000005e-07};
__constant float gpuDXDTminushppr[13] ={
4585008.5,
4931295.5,
5246102,
5529428,
5781273,
6001638,
6190522,
6347925,
6473847.5,
6568289.5,
6631251,
6662731.5,
6666666.5};
#ifdef METAL
#define N1 p_CONSTANT_BUFFER_UINT[CInd_N1]
#define N2 p_CONSTANT_BUFFER_UINT[CInd_N2]
#define N3 p_CONSTANT_BUFFER_UINT[CInd_N3]
#define Limit_I_low_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_I_low_PML]
#define Limit_J_low_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_J_low_PML]
#define Limit_K_low_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_K_low_PML]
#define Limit_I_up_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_I_up_PML]
#define Limit_J_up_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_J_up_PML]
#define Limit_K_up_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_K_up_PML]
#define SizeCorrI p_CONSTANT_BUFFER_UINT[CInd_SizeCorrI]
#define SizeCorrJ p_CONSTANT_BUFFER_UINT[CInd_SizeCorrJ]
#define SizeCorrK p_CONSTANT_BUFFER_UINT[CInd_SizeCorrK]
#define PML_Thickness p_CONSTANT_BUFFER_UINT[CInd_PML_Thickness]
#define NumberSources p_CONSTANT_BUFFER_UINT[CInd_NumberSources]
#define LengthSource p_CONSTANT_BUFFER_UINT[CInd_LengthSource]
#define NumberSensors p_CONSTANT_BUFFER_UINT[CInd_NumberSensors]
#define TimeSteps p_CONSTANT_BUFFER_UINT[CInd_TimeSteps]

#define SizePML p_CONSTANT_BUFFER_UINT[CInd_SizePML]
#define SizePMLxp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLxp1]
#define SizePMLyp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLyp1]
#define SizePMLzp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLzp1]
#define SizePMLxp1yp1zp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLxp1yp1zp1]
#define ZoneCount p_CONSTANT_BUFFER_UINT[CInd_ZoneCount]

#define SelRMSorPeak p_CONSTANT_BUFFER_UINT[CInd_SelRMSorPeak]
#define SelMapsRMSPeak p_CONSTANT_BUFFER_UINT[CInd_SelMapsRMSPeak]
#define IndexRMSPeak_ALLV p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_ALLV]
#define IndexRMSPeak_Vx p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Vx]
#define IndexRMSPeak_Vy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Vy]
#define IndexRMSPeak_Vz p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Vz]
#define IndexRMSPeak_Sigmaxx p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmaxx]
#define IndexRMSPeak_Sigmayy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmayy]
#define IndexRMSPeak_Sigmazz p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmazz]
#define IndexRMSPeak_Sigmaxy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmaxy]
#define IndexRMSPeak_Sigmaxz p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmaxz]
#define IndexRMSPeak_Sigmayz p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmayz]
#define IndexRMSPeak_Pressure p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Pressure]
#define NumberSelRMSPeakMaps p_CONSTANT_BUFFER_UINT[CInd_NumberSelRMSPeakMaps]

#define SelMapsSensors p_CONSTANT_BUFFER_UINT[CInd_SelMapsSensors]
#define IndexSensor_ALLV p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_ALLV]
#define IndexSensor_Vx p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Vx]
#define IndexSensor_Vy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Vy]
#define IndexSensor_Vz p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Vz]
#define IndexSensor_Sigmaxx p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmaxx]
#define IndexSensor_Sigmayy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmayy]
#define IndexSensor_Sigmazz p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmazz]
#define IndexSensor_Sigmaxy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmaxy]
#define IndexSensor_Sigmaxz p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmaxz]
#define IndexSensor_Sigmayz p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmayz]
#define IndexSensor_Pressure p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Pressure]
#define NumberSelSensorMaps p_CONSTANT_BUFFER_UINT[CInd_NumberSelSensorMaps]
#define SensorSubSampling p_CONSTANT_BUFFER_UINT[CInd_SensorSubSampling]
#define SensorStart p_CONSTANT_BUFFER_UINT[CInd_SensorStart]
#define nStep p_CONSTANT_BUFFER_UINT[CInd_nStep]
#define CurrSnap p_CONSTANT_BUFFER_UINT[CInd_CurrSnap]
#define TypeSource p_CONSTANT_BUFFER_UINT[CInd_TypeSource]
#define SelK p_CONSTANT_BUFFER_UINT[CInd_SelK]

#define DT p_CONSTANT_BUFFER_MEX[CInd_DT]
#define InvDXDTplus_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplus)
#define DXDTminus_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminus)
#define InvDXDTplushp_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplushp)
#define DXDTminushp_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminushp)

#define __def_MEX_VAR_0(__NameVar)  (&p_MEX_BUFFER_0[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_1(__NameVar)  (&p_MEX_BUFFER_1[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_2(__NameVar)  (&p_MEX_BUFFER_2[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_3(__NameVar)  (&p_MEX_BUFFER_3[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_4(__NameVar)  (&p_MEX_BUFFER_4[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_5(__NameVar)  (&p_MEX_BUFFER_5[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_6(__NameVar)  (&p_MEX_BUFFER_6[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_7(__NameVar)  (&p_MEX_BUFFER_7[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_8(__NameVar)  (&p_MEX_BUFFER_8[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_9(__NameVar)  (&p_MEX_BUFFER_9[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_10(__NameVar)  (&p_MEX_BUFFER_10[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_11(__NameVar)  (&p_MEX_BUFFER_11[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 

#define __def_UINT_VAR(__NameVar)  (&p_UINT_BUFFER[ ((unsigned long) (p_INDEX_UINT[CInd_ ##__NameVar*2])) | (((unsigned long) (p_INDEX_UINT[CInd_ ##__NameVar*2+1]))<<32) ])

// #define __def_MEX_VAR(__NameVar)  (&p_MEX_BUFFER[ p_INDEX_MEX[CInd_ ##__NameVar ]]) 
// #define __def_UINT_VAR(__NameVar)  (&p_UINT_BUFFER[ p_INDEX_UINT[CInd_ ##__NameVar]])


#define k_V_x_x_pr  __def_MEX_VAR_0(V_x_x)
#define k_V_y_x_pr  __def_MEX_VAR_0(V_y_x)
#define k_V_z_x_pr  __def_MEX_VAR_0(V_z_x)
#define k_V_x_y_pr  __def_MEX_VAR_0(V_x_y)
#define k_V_y_y_pr  __def_MEX_VAR_0(V_y_y)
#define k_V_z_y_pr  __def_MEX_VAR_0(V_z_y)
#define k_V_x_z_pr  __def_MEX_VAR_0(V_x_z)
#define k_V_y_z_pr  __def_MEX_VAR_0(V_y_z)
#define k_V_z_z_pr  __def_MEX_VAR_0(V_z_z)

#define k_Vx_pr  __def_MEX_VAR_1(Vx)
#define k_Vy_pr  __def_MEX_VAR_1(Vy)
#define k_Vz_pr  __def_MEX_VAR_1(Vz)

#define k_Rxx_pr  __def_MEX_VAR_2(Rxx)
#define k_Ryy_pr  __def_MEX_VAR_2(Ryy)
#define k_Rzz_pr  __def_MEX_VAR_2(Rzz)

#define k_Rxy_pr  __def_MEX_VAR_3(Rxy)
#define k_Rxz_pr  __def_MEX_VAR_3(Rxz)
#define k_Ryz_pr  __def_MEX_VAR_3(Ryz)

#define k_Sigma_x_xx_pr  __def_MEX_VAR_4(Sigma_x_xx)
#define k_Sigma_y_xx_pr  __def_MEX_VAR_4(Sigma_y_xx)
#define k_Sigma_z_xx_pr  __def_MEX_VAR_4(Sigma_z_xx)
#define k_Sigma_x_yy_pr  __def_MEX_VAR_4(Sigma_x_yy)
#define k_Sigma_y_yy_pr  __def_MEX_VAR_4(Sigma_y_yy)
#define k_Sigma_z_yy_pr  __def_MEX_VAR_4(Sigma_z_yy)
#define k_Sigma_x_zz_pr  __def_MEX_VAR_4(Sigma_x_zz)
#define k_Sigma_y_zz_pr  __def_MEX_VAR_4(Sigma_y_zz)

#define k_Sigma_z_zz_pr  __def_MEX_VAR_5(Sigma_z_zz)
#define k_Sigma_x_xy_pr  __def_MEX_VAR_5(Sigma_x_xy)
#define k_Sigma_y_xy_pr  __def_MEX_VAR_5(Sigma_y_xy)
#define k_Sigma_x_xz_pr  __def_MEX_VAR_5(Sigma_x_xz)
#define k_Sigma_z_xz_pr  __def_MEX_VAR_5(Sigma_z_xz)
#define k_Sigma_y_yz_pr  __def_MEX_VAR_5(Sigma_y_yz)
#define k_Sigma_z_yz_pr  __def_MEX_VAR_5(Sigma_z_yz)

#define k_Sigma_xy_pr  __def_MEX_VAR_6(Sigma_xy)
#define k_Sigma_xz_pr  __def_MEX_VAR_6(Sigma_xz)
#define k_Sigma_yz_pr  __def_MEX_VAR_6(Sigma_yz)

#define k_Sigma_xx_pr  __def_MEX_VAR_7(Sigma_xx)
#define k_Sigma_yy_pr  __def_MEX_VAR_7(Sigma_yy)
#define k_Sigma_zz_pr  __def_MEX_VAR_7(Sigma_zz)

#define k_SourceFunctions_pr __def_MEX_VAR_8(SourceFunctions)

#define k_LambdaMiuMatOverH_pr  __def_MEX_VAR_9(LambdaMiuMatOverH)
#define k_LambdaMatOverH_pr     __def_MEX_VAR_9(LambdaMatOverH)
#define k_MiuMatOverH_pr        __def_MEX_VAR_9(MiuMatOverH)
#define k_TauLong_pr            __def_MEX_VAR_9(TauLong)
#define k_OneOverTauSigma_pr    __def_MEX_VAR_9(OneOverTauSigma)
#define k_TauShear_pr           __def_MEX_VAR_9(TauShear)
#define k_InvRhoMatH_pr         __def_MEX_VAR_9(InvRhoMatH)
#define k_Ox_pr  __def_MEX_VAR_9(Ox)
#define k_Oy_pr  __def_MEX_VAR_9(Oy)
#define k_Oz_pr  __def_MEX_VAR_9(Oz)
#define k_Pressure_pr  __def_MEX_VAR_9(Pressure)

#define k_SqrAcc_pr  __def_MEX_VAR_10(SqrAcc)

#define k_SensorOutput_pr  __def_MEX_VAR_11(SensorOutput)

#define k_IndexSensorMap_pr  __def_UINT_VAR(IndexSensorMap)
#define k_SourceMap_pr		 __def_UINT_VAR(SourceMap)
#define k_MaterialMap_pr	 __def_UINT_VAR(MaterialMap)
#endif

#if defined(CUDA)
__global__ void StressKernel(InputDataKernel *p,unsigned int nStep, unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void StressKernel(
__global mexType *V_x_x_pr,
__global mexType *V_y_x_pr,
__global mexType *V_z_x_pr,
__global mexType *V_x_y_pr,
__global mexType *V_y_y_pr,
__global mexType *V_z_y_pr,
__global mexType *V_x_z_pr,
__global mexType *V_y_z_pr,
__global mexType *V_z_z_pr,
__global mexType *Vx_pr,
__global mexType *Vy_pr,
__global mexType *Vz_pr,
__global mexType *Rxx_pr,
__global mexType *Ryy_pr,
__global mexType *Rzz_pr,
__global mexType *Rxy_pr,
__global mexType *Rxz_pr,
__global mexType *Ryz_pr,
__global mexType *Sigma_x_xx_pr,
__global mexType *Sigma_y_xx_pr,
__global mexType *Sigma_z_xx_pr,
__global mexType *Sigma_x_yy_pr,
__global mexType *Sigma_y_yy_pr,
__global mexType *Sigma_z_yy_pr,
__global mexType *Sigma_x_zz_pr,
__global mexType *Sigma_y_zz_pr,
__global mexType *Sigma_z_zz_pr,
__global mexType *Sigma_x_xy_pr,
__global mexType *Sigma_y_xy_pr,
__global mexType *Sigma_x_xz_pr,
__global mexType *Sigma_z_xz_pr,
__global mexType *Sigma_y_yz_pr,
__global mexType *Sigma_z_yz_pr,
__global mexType *Sigma_xy_pr,
__global mexType *Sigma_xz_pr,
__global mexType *Sigma_yz_pr,
__global mexType *Sigma_xx_pr,
__global mexType *Sigma_yy_pr,
__global mexType *Sigma_zz_pr,
__global mexType *SourceFunctions_pr,
__global mexType * LambdaMiuMatOverH_pr,
__global mexType * LambdaMatOverH_pr,
__global mexType * MiuMatOverH_pr,
__global mexType * TauLong_pr,
__global mexType * OneOverTauSigma_pr,
__global mexType * TauShear_pr,
__global mexType * InvRhoMatH_pr,
__global mexType * SqrAcc_pr,
__global unsigned int * MaterialMap_pr,
__global unsigned int * SourceMap_pr,
__global mexType * Ox_pr,
__global mexType * Oy_pr,
__global mexType * Oz_pr,
__global mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  const _PT i = (_PT) get_global_id(0);
  const _PT j = (_PT) get_global_id(1);
  const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void StressKernel(
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

    mexType Diff,value,Dx,Dy,Dz,m1,m2,m3,m4,RigidityXY=0.0,RigidityXZ=0.0,
        RigidityYZ=0.0,LambdaMiu,Miu,LambdaMiuComp,MiuComp,
        dFirstPart,OneOverTauSigma,dFirstPartForR,NextR,
            TauShearXY=0.0,TauShearXZ=0.0,TauShearYZ=0.0,
            accum_xx=0.0,accum_yy=0.0,accum_zz=0.0,
            accum_xy=0.0,accum_xz=0.0,accum_yz=0.0,
			accum_p=0.0;
#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
   	_PT index,index2;
	unsigned long MaterialID,source,bAttenuating=1;
	_PT CurZone;
for ( CurZone=0;CurZone<ZoneCount;CurZone++)
  {
  	if (i<N1 && j<N2 && k<N3)
  	{
      index=Ind_MaterialMap(i,j,k);
      MaterialID=ELD(MaterialMap,index);

  		m1=ELD(MiuMatOverH,MaterialID);
  #ifdef USE_2ND_ORDER_EDGES

          //if (m1!=0.0)

          {
              if (i<N1-1)
              {
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i+1,j,k))==0.0,m1==0.0)
                      interfaceX=interfaceX|frontLine;
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i+2,j,k))==0.0,m1==0.0)
                      interfaceX=interfaceX|frontLinep1;
              }
              if(i>0)
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i-1,j,k))==0.0,m1==0.0)
                      interfaceX=interfaceX|backLine;
              if(i>2)
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i-2,j,k))==0.0,m1==0.0)
                      interfaceX=interfaceX|backLinem1;

              if (j<N2-1)
              {
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j+1,k))==0.0,m1==0.0)
                      interfaceY=interfaceY|frontLine;
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j+2,k))==0.0,m1==0.0)
                      interfaceY=interfaceY|frontLinep1;
              }
              if(j>0)
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j-1,k))==0.0,m1==0.0)
                      interfaceY=interfaceY|backLine;
              if(j>1)
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j-2,k))==0.0,m1==0.0)
                      interfaceY=interfaceY|backLinem1;

              if (k<N3-1)
              {
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k+1))==0.0,m1==0.0)
                      interfaceZ=interfaceZ|frontLine;
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k+2))==0.0,m1==0.0)
                      interfaceZ=interfaceZ|frontLinep1;
              }
              if(k>0)
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k-1))==0.0,m1==0.0)
                      interfaceZ=interfaceZ|backLine;
              if(k>1)
                  if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k-1))==0.0,m1==0.0)
                      interfaceZ=interfaceZ|backLinem1;
          }
  #endif

  		/*
          RigidityXY=m1;
          RigidityXZ=m1;
          RigidityYZ=m1;

          TauShearXY=ELD(TauShear,MaterialID);
          TauShearXZ=TauShearXY;
          TauShearYZ=TauShearXY;
          */

  		m2=ELD(MiuMatOverH,EL(MaterialMap,i+1,j,k));
  		m3=ELD(MiuMatOverH,EL(MaterialMap,i,j+1,k));
  		m4=ELD(MiuMatOverH,EL(MaterialMap,i+1,j+1,k));

   		value=m1*m2*m3*m4;
  		RigidityXY =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
  		TauShearXY = value!=0.0 ? 0.25*(ELD(TauShear,MaterialID) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j,k)) +
  							 ELD(TauShear,EL(MaterialMap,i,j+1,k)) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j+1,k)))
  							 : ELD(TauShear,MaterialID);


  		m3=ELD(MiuMatOverH,EL(MaterialMap,i,j,k+1));
  		m4=ELD(MiuMatOverH,EL(MaterialMap,i+1,j,k+1));

  		value=m1*m2*m3*m4;
  		RigidityXZ =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
  		TauShearXZ= value!=0.0 ? 0.25*(ELD(TauShear,MaterialID) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j,k)) +
  							 ELD(TauShear,EL(MaterialMap,i,j,k+1)) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j,k+1)))
  							 : ELD(TauShear,MaterialID);


  		m2=ELD(MiuMatOverH,EL(MaterialMap,i,j+1,k));
  		m4=ELD(MiuMatOverH,EL(MaterialMap,i,j+1,k+1));

          value=m1*m2*m3*m4;

  		RigidityYZ =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
  		TauShearYZ= value!=0.0 ? 0.25*(ELD(TauShear,MaterialID) +
  							 ELD(TauShear,EL(MaterialMap,i,j+1,k)) +
  							 ELD(TauShear,EL(MaterialMap,i,j,k+1)) +
  							 ELD(TauShear,EL(MaterialMap,i,j+1,k+1)))
  							 : ELD(TauShear,MaterialID);


  	}

  	if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 ||  IsOnPML_K(k)==1)//We are in the PML borders
  	 {
  		if (i<N1-1 && j <N2-1 && k < N3-1)
  		{


  			Diff= i>1 && i <N1-1 ? CA*(EL(Vx,i,j,k)-EL(Vx,i-1,j,k)) -
  			                       CB*(EL(Vx,i+1,j,k)-EL(Vx,i-2,j,k))
  			      : i>0 && i <N1 ? (EL(Vx,i,j,k)-EL(Vx,i-1,j,k))  :0;


  			index2=Ind_Sigma_x_xx(i,j,k);
  			ELD(Sigma_x_xx,index2) =InvDXDT_I*(
  											ELD(Sigma_x_xx,index2)*DXDT_I+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff);
  			index2=Ind_Sigma_x_yy(i,j,k);
  			ELD(Sigma_x_yy,index2) =InvDXDT_I*(
  											ELD(Sigma_x_yy,index2)*DXDT_I+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

  			index2=Ind_Sigma_x_zz(i,j,k);
  			ELD(Sigma_x_zz,index2) =InvDXDT_I*(
  											ELD(Sigma_x_zz,index2)*DXDT_I+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);




  			Diff= j>1 && j < N2-1 ? CA*(EL(Vy,i,j,k)-EL(Vy,i,j-1,k))-
  									CB*(EL(Vy,i,j+1,k)-EL(Vy,i,j-2,k))
  			        : j>0 && j < N2 ? EL(Vy,i,j,k)-EL(Vy,i,j-1,k):0;

  			index2=Ind_Sigma_y_xx(i,j,k);
  			ELD(Sigma_y_xx,index2) =InvDXDT_J*(
  											ELD(Sigma_y_xx,index2)*DXDT_J+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

  			index2=Ind_Sigma_y_yy(i,j,k);
  			ELD(Sigma_y_yy,index2) =InvDXDT_J*(
  											ELD(Sigma_y_yy,index2)*DXDT_J+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff);

  			index2=Ind_Sigma_y_zz(i,j,k);
  			ELD(Sigma_y_zz,index2) =InvDXDT_J*(
  											ELD(Sigma_y_zz,index2)*DXDT_J+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);



  			Diff= k>1 && k < N3-1 ? CA*(EL(Vz,i,j,k)-EL(Vz,i,j,k-1)) -
  									CB*(EL(Vz,i,j,k+1)-EL(Vz,i,j,k-2))
  			                       :k>0 && k < N3 ? EL(Vz,i,j,k)-EL(Vz,i,j,k-1) : 0;
  			index2=Ind_Sigma_z_xx(i,j,k);
  			ELD(Sigma_z_xx,index2) =InvDXDT_K*(
  											ELD(Sigma_z_xx,index2)*DXDT_K+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

  			index2=Ind_Sigma_z_yy(i,j,k);
  			ELD(Sigma_z_yy,index2) =InvDXDT_K*(
  											ELD(Sigma_z_yy,index2)*DXDT_K+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

  			index2=Ind_Sigma_z_zz(i,j,k);
  			ELD(Sigma_z_zz,index2) =InvDXDT_K*(
  											ELD(Sigma_z_zz,index2)*DXDT_K+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff);




  			index=Ind_Sigma_xy(i,j,k);
  			index2=Ind_Sigma_x_xy(i,j,k);

  			Diff= i >0 && i<N1-2 ? CA*(EL(Vy,i+1,j,k)-EL(Vy,i,j,k)) -
  			                   CB*(EL(Vy,i+2,j,k)-EL(Vy,i-1,j,k))
  			                    :i<N1-1 ? EL(Vy,i+1,j,k)-EL(Vy,i,j,k):0;

  			ELD(Sigma_x_xy,index2) =InvDXDThp_I*(
  											ELD(Sigma_x_xy,index2)*DXDThp_I+
  											RigidityXY*
  											Diff);


  			index2=Ind_Sigma_x_xz(i,j,k);

  			Diff= i>0 && i<N1-2 ? CA*(EL(Vz,i+1,j,k)-EL(Vz,i,j,k)) -
  								  CB*(EL(Vz,i+2,j,k)-EL(Vz,i-1,j,k))
  								  :i<N1-1 ? EL(Vz,i+1,j,k)-EL(Vz,i,j,k):0;

  			ELD(Sigma_x_xz,index2) =InvDXDThp_I*(
  											ELD(Sigma_x_xz,index2)*DXDThp_I+
  											RigidityXZ*
  											Diff);


  			index=Ind_Sigma_xy(i,j,k);
  			index2=Ind_Sigma_y_xy(i,j,k);

  			Diff=j > 0 && j<N2-2 ? CA*(EL(Vx,i,j+1,k)-EL(Vx,i,j,k) )-
  			                       CB*(EL(Vx,i,j+2,k)-EL(Vx,i,j-1,k) )
  			                       :j<N2-1 ? EL(Vx,i,j+1,k)-EL(Vx,i,j,k) :0;

  			ELD(Sigma_y_xy,index2) =InvDXDThp_J*(
  											ELD(Sigma_y_xy,index2)*DXDThp_J+
  											RigidityXY*
  											Diff);

  			index2=Ind_Sigma_y_yz(i,j,k);

  			Diff=j>0 && j<N2-2 ? CA*(EL(Vz,i,j+1,k)-EL(Vz,i,j,k)) -
  							     CB*(EL(Vz,i,j+2,k)-EL(Vz,i,j-1,k))
  							     :j<N2-1 ? EL(Vz,i,j+1,k)-EL(Vz,i,j,k):0;

  			ELD(Sigma_y_yz,index2) =InvDXDThp_J*(
  											ELD(Sigma_y_yz,index2)*DXDThp_J+
  											RigidityYZ*
  											Diff);

  			index=Ind_Sigma_xy(i,j,k);

  			index2=Ind_Sigma_z_xz(i,j,k);


  			Diff=k >0 && k < N3-2 ? CA*(EL(Vx,i,j,k+1)-EL(Vx,i,j,k)) -
  			                        CB*(EL(Vx,i,j,k+2)-EL(Vx,i,j,k-1)):
                                      k < N3-1 ? EL(Vx,i,j,k+1)-EL(Vx,i,j,k) :0;

  			ELD(Sigma_z_xz,index2) =InvDXDThp_K*(
  											ELD(Sigma_z_xz,index2)*DXDThp_K+
  											RigidityXZ*
  											Diff);

  			index2=Ind_Sigma_z_yz(i,j,k);

  			Diff=k>0 && k < N3-2 ? CA*(EL(Vy,i,j,k+1)-EL(Vy,i,j,k))-
  			                       CB*(EL(Vy,i,j,k+2)-EL(Vy,i,j,k-1))
  			                       :k < N3-1 ? EL(Vy,i,j,k+1)-EL(Vy,i,j,k):0;
  			ELD(Sigma_z_yz,index2) =InvDXDThp_K*(
  											ELD(Sigma_z_yz,index2)*DXDThp_K+
  											RigidityYZ*
  											Diff);


  			index=Ind_Sigma_xx(i,j,k);
  			index2=Ind_Sigma_x_xx(i,j,k);
  			ELD(Sigma_xx,index)= ELD(Sigma_x_xx,index2) + ELD(Sigma_y_xx,index2)+ ELD(Sigma_z_xx,index2);
  			ELD(Sigma_yy,index)= ELD(Sigma_x_yy,index2) + ELD(Sigma_y_yy,index2)+ ELD(Sigma_z_yy,index2);
  			ELD(Sigma_zz,index)= ELD(Sigma_x_zz,index2) + ELD(Sigma_y_zz,index2)+ ELD(Sigma_z_zz,index2);
  		}
  		index=Ind_Sigma_xy(i,j,k);
  		index2=Ind_Sigma_x_xy(i,j,k);
  		ELD(Sigma_xy,index)= ELD(Sigma_x_xy,index2) + ELD(Sigma_y_xy,index2);
  		ELD(Sigma_xz,index)= ELD(Sigma_x_xz,index2) + ELD(Sigma_z_xz,index2);
  		ELD(Sigma_yz,index)= ELD(Sigma_y_yz,index2) + ELD(Sigma_z_yz,index2);

  	}
  	else
  	{
  		//We are in the center, no need to check any limits, the use of the PML simplify this
  		index=Ind_Sigma_xx(i,j,k);

		if (REQUIRES_2ND_ORDER_M(X))
			Dx=EL(Vx,i,j,k)-EL(Vx,i-1,j,k);
		else
			Dx=CA*(EL(Vx,i,j,k)-EL(Vx,i-1,j,k))-
				CB*(EL(Vx,i+1,j,k)-EL(Vx,i-2,j,k));

		if REQUIRES_2ND_ORDER_M(Y)
			Dy=EL(Vy,i,j,k)-EL(Vy,i,j-1,k);
		else
			Dy=CA*(EL(Vy,i,j,k)-EL(Vy,i,j-1,k))-
				CB*(EL(Vy,i,j+1,k)-EL(Vy,i,j-2,k));

		if REQUIRES_2ND_ORDER_M(Z)
			Dz=EL(Vz,i,j,k)-EL(Vz,i,j,k-1);
		else
			Dz=CA*(EL(Vz,i,j,k)-EL(Vz,i,j,k-1))-
              CB*(EL(Vz,i,j,k+1)-EL(Vz,i,j,k-2));

		
		//We will use the particle displacement to estimate the acoustic pressure, and using the conservation of mass formula
		//We can use the stress kernel as V matrices are not being modified in this kernel,
		// and the spatial derivatives are the same ones required for pressure calculation
        // partial(p)/partial(t) = \rho c^2 div(V)
        //it is important to mention that the Python function will need still to multiply the result for the maps of (speed of sound)^2 and density, 
		// and divide by the spatial step.
		EL(Pressure,i,j,k)+=DT*(Dx+Dy+Dz);
        accum_p+=EL(Pressure,i,j,k);


  		LambdaMiu=ELD(LambdaMiuMatOverH,MaterialID)*(1.0+ELD(TauLong,MaterialID));
  		Miu=2.0*ELD(MiuMatOverH,MaterialID)*(1.0+ELD(TauShear,MaterialID));
  		OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
		dFirstPart=LambdaMiu*(Dx+Dy+Dz);
		
		if (ELD(TauLong,MaterialID)!=0.0 || ELD(TauShear,MaterialID)!=0.0) // We avoid unnecessary calculations if there is no attenuation
		{
			
			LambdaMiuComp=DT*ELD(LambdaMiuMatOverH,MaterialID)*(ELD(TauLong,MaterialID)*OneOverTauSigma);
			dFirstPartForR=LambdaMiuComp*(Dx+Dy+Dz);
			MiuComp=DT*2.0*ELD(MiuMatOverH,MaterialID)*(ELD(TauShear,MaterialID)*OneOverTauSigma);
			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxx,index) - dFirstPartForR + MiuComp*(Dy+Dz))
  		    	  /(1+DT*0.5*OneOverTauSigma);

			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy+Dz) + 0.5*(ELD(Rxx,index) + NextR));
			ELD(Rxx,index)=NextR;
		}
		else
		{
			bAttenuating=0;
			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy+Dz));
		}
  		
	    accum_xx+=ELD(Sigma_xx,index);

		if (bAttenuating==1)
		{
  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryy,index) - dFirstPartForR + MiuComp*(Dx+Dz))
  		    	  /(1+DT*0.5*OneOverTauSigma);
				
  			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx+Dz) + 0.5*(ELD(Ryy,index) + NextR));
			ELD(Ryy,index)=NextR;
		}
		else
			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx+Dz));
      	
		accum_yy+=ELD(Sigma_yy,index);

  		if (bAttenuating==1)
		{
			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rzz,index) - dFirstPartForR +MiuComp*(Dx+Dy))
				/(1+DT*0.5*OneOverTauSigma);
  			ELD(Sigma_zz,index)+=	DT*(dFirstPart - Miu*(Dx+Dy) + 0.5*(ELD(Rzz,index) + NextR));
			ELD(Rzz,index)=NextR;
		}
		else
			ELD(Sigma_zz,index)+=	DT*(dFirstPart - Miu*(Dx+Dy));

      	accum_zz+=ELD(Sigma_zz,index);

  		index=Ind_Sigma_xy(i,j,k);

  		if (RigidityXY!=0.0)
  		{
              if (REQUIRES_2ND_ORDER_P(X))
                  Dx=EL(Vy,i+1,j,k)-EL(Vy,i,j,k);
              else
                  Dx=CA*(EL(Vy,i+1,j,k)-EL(Vy,i,j,k))-
                     CB*(EL(Vy,i+2,j,k)-EL(Vy,i-1,j,k));


              if (REQUIRES_2ND_ORDER_P(Y))
                  Dx+=EL(Vx,i,j+1,k)-EL(Vx,i,j,k);
              else
                  Dx+=CA*(EL(Vx,i,j+1,k)-EL(Vx,i,j,k))-
                      CB*(EL(Vx,i,j+2,k)-EL(Vx,i,j-1,k));

  			Miu=RigidityXY*(1.0+TauShearXY);

			if (TauShearXY!=0.0)
			{
				MiuComp=RigidityXY*(TauShearXY*OneOverTauSigma);
				NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxy,index) - DT*MiuComp*Dx)
  		          /(1+DT*0.5*OneOverTauSigma);
				ELD(Sigma_xy,index)+= DT*(Miu*Dx + 0.5*(ELD(Rxy,index) +NextR));
				ELD(Rxy,index)=NextR;
			}
			else
				ELD(Sigma_xy,index)+= DT*(Miu*Dx );
        	
			accum_xy+=ELD(Sigma_xy,index);

  		}
        else
            ELD(Rxy,index)=0.0;


  		if (RigidityXZ!=0.0)
  		{
			if (REQUIRES_2ND_ORDER_P(X))
				Dx=EL(Vz,i+1,j,k)-EL(Vz,i,j,k);
			else
				Dx=CA*(EL(Vz,i+1,j,k)-EL(Vz,i,j,k))-
					CB*(EL(Vz,i+2,j,k)-EL(Vz,i-1,j,k));

			if (REQUIRES_2ND_ORDER_P(Z))
				Dx+=EL(Vx,i,j,k+1)-EL(Vx,i,j,k);
			else
				Dx+=CA*(EL(Vx,i,j,k+1)-EL(Vx,i,j,k))-
					CB*(EL(Vx,i,j,k+2)-EL(Vx,i,j,k-1));

	  		Miu=RigidityXZ*(1.0+TauShearXZ);

			if (TauShearXZ!=0.0)
			{
  				MiuComp=RigidityXZ*(TauShearXZ*OneOverTauSigma);
	  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxz,index) - DT*MiuComp*Dx)
  			          /(1+DT*0.5*OneOverTauSigma);
				ELD(Sigma_xz,index)+= DT*(Miu*Dx + 0.5*(ELD(Rxz,index) +NextR));
				ELD(Rxz,index)=NextR;
			}
			else
				ELD(Sigma_xz,index)+= DT*(Miu*Dx );
        	accum_xz+=ELD(Sigma_xz,index);
  			
  		}
        else
            ELD(Rxz,index)=0.0;

  		if (RigidityYZ!=0.0 )
  		{
			if (REQUIRES_2ND_ORDER_P(Y))
				Dy=EL(Vz,i,j+1,k)-EL(Vz,i,j,k);
			else
				Dy=CA*(EL(Vz,i,j+1,k)-EL(Vz,i,j,k))-
					CB*(EL(Vz,i,j+2,k)-EL(Vz,i,j-1,k));

			if (REQUIRES_2ND_ORDER_P(Z))
				Dy+=EL(Vy,i,j,k+1)-EL(Vy,i,j,k);
			else
				Dy+=CA*(EL(Vy,i,j,k+1)-EL(Vy,i,j,k))-
					CB*(EL(Vy,i,j,k+2)-EL(Vy,i,j,k-1));

  			Miu=RigidityYZ*(1.0+TauShearYZ);
			
			if (TauShearYZ!=0)
			{
				MiuComp=RigidityYZ*(TauShearYZ*OneOverTauSigma);

				NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryz,index) - DT*MiuComp*Dy)
					/(1+DT*0.5*OneOverTauSigma);

				ELD(Sigma_yz,index)+= DT*(Miu*Dy + 0.5*(ELD(Ryz,index) +NextR));
				ELD(Ryz,index)=NextR;
			}
			else 
				ELD(Sigma_yz,index)+= DT*(Miu*Dy );
        	accum_yz+=ELD(Sigma_yz,index);

  		}
          else
              ELD(Ryz,index)=0.0;
		
		if ((nStep < LengthSource) && TypeSource>=2) //Source is stress
  		{
  			index=IndN1N2N3(i,j,k,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
  			  source--; //need to use C index
  			  value=ELD(SourceFunctions,nStep*NumberSources+source)*ELD(Ox,index); // We use Ox as mechanism to provide weighted arrays
				index=Ind_Sigma_xx(i,j,k);
                if ((TypeSource-2)==0)
                {
                    ELD(Sigma_xx,index)+=value;
                    ELD(Sigma_yy,index)+=value;
                    ELD(Sigma_zz,index)+=value;
                }
                else
                {
                   ELD(Sigma_xx,index)=value;
                   ELD(Sigma_yy,index)=value;
                   ELD(Sigma_zz,index)=value;
                }

  			}
  		}
  	}
  }
  if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0 && IsOnPML_K(k)==0 && nStep>=SensorStart*SensorSubSampling)
  {
    accum_xx/=ZoneCount;
    accum_yy/=ZoneCount;
    accum_zz/=ZoneCount;
    accum_xy/=ZoneCount;
    accum_xz/=ZoneCount;
    accum_yz/=ZoneCount;

    CurZone=0;
    index=IndN1N2N3(i,j,k,0);
    index2=N1*N2*N3;

	int bDoPrint=0;
	if (i==64 && j==64 && k==117)
		bDoPrint=1;

    if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
    {
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)+=accum_xx*accum_xx;
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)+=accum_yy*accum_yy;
        if (IS_Sigmazz_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmazz)+=accum_zz*accum_zz;
        if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)+=accum_xy*accum_xy;
        if (IS_Sigmaxz_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxz)+=accum_xz*accum_xz;
        if (IS_Sigmayz_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayz)+=accum_yz*accum_yz;
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
		{
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;
			#ifdef OPENCL
			if (bDoPrint)
				printf("Capturing RMS  Pressure %g,%g,%lu,%lu\n",ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure),
							DT,index,index2*IndexRMSPeak_Pressure);
			#endif
		}
		else{
		#ifdef OPENCL
		if (bDoPrint)
			printf("Capturing Pressure RMS not enabled \n");
		#endif
		}
    }
    if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
        index+=index2*NumberSelRMSPeakMaps;
    if (SelRMSorPeak & SEL_PEAK)
    {
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)=accum_xx>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx) ? accum_xx: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx);
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)=accum_yy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy) ? accum_yy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy);
        if (IS_Sigmazz_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmazz)=accum_zz>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmazz) ? accum_zz: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmazz);
        if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)=accum_xy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy) ? accum_xy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy);
        if (IS_Sigmaxz_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxz)=accum_xz>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxz) ? accum_xz: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxz);
        if (IS_Sigmayz_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayz)=accum_yz>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayz) ? accum_yz: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayz);
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
		{
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p :ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);
			#ifdef OPENCL
			if (bDoPrint)
				printf("Capturing peak  Pressure %g,%g\n",ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure),DT);
			#endif
		}
		else
		{
		#ifdef OPENCL
		if (bDoPrint)
			printf("Capturing Pressure Peak not enabled \n");
		#endif
		}
    }

  }
}

#if defined(CUDA)
__global__ void ParticleKernel(InputDataKernel * p,
			unsigned int nStep,unsigned int TypeSource)
{
	const _PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    const _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
    const _PT k = (_PT) (blockIdx.z * blockDim.z + threadIdx.z);
#endif
#ifdef OPENCL
__kernel void ParticleKernel(
__global mexType *V_x_x_pr,
__global mexType *V_y_x_pr,
__global mexType *V_z_x_pr,
__global mexType *V_x_y_pr,
__global mexType *V_y_y_pr,
__global mexType *V_z_y_pr,
__global mexType *V_x_z_pr,
__global mexType *V_y_z_pr,
__global mexType *V_z_z_pr,
__global mexType *Vx_pr,
__global mexType *Vy_pr,
__global mexType *Vz_pr,
__global mexType *Rxx_pr,
__global mexType *Ryy_pr,
__global mexType *Rzz_pr,
__global mexType *Rxy_pr,
__global mexType *Rxz_pr,
__global mexType *Ryz_pr,
__global mexType *Sigma_x_xx_pr,
__global mexType *Sigma_y_xx_pr,
__global mexType *Sigma_z_xx_pr,
__global mexType *Sigma_x_yy_pr,
__global mexType *Sigma_y_yy_pr,
__global mexType *Sigma_z_yy_pr,
__global mexType *Sigma_x_zz_pr,
__global mexType *Sigma_y_zz_pr,
__global mexType *Sigma_z_zz_pr,
__global mexType *Sigma_x_xy_pr,
__global mexType *Sigma_y_xy_pr,
__global mexType *Sigma_x_xz_pr,
__global mexType *Sigma_z_xz_pr,
__global mexType *Sigma_y_yz_pr,
__global mexType *Sigma_z_yz_pr,
__global mexType *Sigma_xy_pr,
__global mexType *Sigma_xz_pr,
__global mexType *Sigma_yz_pr,
__global mexType *Sigma_xx_pr,
__global mexType *Sigma_yy_pr,
__global mexType *Sigma_zz_pr,
__global mexType *SourceFunctions_pr,
__global mexType * LambdaMiuMatOverH_pr,
__global mexType * LambdaMatOverH_pr,
__global mexType * MiuMatOverH_pr,
__global mexType * TauLong_pr,
__global mexType * OneOverTauSigma_pr,
__global mexType * TauShear_pr,
__global mexType * InvRhoMatH_pr,
__global mexType * SqrAcc_pr,
__global unsigned int * MaterialMap_pr,
__global unsigned int * SourceMap_pr,
__global mexType * Ox_pr,
__global mexType * Oy_pr,
__global mexType * Oz_pr,
__global mexType * Pressure_pr
	, unsigned int nStep,
	unsigned int TypeSource)
{
	const _PT i = (_PT) get_global_id(0);
	const _PT j = (_PT) get_global_id(1);
	const _PT k = (_PT) get_global_id(2);
#endif
#ifdef METAL
kernel void ParticleKernel(
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


#ifdef USE_2ND_ORDER_EDGES
	interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
    _PT index,index2, CurZone;
	unsigned int source;
	mexType AvgInvRhoI,AvgInvRhoJ,AvgInvRhoK,Dx,Dy,Dz,Diff,value,accum_x=0.0,accum_y=0.0,accum_z=0.0;
			//accum_p=0.0;

	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		if (i<N1 && j<N2 && k<N3)
			{

		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 || IsOnPML_K(k)==1)
			{
				index=Ind_MaterialMap(i,j,k);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 && k<N3-1)
				{
					index=Ind_V_x_x(i,j,k);


		            Diff= i>0 && i<N1-2 ? CA*(EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k))-
		                                  CB*(EL(Sigma_xx,i+2,j,k)-EL(Sigma_xx,i-1,j,k))
					                      :i<N1-1 ? EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k):0;

					ELD(V_x_x,index) =InvDXDThp_I*(ELD(V_x_x,index)*DXDThp_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_x(i,j,k);
					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k))-
					                      CB*(EL(Sigma_xy,i,j+1,k)-EL(Sigma_xy,i,j-2,k))
					                      :j>0 && j<N2 ? EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k):0;

					ELD(V_y_x,index) =InvDXDT_J*(
													ELD(V_y_x,index)*DXDT_J+
													AvgInvRhoI*
													Diff);
					index=Ind_V_z_x(i,j,k);
					Diff= k >1 && k < N3-1 ? CA*( EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1))-
					                         CB*( EL(Sigma_xz,i,j,k+1)-EL(Sigma_xz,i,j,k-2)) :
		                                     k >0 && k < N3 ?  EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1):0;

					ELD(V_z_x,index) =InvDXDT_K*(
													ELD(V_z_x,index)*DXDT_K+
													AvgInvRhoI*
													Diff);


				// For coeffs. for V_y

					index=Ind_V_x_y(i,j,k);

					Diff= i>1 && i<N1-1 ? CA *(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k)) -
					                      CB *(EL(Sigma_xy,i+1,j,k)-EL(Sigma_xy,i-2,j,k))
					                      :i>0 && i<N1 ? EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k):0;

					ELD(V_x_y,index) =InvDXDT_I*(
													ELD(V_x_y,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_y(i,j,k);
					Diff= j>0 && j < N2-2 ? CA*( EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k)) -
					                        CB*( EL(Sigma_yy,i,j+2,k)-EL(Sigma_yy,i,j-1,k))
					                        :j < N2-1 ? EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k):0;

					ELD(V_y_y,index) =InvDXDThp_J*(
												ELD(V_y_y,index)*DXDThp_J+
												AvgInvRhoI*
												Diff);
					index=Ind_V_y_z(i,j,k);

					Diff= k>1  && k <N3-1 ? CA*(EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1) )-
					                        CB*(EL(Sigma_yz,i,j,k+1)-EL(Sigma_yz,i,j,k-2) )
					                        :k>0  && k <N3 ? EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1):0;

					ELD(V_z_y,index) =InvDXDT_K*(
												ELD(V_z_y,index)*DXDT_K+
												AvgInvRhoI*
												Diff);

					index=Ind_V_x_z(i,j,k);

					Diff= i> 1 && i <N1-1 ? CA*( EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k)) -
					                        CB*( EL(Sigma_xz,i+1,j,k)-EL(Sigma_xz,i-2,j,k))
					                        :i> 0 && i <N1 ? EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k):0;

					ELD(V_x_z,index) =InvDXDT_I*(
													ELD(V_x_z,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_z(i,j,k);

					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k)) -
					                      CB*(EL(Sigma_yz,i,j+1,k)-EL(Sigma_yz,i,j-2,k)):
		                                  j>0 && j<N2 ? EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k):0;

					ELD(V_y_z,index) =InvDXDT_J*(
												ELD(V_y_z,index)*DXDT_J+
												AvgInvRhoI*
												Diff);
					index=Ind_V_z_z(i,j,k);

					Diff= k>0 && k< N3-2 ? CA*(EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k) )-
					                       CB*(EL(Sigma_zz,i,j,k+2)-EL(Sigma_zz,i,j,k-1) ):
		                                   k< N3-1 ? EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k) :0;

					ELD(V_z_z,index) =InvDXDThp_K*(
												ELD(V_z_z,index)*DXDThp_K+
												AvgInvRhoI*
												Diff);
				 }

				 //We add now for Vx, Vy, Vz

				if ( j <N2 && k < N3)
				{
				   index=Ind_V_x(i,j,k);
				   index2=Ind_V_x_x(i,j,k);
				   ELD(Vx,index)=ELD(V_x_x,index2)+ELD(V_y_x,index2)+ELD(V_z_x,index2);
				}
				if (i<N1 && k < N3 )
				{
					index=Ind_V_y(i,j,k);
					index2=Ind_V_y_y(i,j,k);
					ELD(Vy,index)=ELD(V_x_y,index2)+ELD(V_y_y,index2)+ELD(V_z_y,index2);
				}
				if (i<N1 && j < N2 )
				{
					index=Ind_V_z(i,j,k);
					index2=Ind_V_z_z(i,j,k);
					ELD(Vz,index)=ELD(V_x_z,index2)+ELD(V_y_z,index2)+ELD(V_z_z,index2);
				}
			}
			else
			{
				index=Ind_MaterialMap(i,j,k);

				#ifdef USE_2ND_ORDER_EDGES
						unsigned int m1=ELD(MiuMatOverH,EL(MaterialMap,i,j,k));
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i+2,j,k))==0.0,m1== 0.0)
							interfaceX=interfaceX|frontLinep1;
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i+1,j,k))==0.0,m1 ==0.0)
							interfaceX=interfaceX|frontLine;
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i-1,j,k))==0.0,m1 ==0.0)
							interfaceX=interfaceX|backLine;
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j+2,k))==0.0,m1== 0.0)
							interfaceY=interfaceY|frontLinep1;
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j+1,k))==0.0,m1 ==0.0)
							interfaceY=interfaceY|frontLine;
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j-1,k))==0.0,m1 ==0.0)
							interfaceY=interfaceY|backLine;

						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k+2))==0.0 , m1== 0.0)
							interfaceZ=interfaceZ|frontLinep1;
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k+1))==0.0, m1 ==0.0)
							interfaceZ=interfaceZ|frontLine;
						if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k-1))==0.0, m1 ==0.0)
							interfaceZ=interfaceZ|backLine;
				#endif

				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j,k))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1,k))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				AvgInvRhoK=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j,k+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));

				if REQUIRES_2ND_ORDER_P(X)
					Dx=EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k);
				else
					Dx=CA*(EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k))-
						CB*(EL(Sigma_xx,i+2,j,k)-EL(Sigma_xx,i-1,j,k));

				if REQUIRES_2ND_ORDER_P(Y)
					Dx+=EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k);
				else
					Dx+=CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k))-
						CB*(EL(Sigma_xy,i,j+1,k)-EL(Sigma_xy,i,j-2,k));

				if REQUIRES_2ND_ORDER_P(Z)
					Dx+=EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1);
				else
					Dx+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1))-
						CB*(EL(Sigma_xz,i,j,k+1)-EL(Sigma_xz,i,j,k-2));

				EL(Vx,i,j,k)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j,k);
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

				if REQUIRES_2ND_ORDER_P(Y)
					Dy=EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k);
				else
					Dy=CA*(EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k) )-
						CB*(EL(Sigma_yy,i,j+2,k)-EL(Sigma_yy,i,j-1,k));

				if REQUIRES_2ND_ORDER_P(X)
					Dy+=EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k);
				else
					Dy+=CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k))-
						CB*(EL(Sigma_xy,i+1,j,k)-EL(Sigma_xy,i-2,j,k));

				if REQUIRES_2ND_ORDER_P(Z)
					Dy+=EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1);
				else
					Dy+=CA*( EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1))-
					CB*(EL(Sigma_yz,i,j,k+1)-EL(Sigma_yz,i,j,k-2));
				
				EL(Vy,i,j,k)+=DT*AvgInvRhoJ*Dy;
				accum_y+=EL(Vy,i,j,k);
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


				if REQUIRES_2ND_ORDER_P(Z)
					Dz=EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k);
				else
					Dz=CA*(EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k))-
						CB*( EL(Sigma_zz,i,j,k+2)-EL(Sigma_zz,i,j,k-1));

				if REQUIRES_2ND_ORDER_P(X)
					Dz+=EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k);
				else
					Dz+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k))-
					CB*(EL(Sigma_xz,i+1,j,k)-EL(Sigma_xz,i-2,j,k));

				if REQUIRES_2ND_ORDER_P(Y)
					Dz+=EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k);
				else
					Dz+=CA*( EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k))-
					CB*(EL(Sigma_yz,i,j+1,k)-EL(Sigma_yz,i,j-2,k));

				EL(Vz,i,j,k)+=DT*AvgInvRhoK*Dz;
				accum_z+=EL(Vz,i,j,k);

		}

  		if ((nStep < LengthSource) && TypeSource<2) //Source is particle displacement
  		{
			index=IndN1N2N3(i,j,k,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
				source--; //need to use C index
  			  	value=ELD(SourceFunctions,nStep*NumberSources+source);
				if (TypeSource==0)
				{
					EL(Vx,i,j,k)+=value*ELD(Ox,index);
					EL(Vy,i,j,k)+=value*ELD(Oy,index);
					EL(Vz,i,j,k)+=value*ELD(Oz,index);
				}
				else
				{
					EL(Vx,i,j,k)=value*ELD(Ox,index);
					EL(Vy,i,j,k)=value*ELD(Oy,index);
					EL(Vz,i,j,k)=value*ELD(Oz,index);
				}

  			}
  		}

		}
		if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0 && IsOnPML_K(k)==0 && nStep>=SensorStart*SensorSubSampling)
	    {
			if (ZoneCount>1)
			{
				accum_x/=ZoneCount;
				accum_y/=ZoneCount;
				accum_z/=ZoneCount;
			}
			CurZone=0;
			index=IndN1N2N3(i,j,k,0);
			index2=N1*N2*N3;
			if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
			{
				if (IS_ALLV_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV)+=accum_x*accum_x  +  accum_y*accum_y  +  accum_z*accum_z;
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
				if (IS_Vz_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vz)+=accum_z*accum_z;

			}
			if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
					index+=index2*NumberSelRMSPeakMaps;
			if (SelRMSorPeak & SEL_PEAK)
			{
				if (IS_ALLV_SELECTED(SelMapsRMSPeak))
				{
					value=accum_x*accum_x  +  accum_y*accum_y  +  accum_z*accum_z; //in the Python function we will do the final sqr root`
					ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV)=value > ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV) ? value : ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV);
				}
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
				if (IS_Vz_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vz)=accum_z > ELD(SqrAcc,index+index2*IndexRMSPeak_Vz) ? accum_z : ELD(SqrAcc,index+index2*IndexRMSPeak_Vz);

			}


		}
}


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
__global mexType *V_x_x_pr,
__global mexType *V_y_x_pr,
__global mexType *V_z_x_pr,
__global mexType *V_x_y_pr,
__global mexType *V_y_y_pr,
__global mexType *V_z_y_pr,
__global mexType *V_x_z_pr,
__global mexType *V_y_z_pr,
__global mexType *V_z_z_pr,
__global mexType *Vx_pr,
__global mexType *Vy_pr,
__global mexType *Vz_pr,
__global mexType *Rxx_pr,
__global mexType *Ryy_pr,
__global mexType *Rzz_pr,
__global mexType *Rxy_pr,
__global mexType *Rxz_pr,
__global mexType *Ryz_pr,
__global mexType *Sigma_x_xx_pr,
__global mexType *Sigma_y_xx_pr,
__global mexType *Sigma_z_xx_pr,
__global mexType *Sigma_x_yy_pr,
__global mexType *Sigma_y_yy_pr,
__global mexType *Sigma_z_yy_pr,
__global mexType *Sigma_x_zz_pr,
__global mexType *Sigma_y_zz_pr,
__global mexType *Sigma_z_zz_pr,
__global mexType *Sigma_x_xy_pr,
__global mexType *Sigma_y_xy_pr,
__global mexType *Sigma_x_xz_pr,
__global mexType *Sigma_z_xz_pr,
__global mexType *Sigma_y_yz_pr,
__global mexType *Sigma_z_yz_pr,
__global mexType *Sigma_xy_pr,
__global mexType *Sigma_xz_pr,
__global mexType *Sigma_yz_pr,
__global mexType *Sigma_xx_pr,
__global mexType *Sigma_yy_pr,
__global mexType *Sigma_zz_pr,
__global mexType *SourceFunctions_pr,
__global mexType * LambdaMiuMatOverH_pr,
__global mexType * LambdaMatOverH_pr,
__global mexType * MiuMatOverH_pr,
__global mexType * TauLong_pr,
__global mexType * OneOverTauSigma_pr,
__global mexType * TauShear_pr,
__global mexType * InvRhoMatH_pr,
__global mexType * SqrAcc_pr,
__global unsigned int * MaterialMap_pr,
__global unsigned int * SourceMap_pr,
__global mexType * Ox_pr,
__global mexType * Oy_pr,
__global mexType * Oz_pr,
__global mexType * Pressure_pr
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
_PT index=(((_PT)nStep)/((_PT)SensorSubSampling)-((_PT)SensorStart))*((_PT)NumberSensors)+(_PT)sj;
_PT  i,j,k;
_PT index2,index3,
    subarrsize=(((_PT)NumberSensors)*(((_PT)TimeSteps)/((_PT)SensorSubSampling)+1-((_PT)SensorStart)));
index2=IndexSensorMap_pr[sj]-1;

mexType accumX=0.0,accumY=0.0,accumZ=0.0,
        accumXX=0.0, accumYY=0.0, accumZZ=0.0,
        accumXY=0.0, accumXZ=0.0, accumYZ=0.0, accum_p=0;;
for (_PT CurZone=0;CurZone<ZoneCount;CurZone++)
  {
    k=index2/(N1*N2);
    j=index2%(N1*N2);
    i=j%(N1);
    j=j/N1;

    if (IS_ALLV_SELECTED(SelMapsSensors) || IS_Vx_SELECTED(SelMapsSensors))
        accumX+=EL(Vx,i,j,k);
    if (IS_ALLV_SELECTED(SelMapsSensors) || IS_Vy_SELECTED(SelMapsSensors))
        accumY+=EL(Vy,i,j,k);
    if (IS_ALLV_SELECTED(SelMapsSensors) || IS_Vz_SELECTED(SelMapsSensors))
        accumZ+=EL(Vz,i,j,k);

    index3=Ind_Sigma_xx(i,j,k);
  #ifdef METAL
    //No idea why in this kernel the ELD(SigmaXX...) macros do not expand correctly
    //So we go a bit more manual
  if (IS_Sigmaxx_SELECTED(SelMapsSensors))
      accumXX+=k_Sigma_xx_pr[index3];
  if (IS_Sigmayy_SELECTED(SelMapsSensors))
      accumYY+=k_Sigma_yy_pr[index3];
  if (IS_Sigmazz_SELECTED(SelMapsSensors))
      accumZZ+=k_Sigma_zz_pr[index3];
  if (IS_Pressure_SELECTED(SelMapsSensors))
      accum_p+=k_Pressure_pr[index3];
  index3=Ind_Sigma_xy(i,j,k);
  if (IS_Sigmaxy_SELECTED(SelMapsSensors))
      accumXY+=k_Sigma_xy_pr[index3];
  if (IS_Sigmaxz_SELECTED(SelMapsSensors))
      accumXZ+=k_Sigma_xz_pr[index3];
  if (IS_Sigmayz_SELECTED(SelMapsSensors))
      accumYZ+=k_Sigma_yz_pr[index3];
  
  #else
    if (IS_Sigmaxx_SELECTED(SelMapsSensors))
        accumXX+=ELD(Sigma_xx,index3);
    if (IS_Sigmayy_SELECTED(SelMapsSensors))
        accumYY+=ELD(Sigma_yy,index3);
    if (IS_Sigmazz_SELECTED(SelMapsSensors))
        accumZZ+=ELD(Sigma_zz,index3);
    if (IS_Pressure_SELECTED(SelMapsSensors))
        accum_p+=ELD(Pressure,index3);
    index3=Ind_Sigma_xy(i,j,k);
    if (IS_Sigmaxy_SELECTED(SelMapsSensors))
        accumXY+=ELD(Sigma_xy,index3);
    if (IS_Sigmaxz_SELECTED(SelMapsSensors))
        accumXZ+=ELD(Sigma_xz,index3);
    if (IS_Sigmayz_SELECTED(SelMapsSensors))
        accumYZ+=ELD(Sigma_yz,index3);
   #endif
  }
accumX/=ZoneCount;
accumY/=ZoneCount;
accumZ/=ZoneCount;
accumXX/=ZoneCount;
accumYY/=ZoneCount;
accumZZ/=ZoneCount;
accumXY/=ZoneCount;
accumXZ/=ZoneCount;
accumYZ/=ZoneCount;
accum_p/=ZoneCount;
//ELD(SensorOutput,index)=accumX*accumX+accumY*accumY+accumZ*accumZ;
if (IS_ALLV_SELECTED(SelMapsSensors))
      ELD(SensorOutput,index+subarrsize*IndexSensor_ALLV)=
        (accumX*accumX*+accumY*accumY+accumZ*accumZ);
if (IS_Vx_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Vx)=accumX;
if (IS_Vy_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Vy)=accumY;
if (IS_Vz_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Vz)=accumZ;
if (IS_Sigmaxx_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmaxx)=accumXX;
if (IS_Sigmayy_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmayy)=accumYY;
if (IS_Sigmazz_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmazz)=accumZZ;
if (IS_Sigmaxy_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmaxy)=accumXY;
if (IS_Sigmaxz_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmaxz)=accumXZ;
if (IS_Sigmayz_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmayz)=accumYZ;
if (IS_Pressure_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Pressure)=accum_p;

}