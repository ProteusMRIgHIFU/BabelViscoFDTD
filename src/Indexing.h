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


#define MASK_ALLV		0x0000000001
#define MASK_Vx   		0x0000000002
#define MASK_Vy   		0x0000000004
#define MASK_Vz   		0x0000000008
#define MASK_Sigmaxx    0x0000000010
#define MASK_Sigmayy    0x0000000020
#define MASK_Sigmazz    0x0000000040
#define MASK_Sigmaxy    0x0000000080
#define MASK_Sigmaxz    0x0000000100
#define MASK_Sigmayz    0x0000000200
#define MASK_Pressure   0x0000000400

#define IS_ALLV_SELECTED(_Value) 				(_Value &MASK_ALLV)
#define IS_Vx_SELECTED(_Value) 					(_Value &MASK_Vx)
#define IS_Vy_SELECTED(_Value) 					(_Value &MASK_Vy)
#define IS_Vz_SELECTED(_Value) 					(_Value &MASK_Vz)
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
