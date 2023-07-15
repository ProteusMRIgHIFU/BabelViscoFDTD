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
#define __PRE_MAT 
#elif defined(METAL)
#define __PRE_MAT k_
#else
#define __PRE_MAT
#endif

#if defined(METAL)
#define EL(_Mat,_i,_j) CONCAT(__PRE_MAT,_Mat ## _pr[Ind_##_Mat(_i,_j)])
#define ELD(_Mat,_index) CONCAT(__PRE_MAT,_Mat ## _pr[_index])
#else
#define EL(_Mat,_i,_j) __PRE_MAT _Mat##_pr[Ind_##_Mat(_i,_j)]
#define ELD(_Mat,_index) __PRE_MAT _Mat##_pr[_index]
#endif

#define ELO(_Mat,_i,_j)  _Mat##_pr[Ind_##_Mat(_i,_j)]
#define ELDO(_Mat,_index)  _Mat##_pr[_index]

#define hELO(_Mat,_i,_j)  _Mat##_pr[hInd_##_Mat(_i,_j)]


//////////////////////////////////////////////
#define hInd_Source(a,b)((b)*INHOST(N2)+a)

#define hIndN1N2Snap(a,b) ((b)*INHOST(N1)+a)
#define hIndN1N2(a,b,_ZoneSize)   ((b)*INHOST(N1)    +a+(CurZone*(_ZoneSize)))
#define hIndN1p1N2(a,b,_ZoneSize) ( (b)*(INHOST(N1)+1)+a+(CurZone*(_ZoneSize)))
#define hIndN1N2p1(a,b,_ZoneSize) ((b)*(INHOST(N1))  +a+(CurZone*(_ZoneSize)))
#define hIndN1N2(a,b,_ZoneSize) ((b)*INHOST(N1)    +a+(CurZone*(_ZoneSize)))
#define hIndN1p1N2p1(a,b,_ZoneSize) ((b)*(INHOST(N1)+1)+a +(CurZone*(_ZoneSize)))

#define hCorrecI(_i,_j) ((_j)>hLimit_J_low_PML && (_j)<hLimit_J_up_PML && (_i)> hLimit_I_low_PML ?hSizeCorrI :0)
#define hCorrecJ(_j) ((_j)>hLimit_J_low_PML+1 ?((_j)<hLimit_J_up_PML?((_j)-hLimit_J_low_PML-1)*(hSizeCorrI):hSizeCorrI*hSizeCorrJ):0)

#define hIndexPML(_i,_j,_ZoneSize)  (hIndN1N2(_i,_j,_ZoneSize) - hCorrecI(_i,_j) - hCorrecJ(_j))

#define hIndexPMLxp1(_i,_j,_ZoneSize) (hIndN1p1N2(_i,_j,_ZoneSize) - hCorrecI(_i,_j) )
#define hIndexPMLyp1(_i,_j,_ZoneSize) (hIndN1N2p1(_i,_j,_ZoneSize) - hCorrecI(_i,_j) )

#define hInd_MaterialMap(_i,_j) (hIndN1p1N2p1(_i,_j,(INHOST(N1)+1)*(INHOST(N2)+1)))

#define hInd_V_x(_i,_j) (hIndN1p1N2(_i,_j,(INHOST(N1)+1)*INHOST(N2)))
#define hInd_V_y(_i,_j) (hIndN1N2p1(_i,_j,INHOST(N1)*(INHOST(N2)+1)))

#define hInd_Vx(_i,_j) (hIndN1p1N2(_i,_j,(INHOST(N1)+1)*INHOST(N2)))
#define hInd_Vy(_i,_j) (hIndN1N2p1(_i,_j,INHOST(N1)*(INHOST(N2)+1)))

#define hInd_Sigma_xx(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))
#define hInd_Sigma_yy(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))

#define hInd_Pressure(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))
#define hInd_Pressure_old(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))

#define hInd_Sigma_xy(_i,_j) (hIndN1p1N2p1(_i,_j,(INHOST(N1)+1)*(INHOST(N2)+1)))

#define hInd_SqrAcc(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))

#define hInd_V_x_x(_i,_j) (hIndexPMLxp1(_i,_j,INHOST(SizePMLxp1)))
#define hInd_V_y_x(_i,_j) (hIndexPMLxp1(_i,_j,INHOST(SizePMLxp1)))

#define hInd_V_x_y(_i,_j) (hIndexPMLyp1(_i,_j,INHOST(SizePMLyp1)))
#define hInd_V_y_y(_i,_j) (hIndexPMLyp1(_i,_j,INHOST(SizePMLyp1)))



#define hInd_Sigma_x_xx(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )
#define hInd_Sigma_y_xx(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )

#define hInd_Sigma_x_yy(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )
#define hInd_Sigma_y_yy(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )


#define IsOnPML_I(_i) ((_i) <=Limit_I_low_PML || (_i)>=Limit_I_up_PML ? 1:0)
#define IsOnPML_J(_j) ((_j) <=Limit_J_low_PML || (_j)>=Limit_J_up_PML ? 1:0)
#define IsOnPML_K(_k) ((_k) <=Limit_K_low_PML || (_k)>=Limit_K_up_PML ? 1:0)

#define IsOnLowPML_I(_i) (_i) <=Limit_I_low_PML
#define IsOnLowPML_J(_j) (_j) <=Limit_J_low_PML
#define IsOnLowPML_K(_k) (_k) <=Limit_K_low_PML

////////////////////////////////////////
#define Ind_Source(a,b)((b)*N2+a)

#define IndN1N2Snap(a,b) ((b)*N1+a)

#define IndN1N2(a,b,_ZoneSize)   ((b)*N1    +a+(CurZone*(_ZoneSize)))
#define IndN1p1N2(a,b,_ZoneSize) ( (b)*(N1+1)+a+(CurZone*(_ZoneSize)))
#define IndN1N2p1(a,b,_ZoneSize) ((b)*N1 +a+(CurZone*(_ZoneSize)))
#define IndN1N2(a,b,_ZoneSize) ((b)*N1   +a+(CurZone*(_ZoneSize)))
#define IndN1p1N2p1(a,b,_ZoneSize) ((b)*(N1+1)+a +(CurZone*(_ZoneSize)))

#define CorrecI(_i,_j) ((_j)>Limit_J_low_PML  && (_j)<Limit_J_up_PML  && (_i)> Limit_I_low_PML ?SizeCorrI :0)
#define CorrecJ(_j) ( (_j)>Limit_J_low_PML+1 ?((_j)<Limit_J_up_PML?((_j)-Limit_J_low_PML-1)*(SizeCorrI):SizeCorrI*SizeCorrJ):0)

#define IndexPML(_i,_j,_ZoneSize)  (IndN1N2(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))

#define IndexPMLxp1(_i,_j,_ZoneSize) (IndN1p1N2(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))
#define IndexPMLyp1(_i,_j,_ZoneSize) (IndN1N2p1(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))
#define IndexPMLxp1yp1(_i,_j,_ZoneSize) (IndN1p1N2p1(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))

#define Ind_MaterialMap(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))

#define Ind_V_x(_i,_j) (IndN1p1N2(_i,_j,(N1+1)*N2))
#define Ind_V_y(_i,_j) (IndN1N2p1(_i,_j,N1*(N2+1)))


#define Ind_Vx(_i,_j) (IndN1p1N2(_i,_j,(N1+1)*N2))
#define Ind_Vy(_i,_j) (IndN1N2p1(_i,_j,N1*(N2+1)))
#define Ind_Vz(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_Sigma_xx(_i,_j) (IndN1N2(_i,_j,N1*N2))
#define Ind_Sigma_yy(_i,_j) (IndN1N2(_i,_j,N1*N2))
#define Ind_Sigma_zz(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_Pressure(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_Sigma_xy(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))
#define Ind_Sigma_xz(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))
#define Ind_Sigma_yz(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))

#define Ind_SqrAcc(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_V_x_x(_i,_j) (IndexPMLxp1(_i,_j,SizePMLxp1))
#define Ind_V_y_x(_i,_j) (IndexPMLxp1(_i,_j,SizePMLxp1))

#define Ind_V_x_y(_i,_j) (IndexPMLyp1(_i,_j,SizePMLyp1))
#define Ind_V_y_y(_i,_j) (IndexPMLyp1(_i,_j,SizePMLyp1))


#define Ind_Sigma_x_xx(_i,_j) (IndexPML(_i,_j,SizePML) )
#define Ind_Sigma_y_xx(_i,_j) (IndexPML(_i,_j,SizePML) )

#define Ind_Sigma_x_yy(_i,_j) (IndexPML(_i,_j,SizePML) )
#define Ind_Sigma_y_yy(_i,_j) (IndexPML(_i,_j,SizePML) )

#define Ind_Sigma_x_xy(_i,_j)(IndexPMLxp1yp1(_i,_j,SizePMLxp1yp1) )
#define Ind_Sigma_y_xy(_i,_j)(IndexPMLxp1yp1(_i,_j,SizePMLxp1yp1) )


#define iPML(_i) ((_i) <=Limit_I_low_PML ? (_i) : ((_i)<Limit_I_up_PML ? PML_Thickness : (_i)<N1 ? PML_Thickness-1-(_i)+Limit_I_up_PML:0))
#define jPML(_j) ((_j) <=Limit_J_low_PML ? (_j) : ((_j)<Limit_J_up_PML ? PML_Thickness : (_j)<N2 ? PML_Thickness-1-(_j)+Limit_J_up_PML:0))


#if defined(CUDA) || defined(OPENCL)
#define InvDXDT_I 	(IsOnLowPML_I(i) ? gpuInvDXDTpluspr[iPML(i)] : gpuInvDXDTplushppr[iPML(i)] )
#define DXDT_I 		(IsOnLowPML_I(i) ? gpuDXDTminuspr[iPML(i)] : gpuDXDTminushppr[iPML(i)] )
#define InvDXDT_J 	(IsOnLowPML_J(j) ? gpuInvDXDTpluspr[jPML(j)] : gpuInvDXDTplushppr[jPML(j)] )
#define DXDT_J 		(IsOnLowPML_J(j) ? gpuDXDTminuspr[jPML(j)] : gpuDXDTminushppr[jPML(j)] )

#define InvDXDThp_I 	(IsOnLowPML_I(i) ? gpuInvDXDTplushppr[iPML(i)] : gpuInvDXDTpluspr[iPML(i)] )
#define DXDThp_I 		(IsOnLowPML_I(i) ? gpuDXDTminushppr[iPML(i)] : gpuDXDTminuspr[iPML(i)] )
#define InvDXDThp_J 	(IsOnLowPML_J(j) ? gpuInvDXDTplushppr[jPML(j)] : gpuInvDXDTpluspr[jPML(j)] )
#define DXDThp_J 		(IsOnLowPML_J(j) ? gpuDXDTminushppr[jPML(j)] : gpuDXDTminuspr[jPML(j)] )
#else
#define InvDXDT_I 	(IsOnLowPML_I(i) ? InvDXDTplus_pr[iPML(i)] : InvDXDTplushp_pr[iPML(i)] )
#define DXDT_I 		(IsOnLowPML_I(i) ? DXDTminus_pr[iPML(i)] : DXDTminushp_pr[iPML(i)] )
#define InvDXDT_J 	(IsOnLowPML_J(j) ? InvDXDTplus_pr[jPML(j)] : InvDXDTplushp_pr[jPML(j)] )
#define DXDT_J 		(IsOnLowPML_J(j) ? DXDTminus_pr[jPML(j)] : DXDTminushp_pr[jPML(j)] )

#define InvDXDThp_I 	(IsOnLowPML_I(i) ? InvDXDTplushp_pr[iPML(i)] : InvDXDTplus_pr[iPML(i)] )
#define DXDThp_I 		(IsOnLowPML_I(i) ? DXDTminushp_pr[iPML(i)] : DXDTminus_pr[iPML(i)] )
#define InvDXDThp_J 	(IsOnLowPML_J(j) ? InvDXDTplushp_pr[jPML(j)] : InvDXDTplus_pr[jPML(j)] )
#define DXDThp_J 		(IsOnLowPML_J(j) ? DXDTminushp_pr[jPML(j)] : DXDTminus_pr[jPML(j)] )
#endif

#define MASK_Vx   			0x0000000001
#define MASK_Vy   			0x0000000002
#define MASK_Sigmaxx    	0x0000000004
#define MASK_Sigmayy    	0x0000000008
#define MASK_Sigmaxy    	0x0000000010
#define MASK_Pressure      	0x0000000020
#define MASK_Pressure_Gx   	0x0000000040
#define MASK_Pressure_Gy   	0x0000000080

#define IS_Vx_SELECTED(_Value) 					(_Value &MASK_Vx)
#define IS_Vy_SELECTED(_Value) 					(_Value &MASK_Vy)
#define IS_Sigmaxx_SELECTED(_Value) 			(_Value &MASK_Sigmaxx)
#define IS_Sigmayy_SELECTED(_Value) 			(_Value &MASK_Sigmayy)
#define IS_Sigmaxy_SELECTED(_Value) 			(_Value &MASK_Sigmaxy)
#define IS_Pressure_SELECTED(_Value) 			(_Value &MASK_Pressure)
#define IS_Pressure_Gx_SELECTED(_Value) 		(_Value &MASK_Pressure_Gx)
#define IS_Pressure_Gy_SELECTED(_Value) 		(_Value &MASK_Pressure_Gy)

#define COUNT_SELECTIONS(_VarName,_Value) \
				{ _VarName =0;\
					_VarName += IS_Vx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Vy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmayy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Pressure_SELECTED(_Value) ? 1 : 0;\
					_VarName += IS_Pressure_Gx_SELECTED(_Value) ? 1 : 0;\
					_VarName += IS_Pressure_Gy_SELECTED(_Value) ? 1 : 0}

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
#ifndef METALCOMPUTE
#define CInd_N1 0
#define CInd_N2 1
#define CInd_Limit_I_low_PML 2
#define CInd_Limit_J_low_PML 3
#define CInd_Limit_I_up_PML 4
#define CInd_Limit_J_up_PML 5
#define CInd_SizeCorrI 6
#define CInd_SizeCorrJ 7
#define CInd_PML_Thickness 8
#define CInd_NumberSources 9
#define CInd_NumberSensors 10
#define CInd_TimeSteps 11
#define CInd_SizePML 12
#define CInd_SizePMLxp1 13
#define CInd_SizePMLyp1 14
#define CInd_SizePMLxp1yp1 15
#define CInd_ZoneCount 16
#define CInd_SelRMSorPeak 17
#define CInd_SelMapsRMSPeak 18
#define CInd_IndexRMSPeak_Vx 19
#define CInd_IndexRMSPeak_Vy 20
#define CInd_IndexRMSPeak_Sigmaxx 21
#define CInd_IndexRMSPeak_Sigmayy 22
#define CInd_IndexRMSPeak_Sigmaxy 23
#define CInd_NumberSelRMSPeakMaps 24
#define CInd_SelMapsSensors 25
#define CInd_IndexSensor_Vx 26
#define CInd_IndexSensor_Vy 27
#define CInd_IndexSensor_Sigmaxx 28
#define CInd_IndexSensor_Sigmayy 29
#define CInd_IndexSensor_Sigmaxy 30
#define CInd_NumberSelSensorMaps 31
#define CInd_SensorSubSampling 32
#define CInd_nStep 33
#define CInd_TypeSource 34
#define CInd_CurrSnap 35
#define CInd_LengthSource 36
#define CInd_IndexRMSPeak_Pressure 37
#define CInd_IndexSensor_Pressure 38
#define CInd_IndexSensor_Pressure_gx 39
#define CInd_IndexSensor_Pressure_gy 40
#define CInd_SensorStart 41


//Make LENGTH_CONST_UINT one value larger than the last index
#define LENGTH_CONST_UINT 42

//Indexes for float
#define CInd_DT 0
#define CInd_InvDXDTplus 1
#define CInd_DXDTminus (1+MAX_SIZE_PML)
#define CInd_InvDXDTplushp (1+MAX_SIZE_PML*2)
#define CInd_DXDTminushp (1+MAX_SIZE_PML*3)
//Make LENGTH_CONST_MEX one value larger than the last index
#define LENGTH_CONST_MEX (1+MAX_SIZE_PML*4)
#else
#define CInd_nStep 0
#define CInd_TypeSource 1
#define LENGTH_CONST_UINT 2
#endif

#define CInd_V_x_x 0
#define CInd_V_y_x 1
#define CInd_V_x_y 2
#define CInd_V_y_y 3

#define CInd_Vx 4
#define CInd_Vy 5

#define CInd_Rxx 6
#define CInd_Ryy 7

#define CInd_Rxy 8

#define CInd_Sigma_x_xx 9
#define CInd_Sigma_y_xx 10
#define CInd_Sigma_x_yy 11
#define CInd_Sigma_y_yy 12

#define CInd_Sigma_x_xy 13
#define CInd_Sigma_y_xy 14

#define CInd_Sigma_xy 15

#define CInd_Sigma_xx 16
#define CInd_Sigma_yy 17

#define CInd_SourceFunctions 18

#define CInd_LambdaMiuMatOverH  19
#define CInd_LambdaMatOverH	 20
#define CInd_MiuMatOverH 21
#define CInd_TauLong 22
#define CInd_OneOverTauSigma	23
#define CInd_TauShear 24
#define CInd_InvRhoMatH	25
#define CInd_Ox 26
#define CInd_Oy 27
#define CInd_Pressure 28

#define CInd_SqrAcc 29

#define CInd_SensorOutput 30

#define LENGTH_INDEX_MEX 31

#define CInd_IndexSensorMap  0
#define CInd_SourceMap	1
#define CInd_MaterialMap 2

#define LENGTH_INDEX_UINT 3

#endif

#endif
