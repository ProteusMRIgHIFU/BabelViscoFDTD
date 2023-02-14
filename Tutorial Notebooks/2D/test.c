#define mexType float
#define METAL
#define METALCOMPUTE
#define MAX_SIZE_PML 101
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

#define MASK_Vx   		0x0000000001
#define MASK_Vy   		0x0000000002
#define MASK_Sigmaxx    0x0000000004
#define MASK_Sigmayy    0x0000000008
#define MASK_Sigmaxy    0x0000000010
#define MASK_Pressure   0x0000000020

#define IS_Vx_SELECTED(_Value) 					(_Value &MASK_Vx)
#define IS_Vy_SELECTED(_Value) 					(_Value &MASK_Vy)
#define IS_Sigmaxx_SELECTED(_Value) 			(_Value &MASK_Sigmaxx)
#define IS_Sigmayy_SELECTED(_Value) 			(_Value &MASK_Sigmayy)
#define IS_Sigmaxy_SELECTED(_Value) 			(_Value &MASK_Sigmaxy)
#define IS_Pressure_SELECTED(_Value) 			(_Value &MASK_Pressure)

#define COUNT_SELECTIONS(_VarName,_Value) \
				{ _VarName =0;\
					_VarName += IS_Vx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Vy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmayy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxy_SELECTED(_Value) ? 1 : 0; \
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
#define CInd_SensorStart 39

//Make LENGTH_CONST_UINT one value larger than the last index
#define LENGTH_CONST_UINT 40

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
constant float DT = 1.50000005e-07;
constant  _PT N1 = 129;
constant  _PT N2 = 234;
constant  _PT Limit_I_low_PML = 11;
constant  _PT Limit_J_low_PML = 11;
constant  _PT Limit_I_up_PML = 117;
constant  _PT Limit_J_up_PML = 222;
constant  _PT SizeCorrI = 105;
constant  _PT SizeCorrJ = 210;
constant  _PT PML_Thickness = 12;
constant  _PT NumberSources = 1;
constant  _PT LengthSource = 78;
constant  _PT ZoneCount = 1;
constant  _PT SizePMLxp1 = 8371;
constant  _PT SizePMLyp1 = 8266;
constant  _PT SizePML = 8137;
constant  _PT SizePMLxp1yp1 = 8501;
constant  _PT NumberSensors = 22050;
constant  _PT TimeSteps = 497;
constant  _PT SelRMSorPeak = 1;
constant  _PT SelMapsRMSPeak = 32;
constant  _PT IndexRMSPeak_Vx = 0;
constant  _PT IndexRMSPeak_Vy = 0;
constant  _PT IndexRMSPeak_Sigmaxx = 0;
constant  _PT IndexRMSPeak_Sigmayy = 0;
constant  _PT IndexRMSPeak_Sigmaxy = 0;
constant  _PT IndexRMSPeak_Pressure = 0;
constant  _PT NumberSelRMSPeakMaps = 1;
constant  _PT SelMapsSensors = 35;
constant  _PT IndexSensor_Vx = 0;
constant  _PT IndexSensor_Vy = 1;
constant  _PT IndexSensor_Sigmaxx = 0;
constant  _PT IndexSensor_Sigmayy = 0;
constant  _PT IndexSensor_Sigmaxy = 0;
constant  _PT IndexSensor_Pressure = 2;
constant  _PT NumberSelSensorMaps = 3;
constant  _PT SensorSubSampling = 2;
constant  _PT SensorStart = 0;
constant float InvDXDTplus_pr[13] ={
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
constant float DXDTminus_pr[13] ={
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
constant float InvDXDTplushp_pr[13] ={
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
constant float DXDTminushp_pr[13] ={
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
#ifndef METALCOMPUTE
#define N1 p_CONSTANT_BUFFER_UINT[CInd_N1]
#define N2 p_CONSTANT_BUFFER_UINT[CInd_N2]
#define Limit_I_low_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_I_low_PML]
#define Limit_J_low_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_J_low_PML]
#define Limit_I_up_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_I_up_PML]
#define Limit_J_up_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_J_up_PML]
#define SizeCorrI p_CONSTANT_BUFFER_UINT[CInd_SizeCorrI]
#define SizeCorrJ p_CONSTANT_BUFFER_UINT[CInd_SizeCorrJ]
#define PML_Thickness p_CONSTANT_BUFFER_UINT[CInd_PML_Thickness]
#define NumberSources p_CONSTANT_BUFFER_UINT[CInd_NumberSources]
#define LengthSource p_CONSTANT_BUFFER_UINT[CInd_LengthSource]
#define NumberSensors p_CONSTANT_BUFFER_UINT[CInd_NumberSensors]
#define TimeSteps p_CONSTANT_BUFFER_UINT[CInd_TimeSteps]

#define SizePML p_CONSTANT_BUFFER_UINT[CInd_SizePML]
#define SizePMLxp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLxp1]
#define SizePMLyp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLyp1]
#define SizePMLxp1yp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLxp1yp1]
#define ZoneCount p_CONSTANT_BUFFER_UINT[CInd_ZoneCount]

#define SelRMSorPeak p_CONSTANT_BUFFER_UINT[CInd_SelRMSorPeak]
#define SelMapsRMSPeak p_CONSTANT_BUFFER_UINT[CInd_SelMapsRMSPeak]
#define IndexRMSPeak_Vx p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Vx]
#define IndexRMSPeak_Vy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Vy]
#define IndexRMSPeak_Sigmaxx p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmaxx]
#define IndexRMSPeak_Sigmayy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmayy]
#define IndexRMSPeak_Sigmaxy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmaxy]
#define IndexRMSPeak_Pressure p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Pressure]
#define NumberSelRMSPeakMaps p_CONSTANT_BUFFER_UINT[CInd_NumberSelRMSPeakMaps]

#define SelMapsSensors p_CONSTANT_BUFFER_UINT[CInd_SelMapsSensors]
#define IndexSensor_Vx p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Vx]
#define IndexSensor_Vy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Vy]
#define IndexSensor_Sigmaxx p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmaxx]
#define IndexSensor_Sigmayy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmayy]
#define IndexSensor_Sigmaxy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmaxy]
#define IndexSensor_Pressure p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Pressure]
#define NumberSelSensorMaps p_CONSTANT_BUFFER_UINT[CInd_NumberSelSensorMaps]
#define SensorSubSampling p_CONSTANT_BUFFER_UINT[CInd_SensorSubSampling]
#define SensorStart p_CONSTANT_BUFFER_UINT[CInd_SensorStart]
#define nStep p_CONSTANT_BUFFER_UINT[CInd_nStep]
#define CurrSnap p_CONSTANT_BUFFER_UINT[CInd_CurrSnap]
#define TypeSource p_CONSTANT_BUFFER_UINT[CInd_TypeSource]

#define DT p_CONSTANT_BUFFER_MEX[CInd_DT]
#define InvDXDTplus_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplus)
#define DXDTminus_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminus)
#define InvDXDTplushp_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplushp)
#define DXDTminushp_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminushp)
#else
#define nStep p_CONSTANT_BUFFER_UINT[CInd_nStep]
#define TypeSource p_CONSTANT_BUFFER_UINT[CInd_TypeSource]
#endif

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
#define k_V_x_y_pr  __def_MEX_VAR_0(V_x_y)
#define k_V_y_y_pr  __def_MEX_VAR_0(V_y_y)

#define k_Vx_pr  __def_MEX_VAR_1(Vx)
#define k_Vy_pr  __def_MEX_VAR_1(Vy)

#define k_Rxx_pr  __def_MEX_VAR_2(Rxx)
#define k_Ryy_pr  __def_MEX_VAR_2(Ryy)

#define k_Rxy_pr  __def_MEX_VAR_3(Rxy)

#define k_Sigma_x_xx_pr  __def_MEX_VAR_4(Sigma_x_xx)
#define k_Sigma_y_xx_pr  __def_MEX_VAR_4(Sigma_y_xx)
#define k_Sigma_x_yy_pr  __def_MEX_VAR_4(Sigma_x_yy)
#define k_Sigma_y_yy_pr  __def_MEX_VAR_4(Sigma_y_yy)

#define k_Sigma_x_xy_pr  __def_MEX_VAR_5(Sigma_x_xy)
#define k_Sigma_y_xy_pr  __def_MEX_VAR_5(Sigma_y_xy)

#define k_Sigma_xy_pr  __def_MEX_VAR_6(Sigma_xy)
#define k_Sigma_xx_pr  __def_MEX_VAR_7(Sigma_xx)
#define k_Sigma_yy_pr  __def_MEX_VAR_7(Sigma_yy)

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
#define k_Pressure_pr  __def_MEX_VAR_9(Pressure)

#define k_SqrAcc_pr  __def_MEX_VAR_10(SqrAcc)

#define k_SensorOutput_pr  __def_MEX_VAR_11(SensorOutput)

#define k_IndexSensorMap_pr  __def_UINT_VAR(IndexSensorMap)
#define k_SourceMap_pr		 __def_UINT_VAR(SourceMap)
#define k_MaterialMap_pr	 __def_UINT_VAR(MaterialMap)

#ifdef METALCOMPUTE
#define CGID uint
#else
#define CGID uint3
#endif
#ifndef METALCOMPUTE
#define METAL_PARAMS\
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],\
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],\
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],\
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],\
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],\
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],\
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],\
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],\
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],\
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],\
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],\
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],\
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],\
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],\
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],\
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],\
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],\
	CGID gid[[thread_position_in_grid]])\
{
#else
#define METAL_PARAMS\
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],\
	const device unsigned int *p_INDEX_MEX [[ buffer(1) ]],\
	const device unsigned int *p_INDEX_UINT [[ buffer(2) ]],\
	const device unsigned int *p_UINT_BUFFER [[ buffer(3) ]],\
	device mexType * p_MEX_BUFFER_0 [[ buffer(4) ]],\
	device mexType * p_MEX_BUFFER_1 [[ buffer(5) ]],\
	device mexType * p_MEX_BUFFER_2 [[ buffer(6) ]],\
	device mexType * p_MEX_BUFFER_3 [[ buffer(7) ]],\
	device mexType * p_MEX_BUFFER_4 [[ buffer(8) ]],\
	device mexType * p_MEX_BUFFER_5 [[ buffer(9) ]],\
	device mexType * p_MEX_BUFFER_6 [[ buffer(10) ]],\
	device mexType * p_MEX_BUFFER_7 [[ buffer(11) ]],\
	device mexType * p_MEX_BUFFER_8 [[ buffer(12) ]],\
	device mexType * p_MEX_BUFFER_9 [[ buffer(13) ]],\
	device mexType * p_MEX_BUFFER_10 [[ buffer(14) ]],\
	device mexType * p_MEX_BUFFER_11 [[ buffer(15) ]],\
	CGID gid[[thread_position_in_grid]])\
{
#endif
#endif
/// PMLS
#if defined(METAL)  || defined(USE_MINI_KERNELS_CUDA)
#define _ST_PML_1
#define _ST_PML_2
#define _ST_PML_3
#define _PML_KERNEL_CORNER
#ifdef CUDA
extern "C" __global__ void PML_1_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void PML_1_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
kernel void PML_1_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
	#else
	#define nN1 (PML_Thickness*2)
	#define nN2 (PML_Thickness*2)
  	_PT j = (_PT) ((gid )/nN1);
  	_PT i = (_PT) (gid-j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	// Each i,jgo from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1)
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML
#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size
#endif

#if defined(_MAIN_KERNEL) 
	i+=PML_Thickness;
	j+=PML_Thickness;
#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
	
#if defined(_ST_PML_1) || defined(_ST_PML_2)   
	mexType Diff;
#endif
#if defined(_ST_PML_1) || defined(_ST_PML_2)
	mexType Diff2;
#endif


#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2)  
	mexType Dx;
#endif
#if defined(_ST_MAIN_1)  
	mexType Dy;
#endif

#if defined(_ST_MAIN_1) || defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  
	mexType value;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)
	mexType m1;
	mexType m2;
	mexType m3;
	mexType m4;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2) 
	mexType RigidityXY=0.0;
#endif

#if defined(_ST_MAIN_2) 
	mexType TauShearXY=0.0;
#endif
#if defined(_ST_MAIN_1)
	mexType LambdaMiu;
	mexType LambdaMiuComp;
	mexType dFirstPart;
	mexType dFirstPartForR;
	mexType accum_xx=0.0;
	mexType accum_yy=0.0;
	mexType accum_p=0.0;
	_PT source;
	_PT bAttenuating=1;
#endif
#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2) 
	mexType Miu;
	mexType MiuComp;
	mexType OneOverTauSigma;
	mexType NextR;
#endif

#if defined(_ST_MAIN_2)
	mexType accum_xy=0.0;
#endif



#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
//#if defined(_ST_MAIN_1) || defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) ||  defined(_ST_PML_4) ||  defined(_ST_PML_5) ||  defined(_ST_PML_6)  
_PT index2;
//#endif
_PT index;
_PT  MaterialID;

_PT CurZone;

for ( CurZone=0;CurZone<ZoneCount;CurZone++)
  {
	  index=Ind_MaterialMap(i,j);
      MaterialID=ELD(MaterialMap,index);

	  #if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  

  		m1=ELD(MiuMatOverH,MaterialID);
  		m2=ELD(MiuMatOverH,EL(MaterialMap,i+1,j));
  		m3=ELD(MiuMatOverH,EL(MaterialMap,i,j+1));
  		m4=ELD(MiuMatOverH,EL(MaterialMap,i+1,j+1));
   		value=m1*m2*m3*m4;
  		RigidityXY =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
      #endif

	  #if  defined(_ST_MAIN_2) 
  		TauShearXY = value!=0.0 ? 0.25*(ELD(TauShear,MaterialID) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j)) +
  							 ELD(TauShear,EL(MaterialMap,i,j+1)) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j+1)))
  							 : ELD(TauShear,MaterialID);

	   #endif
	   
  	
  	if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )//We are in the PML borders
  	 {

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  		if (i<N1-1 && j <N2-1 )
  		{

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  			Diff= i>1 && i <N1-1 ? CA*(EL(Vx,i,j)-EL(Vx,i-1,j)) -
  			                       CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j))
  			      : i>0 && i <N1 ? (EL(Vx,i,j)-EL(Vx,i-1,j))  :0;

			Diff2= j>1 && j < N2-1 ? CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
  									CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2))
  			        : j>0 && j < N2 ? EL(Vy,i,j)-EL(Vy,i,j-1):0;

#endif

#if defined(_ST_PML_1)
  			
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_x_xx,index2) =InvDXDT_I*(
  											ELD(Sigma_x_xx,index2)*DXDT_I+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff);
			index2=Ind_Sigma_y_xx(i,j);
  			ELD(Sigma_y_xx,index2) =InvDXDT_J*(
  											ELD(Sigma_y_xx,index2)*DXDT_J+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff2);

			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_xx,index)= ELD(Sigma_x_xx,index2) + ELD(Sigma_y_xx,index2);
 #endif 			

 #if defined(_ST_PML_2)			
			index2=Ind_Sigma_x_yy(i,j);
  			ELD(Sigma_x_yy,index2) =InvDXDT_I*(
  											ELD(Sigma_x_yy,index2)*DXDT_I+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

			index2=Ind_Sigma_y_yy(i,j);
  			ELD(Sigma_y_yy,index2) =InvDXDT_J*(
  											ELD(Sigma_y_yy,index2)*DXDT_J+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff2);


			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_yy,index)= ELD(Sigma_x_yy,index2) + ELD(Sigma_y_yy,index2);

#endif

#if defined(_ST_PML_3)
  			index2=Ind_Sigma_x_xy(i,j);

  			Diff= i >0 && i<N1-2 ? CA*(EL(Vy,i+1,j)-EL(Vy,i,j)) -
  			                   CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j))
  			                    :i<N1-1 ? EL(Vy,i+1,j)-EL(Vy,i,j):0;

  			ELD(Sigma_x_xy,index2) =InvDXDThp_I*(
  											ELD(Sigma_x_xy,index2)*DXDThp_I+
  											RigidityXY*
  											Diff);
			index2=Ind_Sigma_y_xy(i,j);

  			Diff=j > 0 && j<N2-2 ? CA*(EL(Vx,i,j+1)-EL(Vx,i,j) )-
  			                       CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1) )
  			                       :j<N2-1 ? EL(Vx,i,j+1)-EL(Vx,i,j) :0;

  			ELD(Sigma_y_xy,index2) =InvDXDThp_J*(
  											ELD(Sigma_y_xy,index2)*DXDThp_J+
  											RigidityXY*
  											Diff);
			index=Ind_Sigma_xy(i,j);

			ELD(Sigma_xy,index)= ELD(Sigma_x_xy,Ind_Sigma_x_xy(i,j)) + ELD(Sigma_y_xy,index2);
#endif

		  }	   
#endif
	}
  	else
  	{
#if defined(_ST_MAIN_1)
  		//We are in the center, no need to check any limits, the use of the PML simplify this
  		index=Ind_Sigma_xx(i,j);

		if (REQUIRES_2ND_ORDER_M(X))
			Dx=EL(Vx,i,j)-EL(Vx,i-1,j);
		else
			Dx=CA*(EL(Vx,i,j)-EL(Vx,i-1,j))-
				CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j));

		if REQUIRES_2ND_ORDER_M(Y)
			Dy=EL(Vy,i,j)-EL(Vy,i,j-1);
		else
			Dy=CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
				CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2));

		//We will use the particle displacement to estimate the acoustic pressure, and using the conservation of mass formula
		//We can use the stress kernel as V matrices are not being modified in this kernel,
		// and the spatial derivatives are the same ones required for pressure calculation
        // partial(p)/partial(t) = \rho c^2 div(V)
        //it is important to mention that the Python function will need still to multiply the result for the maps of (speed of sound)^2 and density, 
		// and divide by the spatial step.
		EL(Pressure,i,j)+=DT*(Dx+Dy);
        accum_p+=EL(Pressure,i,j);


  		LambdaMiu=ELD(LambdaMiuMatOverH,MaterialID)*(1.0+ELD(TauLong,MaterialID));
  		Miu=2.0*ELD(MiuMatOverH,MaterialID)*(1.0+ELD(TauShear,MaterialID));
  		OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
		dFirstPart=LambdaMiu*(Dx+Dy);
		
		if (ELD(TauLong,MaterialID)!=0.0 || ELD(TauShear,MaterialID)!=0.0) // We avoid unnecessary calculations if there is no attenuation
		{
			
			LambdaMiuComp=DT*ELD(LambdaMiuMatOverH,MaterialID)*(ELD(TauLong,MaterialID)*OneOverTauSigma);
			dFirstPartForR=LambdaMiuComp*(Dx+Dy);
			MiuComp=DT*2.0*ELD(MiuMatOverH,MaterialID)*(ELD(TauShear,MaterialID)*OneOverTauSigma);
			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxx,index) - dFirstPartForR + MiuComp*(Dy))
  		    	  /(1+DT*0.5*OneOverTauSigma);

			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy) + 0.5*(ELD(Rxx,index) + NextR));
			ELD(Rxx,index)=NextR;
		}
		else
		{
			bAttenuating=0;
			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy));
		}
  		
	    accum_xx+=ELD(Sigma_xx,index);

		if (bAttenuating==1)
		{
  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryy,index) - dFirstPartForR + MiuComp*(Dx))
  		    	  /(1+DT*0.5*OneOverTauSigma);
				
  			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx) + 0.5*(ELD(Ryy,index) + NextR));
			ELD(Ryy,index)=NextR;
		}
		else
			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx));
      	
		accum_yy+=ELD(Sigma_yy,index);

  		

#endif		
#if defined(_ST_MAIN_2) ||  defined(_ST_MAIN_3) || defined(_ST_MAIN_4)
  		index=Ind_Sigma_xy(i,j);
#endif
#if defined(_ST_MAIN_2) 
		if (RigidityXY!=0.0)
  		{
			  OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
              if (REQUIRES_2ND_ORDER_P(X))
                  Dx=EL(Vy,i+1,j)-EL(Vy,i,j);
              else
                  Dx=CA*(EL(Vy,i+1,j)-EL(Vy,i,j))-
                     CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j));


              if (REQUIRES_2ND_ORDER_P(Y))
                  Dx+=EL(Vx,i,j+1)-EL(Vx,i,j);
              else
                  Dx+=CA*(EL(Vx,i,j+1)-EL(Vx,i,j))-
                      CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1));

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
        // else
        //     ELD(Rxy,index)=0.0;
#endif


	#if defined(_ST_MAIN_1)
		if ((nStep < LengthSource) && TypeSource>=2) //Source is stress
  		{
  			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
  			  source--; //need to use C index
  			  value=ELD(SourceFunctions,nStep*NumberSources+source); 
				index=Ind_Sigma_xx(i,j);
                if ((TypeSource-2)==0)
                {
                    ELD(Sigma_xx,index)+=value*ELD(Ox,index);
                    ELD(Sigma_yy,index)+=value*ELD(Oy,index);
                }
                else
                {
                   ELD(Sigma_xx,index)=value*ELD(Ox,index);
                   ELD(Sigma_yy,index)=value*ELD(Oy,index);
                }

  			}
  		}
	#endif
  	}
  }
  if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
  {
	#if defined(_ST_MAIN_1) 
    accum_xx/=ZoneCount;
    accum_yy/=ZoneCount;
	#endif
	#if defined(_ST_MAIN_2)
    accum_xy/=ZoneCount;
	#endif


    CurZone=0;
    index=IndN1N2(i,j,0);
    index2=N1*N2;


    if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)+=accum_xx*accum_xx;
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)+=accum_yy*accum_yy;
    
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;
		#endif
		#if defined(_ST_MAIN_2)
        if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)+=accum_xy*accum_xy;
		#endif

		
    }
    if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
        index+=index2*NumberSelRMSPeakMaps;
    if (SelRMSorPeak & SEL_PEAK)
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)=accum_xx>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx) ? accum_xx: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx);
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)=accum_yy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy) ? accum_yy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy);
        if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p :ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);
	    #endif
		#if defined(_ST_MAIN_2)
		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)=accum_xy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy) ? accum_xy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy);
        #endif
		
    }

  }
}
#undef _PML_KERNEL_CORNER

#define _PML_KERNEL_LEFT_RIGHT
#ifdef CUDA
extern "C" __global__ void PML_2_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void PML_2_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
kernel void PML_2_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
	_PT i = (_PT) gid.x;
	_PT j = (_PT) gid.y;
	#else
	#define nN1 (PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
  	_PT j = (_PT) ((gid )/nN1);
  	_PT i = (_PT) (gid -j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	// Each i,jgo from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1)
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML
#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size
#endif

#if defined(_MAIN_KERNEL) 
	i+=PML_Thickness;
	j+=PML_Thickness;
#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
	
#if defined(_ST_PML_1) || defined(_ST_PML_2)   
	mexType Diff;
#endif
#if defined(_ST_PML_1) || defined(_ST_PML_2)
	mexType Diff2;
#endif


#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2)  
	mexType Dx;
#endif
#if defined(_ST_MAIN_1)  
	mexType Dy;
#endif

#if defined(_ST_MAIN_1) || defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  
	mexType value;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)
	mexType m1;
	mexType m2;
	mexType m3;
	mexType m4;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2) 
	mexType RigidityXY=0.0;
#endif

#if defined(_ST_MAIN_2) 
	mexType TauShearXY=0.0;
#endif
#if defined(_ST_MAIN_1)
	mexType LambdaMiu;
	mexType LambdaMiuComp;
	mexType dFirstPart;
	mexType dFirstPartForR;
	mexType accum_xx=0.0;
	mexType accum_yy=0.0;
	mexType accum_p=0.0;
	_PT source;
	_PT bAttenuating=1;
#endif
#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2) 
	mexType Miu;
	mexType MiuComp;
	mexType OneOverTauSigma;
	mexType NextR;
#endif

#if defined(_ST_MAIN_2)
	mexType accum_xy=0.0;
#endif



#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
//#if defined(_ST_MAIN_1) || defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) ||  defined(_ST_PML_4) ||  defined(_ST_PML_5) ||  defined(_ST_PML_6)  
_PT index2;
//#endif
_PT index;
_PT  MaterialID;

_PT CurZone;

for ( CurZone=0;CurZone<ZoneCount;CurZone++)
  {
	  index=Ind_MaterialMap(i,j);
      MaterialID=ELD(MaterialMap,index);

	  #if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  

  		m1=ELD(MiuMatOverH,MaterialID);
  		m2=ELD(MiuMatOverH,EL(MaterialMap,i+1,j));
  		m3=ELD(MiuMatOverH,EL(MaterialMap,i,j+1));
  		m4=ELD(MiuMatOverH,EL(MaterialMap,i+1,j+1));
   		value=m1*m2*m3*m4;
  		RigidityXY =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
      #endif

	  #if  defined(_ST_MAIN_2) 
  		TauShearXY = value!=0.0 ? 0.25*(ELD(TauShear,MaterialID) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j)) +
  							 ELD(TauShear,EL(MaterialMap,i,j+1)) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j+1)))
  							 : ELD(TauShear,MaterialID);

	   #endif
	   
  	
  	if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )//We are in the PML borders
  	 {

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  		if (i<N1-1 && j <N2-1 )
  		{

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  			Diff= i>1 && i <N1-1 ? CA*(EL(Vx,i,j)-EL(Vx,i-1,j)) -
  			                       CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j))
  			      : i>0 && i <N1 ? (EL(Vx,i,j)-EL(Vx,i-1,j))  :0;

			Diff2= j>1 && j < N2-1 ? CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
  									CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2))
  			        : j>0 && j < N2 ? EL(Vy,i,j)-EL(Vy,i,j-1):0;

#endif

#if defined(_ST_PML_1)
  			
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_x_xx,index2) =InvDXDT_I*(
  											ELD(Sigma_x_xx,index2)*DXDT_I+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff);
			index2=Ind_Sigma_y_xx(i,j);
  			ELD(Sigma_y_xx,index2) =InvDXDT_J*(
  											ELD(Sigma_y_xx,index2)*DXDT_J+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff2);

			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_xx,index)= ELD(Sigma_x_xx,index2) + ELD(Sigma_y_xx,index2);
 #endif 			

 #if defined(_ST_PML_2)			
			index2=Ind_Sigma_x_yy(i,j);
  			ELD(Sigma_x_yy,index2) =InvDXDT_I*(
  											ELD(Sigma_x_yy,index2)*DXDT_I+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

			index2=Ind_Sigma_y_yy(i,j);
  			ELD(Sigma_y_yy,index2) =InvDXDT_J*(
  											ELD(Sigma_y_yy,index2)*DXDT_J+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff2);


			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_yy,index)= ELD(Sigma_x_yy,index2) + ELD(Sigma_y_yy,index2);

#endif

#if defined(_ST_PML_3)
  			index2=Ind_Sigma_x_xy(i,j);

  			Diff= i >0 && i<N1-2 ? CA*(EL(Vy,i+1,j)-EL(Vy,i,j)) -
  			                   CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j))
  			                    :i<N1-1 ? EL(Vy,i+1,j)-EL(Vy,i,j):0;

  			ELD(Sigma_x_xy,index2) =InvDXDThp_I*(
  											ELD(Sigma_x_xy,index2)*DXDThp_I+
  											RigidityXY*
  											Diff);
			index2=Ind_Sigma_y_xy(i,j);

  			Diff=j > 0 && j<N2-2 ? CA*(EL(Vx,i,j+1)-EL(Vx,i,j) )-
  			                       CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1) )
  			                       :j<N2-1 ? EL(Vx,i,j+1)-EL(Vx,i,j) :0;

  			ELD(Sigma_y_xy,index2) =InvDXDThp_J*(
  											ELD(Sigma_y_xy,index2)*DXDThp_J+
  											RigidityXY*
  											Diff);
			index=Ind_Sigma_xy(i,j);

			ELD(Sigma_xy,index)= ELD(Sigma_x_xy,Ind_Sigma_x_xy(i,j)) + ELD(Sigma_y_xy,index2);
#endif

		  }	   
#endif
	}
  	else
  	{
#if defined(_ST_MAIN_1)
  		//We are in the center, no need to check any limits, the use of the PML simplify this
  		index=Ind_Sigma_xx(i,j);

		if (REQUIRES_2ND_ORDER_M(X))
			Dx=EL(Vx,i,j)-EL(Vx,i-1,j);
		else
			Dx=CA*(EL(Vx,i,j)-EL(Vx,i-1,j))-
				CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j));

		if REQUIRES_2ND_ORDER_M(Y)
			Dy=EL(Vy,i,j)-EL(Vy,i,j-1);
		else
			Dy=CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
				CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2));

		//We will use the particle displacement to estimate the acoustic pressure, and using the conservation of mass formula
		//We can use the stress kernel as V matrices are not being modified in this kernel,
		// and the spatial derivatives are the same ones required for pressure calculation
        // partial(p)/partial(t) = \rho c^2 div(V)
        //it is important to mention that the Python function will need still to multiply the result for the maps of (speed of sound)^2 and density, 
		// and divide by the spatial step.
		EL(Pressure,i,j)+=DT*(Dx+Dy);
        accum_p+=EL(Pressure,i,j);


  		LambdaMiu=ELD(LambdaMiuMatOverH,MaterialID)*(1.0+ELD(TauLong,MaterialID));
  		Miu=2.0*ELD(MiuMatOverH,MaterialID)*(1.0+ELD(TauShear,MaterialID));
  		OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
		dFirstPart=LambdaMiu*(Dx+Dy);
		
		if (ELD(TauLong,MaterialID)!=0.0 || ELD(TauShear,MaterialID)!=0.0) // We avoid unnecessary calculations if there is no attenuation
		{
			
			LambdaMiuComp=DT*ELD(LambdaMiuMatOverH,MaterialID)*(ELD(TauLong,MaterialID)*OneOverTauSigma);
			dFirstPartForR=LambdaMiuComp*(Dx+Dy);
			MiuComp=DT*2.0*ELD(MiuMatOverH,MaterialID)*(ELD(TauShear,MaterialID)*OneOverTauSigma);
			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxx,index) - dFirstPartForR + MiuComp*(Dy))
  		    	  /(1+DT*0.5*OneOverTauSigma);

			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy) + 0.5*(ELD(Rxx,index) + NextR));
			ELD(Rxx,index)=NextR;
		}
		else
		{
			bAttenuating=0;
			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy));
		}
  		
	    accum_xx+=ELD(Sigma_xx,index);

		if (bAttenuating==1)
		{
  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryy,index) - dFirstPartForR + MiuComp*(Dx))
  		    	  /(1+DT*0.5*OneOverTauSigma);
				
  			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx) + 0.5*(ELD(Ryy,index) + NextR));
			ELD(Ryy,index)=NextR;
		}
		else
			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx));
      	
		accum_yy+=ELD(Sigma_yy,index);

  		

#endif		
#if defined(_ST_MAIN_2) ||  defined(_ST_MAIN_3) || defined(_ST_MAIN_4)
  		index=Ind_Sigma_xy(i,j);
#endif
#if defined(_ST_MAIN_2) 
		if (RigidityXY!=0.0)
  		{
			  OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
              if (REQUIRES_2ND_ORDER_P(X))
                  Dx=EL(Vy,i+1,j)-EL(Vy,i,j);
              else
                  Dx=CA*(EL(Vy,i+1,j)-EL(Vy,i,j))-
                     CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j));


              if (REQUIRES_2ND_ORDER_P(Y))
                  Dx+=EL(Vx,i,j+1)-EL(Vx,i,j);
              else
                  Dx+=CA*(EL(Vx,i,j+1)-EL(Vx,i,j))-
                      CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1));

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
        // else
        //     ELD(Rxy,index)=0.0;
#endif


	#if defined(_ST_MAIN_1)
		if ((nStep < LengthSource) && TypeSource>=2) //Source is stress
  		{
  			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
  			  source--; //need to use C index
  			  value=ELD(SourceFunctions,nStep*NumberSources+source); 
				index=Ind_Sigma_xx(i,j);
                if ((TypeSource-2)==0)
                {
                    ELD(Sigma_xx,index)+=value*ELD(Ox,index);
                    ELD(Sigma_yy,index)+=value*ELD(Oy,index);
                }
                else
                {
                   ELD(Sigma_xx,index)=value*ELD(Ox,index);
                   ELD(Sigma_yy,index)=value*ELD(Oy,index);
                }

  			}
  		}
	#endif
  	}
  }
  if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
  {
	#if defined(_ST_MAIN_1) 
    accum_xx/=ZoneCount;
    accum_yy/=ZoneCount;
	#endif
	#if defined(_ST_MAIN_2)
    accum_xy/=ZoneCount;
	#endif


    CurZone=0;
    index=IndN1N2(i,j,0);
    index2=N1*N2;


    if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)+=accum_xx*accum_xx;
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)+=accum_yy*accum_yy;
    
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;
		#endif
		#if defined(_ST_MAIN_2)
        if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)+=accum_xy*accum_xy;
		#endif

		
    }
    if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
        index+=index2*NumberSelRMSPeakMaps;
    if (SelRMSorPeak & SEL_PEAK)
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)=accum_xx>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx) ? accum_xx: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx);
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)=accum_yy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy) ? accum_yy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy);
        if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p :ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);
	    #endif
		#if defined(_ST_MAIN_2)
		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)=accum_xy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy) ? accum_xy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy);
        #endif
		
    }

  }
}
#undef _PML_KERNEL_LEFT_RIGHT

#define _PML_KERNEL_TOP_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_3_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void PML_3_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
kernel void PML_3_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
	_PT i = (_PT) gid.x;
	_PT j = (_PT) gid.y;
	#else
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (PML_Thickness*2)
  	_PT j = (_PT) ((gid)/nN1);
  	_PT i = (_PT) (gid -j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	// Each i,jgo from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1)
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML
#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size
#endif

#if defined(_MAIN_KERNEL) 
	i+=PML_Thickness;
	j+=PML_Thickness;
#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
	
#if defined(_ST_PML_1) || defined(_ST_PML_2)   
	mexType Diff;
#endif
#if defined(_ST_PML_1) || defined(_ST_PML_2)
	mexType Diff2;
#endif


#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2)  
	mexType Dx;
#endif
#if defined(_ST_MAIN_1)  
	mexType Dy;
#endif

#if defined(_ST_MAIN_1) || defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  
	mexType value;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)
	mexType m1;
	mexType m2;
	mexType m3;
	mexType m4;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2) 
	mexType RigidityXY=0.0;
#endif

#if defined(_ST_MAIN_2) 
	mexType TauShearXY=0.0;
#endif
#if defined(_ST_MAIN_1)
	mexType LambdaMiu;
	mexType LambdaMiuComp;
	mexType dFirstPart;
	mexType dFirstPartForR;
	mexType accum_xx=0.0;
	mexType accum_yy=0.0;
	mexType accum_p=0.0;
	_PT source;
	_PT bAttenuating=1;
#endif
#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2) 
	mexType Miu;
	mexType MiuComp;
	mexType OneOverTauSigma;
	mexType NextR;
#endif

#if defined(_ST_MAIN_2)
	mexType accum_xy=0.0;
#endif



#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
//#if defined(_ST_MAIN_1) || defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) ||  defined(_ST_PML_4) ||  defined(_ST_PML_5) ||  defined(_ST_PML_6)  
_PT index2;
//#endif
_PT index;
_PT  MaterialID;

_PT CurZone;

for ( CurZone=0;CurZone<ZoneCount;CurZone++)
  {
	  index=Ind_MaterialMap(i,j);
      MaterialID=ELD(MaterialMap,index);

	  #if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  

  		m1=ELD(MiuMatOverH,MaterialID);
  		m2=ELD(MiuMatOverH,EL(MaterialMap,i+1,j));
  		m3=ELD(MiuMatOverH,EL(MaterialMap,i,j+1));
  		m4=ELD(MiuMatOverH,EL(MaterialMap,i+1,j+1));
   		value=m1*m2*m3*m4;
  		RigidityXY =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
      #endif

	  #if  defined(_ST_MAIN_2) 
  		TauShearXY = value!=0.0 ? 0.25*(ELD(TauShear,MaterialID) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j)) +
  							 ELD(TauShear,EL(MaterialMap,i,j+1)) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j+1)))
  							 : ELD(TauShear,MaterialID);

	   #endif
	   
  	
  	if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )//We are in the PML borders
  	 {

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  		if (i<N1-1 && j <N2-1 )
  		{

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  			Diff= i>1 && i <N1-1 ? CA*(EL(Vx,i,j)-EL(Vx,i-1,j)) -
  			                       CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j))
  			      : i>0 && i <N1 ? (EL(Vx,i,j)-EL(Vx,i-1,j))  :0;

			Diff2= j>1 && j < N2-1 ? CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
  									CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2))
  			        : j>0 && j < N2 ? EL(Vy,i,j)-EL(Vy,i,j-1):0;

#endif

#if defined(_ST_PML_1)
  			
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_x_xx,index2) =InvDXDT_I*(
  											ELD(Sigma_x_xx,index2)*DXDT_I+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff);
			index2=Ind_Sigma_y_xx(i,j);
  			ELD(Sigma_y_xx,index2) =InvDXDT_J*(
  											ELD(Sigma_y_xx,index2)*DXDT_J+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff2);

			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_xx,index)= ELD(Sigma_x_xx,index2) + ELD(Sigma_y_xx,index2);
 #endif 			

 #if defined(_ST_PML_2)			
			index2=Ind_Sigma_x_yy(i,j);
  			ELD(Sigma_x_yy,index2) =InvDXDT_I*(
  											ELD(Sigma_x_yy,index2)*DXDT_I+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

			index2=Ind_Sigma_y_yy(i,j);
  			ELD(Sigma_y_yy,index2) =InvDXDT_J*(
  											ELD(Sigma_y_yy,index2)*DXDT_J+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff2);


			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_yy,index)= ELD(Sigma_x_yy,index2) + ELD(Sigma_y_yy,index2);

#endif

#if defined(_ST_PML_3)
  			index2=Ind_Sigma_x_xy(i,j);

  			Diff= i >0 && i<N1-2 ? CA*(EL(Vy,i+1,j)-EL(Vy,i,j)) -
  			                   CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j))
  			                    :i<N1-1 ? EL(Vy,i+1,j)-EL(Vy,i,j):0;

  			ELD(Sigma_x_xy,index2) =InvDXDThp_I*(
  											ELD(Sigma_x_xy,index2)*DXDThp_I+
  											RigidityXY*
  											Diff);
			index2=Ind_Sigma_y_xy(i,j);

  			Diff=j > 0 && j<N2-2 ? CA*(EL(Vx,i,j+1)-EL(Vx,i,j) )-
  			                       CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1) )
  			                       :j<N2-1 ? EL(Vx,i,j+1)-EL(Vx,i,j) :0;

  			ELD(Sigma_y_xy,index2) =InvDXDThp_J*(
  											ELD(Sigma_y_xy,index2)*DXDThp_J+
  											RigidityXY*
  											Diff);
			index=Ind_Sigma_xy(i,j);

			ELD(Sigma_xy,index)= ELD(Sigma_x_xy,Ind_Sigma_x_xy(i,j)) + ELD(Sigma_y_xy,index2);
#endif

		  }	   
#endif
	}
  	else
  	{
#if defined(_ST_MAIN_1)
  		//We are in the center, no need to check any limits, the use of the PML simplify this
  		index=Ind_Sigma_xx(i,j);

		if (REQUIRES_2ND_ORDER_M(X))
			Dx=EL(Vx,i,j)-EL(Vx,i-1,j);
		else
			Dx=CA*(EL(Vx,i,j)-EL(Vx,i-1,j))-
				CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j));

		if REQUIRES_2ND_ORDER_M(Y)
			Dy=EL(Vy,i,j)-EL(Vy,i,j-1);
		else
			Dy=CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
				CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2));

		//We will use the particle displacement to estimate the acoustic pressure, and using the conservation of mass formula
		//We can use the stress kernel as V matrices are not being modified in this kernel,
		// and the spatial derivatives are the same ones required for pressure calculation
        // partial(p)/partial(t) = \rho c^2 div(V)
        //it is important to mention that the Python function will need still to multiply the result for the maps of (speed of sound)^2 and density, 
		// and divide by the spatial step.
		EL(Pressure,i,j)+=DT*(Dx+Dy);
        accum_p+=EL(Pressure,i,j);


  		LambdaMiu=ELD(LambdaMiuMatOverH,MaterialID)*(1.0+ELD(TauLong,MaterialID));
  		Miu=2.0*ELD(MiuMatOverH,MaterialID)*(1.0+ELD(TauShear,MaterialID));
  		OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
		dFirstPart=LambdaMiu*(Dx+Dy);
		
		if (ELD(TauLong,MaterialID)!=0.0 || ELD(TauShear,MaterialID)!=0.0) // We avoid unnecessary calculations if there is no attenuation
		{
			
			LambdaMiuComp=DT*ELD(LambdaMiuMatOverH,MaterialID)*(ELD(TauLong,MaterialID)*OneOverTauSigma);
			dFirstPartForR=LambdaMiuComp*(Dx+Dy);
			MiuComp=DT*2.0*ELD(MiuMatOverH,MaterialID)*(ELD(TauShear,MaterialID)*OneOverTauSigma);
			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxx,index) - dFirstPartForR + MiuComp*(Dy))
  		    	  /(1+DT*0.5*OneOverTauSigma);

			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy) + 0.5*(ELD(Rxx,index) + NextR));
			ELD(Rxx,index)=NextR;
		}
		else
		{
			bAttenuating=0;
			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy));
		}
  		
	    accum_xx+=ELD(Sigma_xx,index);

		if (bAttenuating==1)
		{
  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryy,index) - dFirstPartForR + MiuComp*(Dx))
  		    	  /(1+DT*0.5*OneOverTauSigma);
				
  			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx) + 0.5*(ELD(Ryy,index) + NextR));
			ELD(Ryy,index)=NextR;
		}
		else
			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx));
      	
		accum_yy+=ELD(Sigma_yy,index);

  		

#endif		
#if defined(_ST_MAIN_2) ||  defined(_ST_MAIN_3) || defined(_ST_MAIN_4)
  		index=Ind_Sigma_xy(i,j);
#endif
#if defined(_ST_MAIN_2) 
		if (RigidityXY!=0.0)
  		{
			  OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
              if (REQUIRES_2ND_ORDER_P(X))
                  Dx=EL(Vy,i+1,j)-EL(Vy,i,j);
              else
                  Dx=CA*(EL(Vy,i+1,j)-EL(Vy,i,j))-
                     CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j));


              if (REQUIRES_2ND_ORDER_P(Y))
                  Dx+=EL(Vx,i,j+1)-EL(Vx,i,j);
              else
                  Dx+=CA*(EL(Vx,i,j+1)-EL(Vx,i,j))-
                      CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1));

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
        // else
        //     ELD(Rxy,index)=0.0;
#endif


	#if defined(_ST_MAIN_1)
		if ((nStep < LengthSource) && TypeSource>=2) //Source is stress
  		{
  			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
  			  source--; //need to use C index
  			  value=ELD(SourceFunctions,nStep*NumberSources+source); 
				index=Ind_Sigma_xx(i,j);
                if ((TypeSource-2)==0)
                {
                    ELD(Sigma_xx,index)+=value*ELD(Ox,index);
                    ELD(Sigma_yy,index)+=value*ELD(Oy,index);
                }
                else
                {
                   ELD(Sigma_xx,index)=value*ELD(Ox,index);
                   ELD(Sigma_yy,index)=value*ELD(Oy,index);
                }

  			}
  		}
	#endif
  	}
  }
  if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
  {
	#if defined(_ST_MAIN_1) 
    accum_xx/=ZoneCount;
    accum_yy/=ZoneCount;
	#endif
	#if defined(_ST_MAIN_2)
    accum_xy/=ZoneCount;
	#endif


    CurZone=0;
    index=IndN1N2(i,j,0);
    index2=N1*N2;


    if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)+=accum_xx*accum_xx;
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)+=accum_yy*accum_yy;
    
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;
		#endif
		#if defined(_ST_MAIN_2)
        if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)+=accum_xy*accum_xy;
		#endif

		
    }
    if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
        index+=index2*NumberSelRMSPeakMaps;
    if (SelRMSorPeak & SEL_PEAK)
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)=accum_xx>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx) ? accum_xx: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx);
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)=accum_yy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy) ? accum_yy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy);
        if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p :ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);
	    #endif
		#if defined(_ST_MAIN_2)
		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)=accum_xy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy) ? accum_xy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy);
        #endif
		
    }

  }
}
#undef _PML_KERNEL_TOP_BOTTOM


#undef _ST_PML_1
#undef _ST_PML_2
#undef _ST_PML_3
#endif

#define _ST_MAIN_1
#define _ST_MAIN_2
#define _ST_MAIN_3
#define _ST_MAIN_4
#define _MAIN_KERNEL
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA))
#define _ST_PML_1
#define _ST_PML_2
#define _ST_PML_3

#endif
#ifdef CUDA
extern "C" __global__ void MAIN_1_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void MAIN_1_StressKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
kernel void MAIN_1_StressKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
	#else	
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
  	_PT j = (_PT) ((gid )/nN1);
  	_PT i = (_PT) (gid -j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	// Each i,jgo from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1)
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML
#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size
#endif

#if defined(_MAIN_KERNEL) 
	i+=PML_Thickness;
	j+=PML_Thickness;
#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
	
#if defined(_ST_PML_1) || defined(_ST_PML_2)   
	mexType Diff;
#endif
#if defined(_ST_PML_1) || defined(_ST_PML_2)
	mexType Diff2;
#endif


#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2)  
	mexType Dx;
#endif
#if defined(_ST_MAIN_1)  
	mexType Dy;
#endif

#if defined(_ST_MAIN_1) || defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  
	mexType value;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)
	mexType m1;
	mexType m2;
	mexType m3;
	mexType m4;
#endif
#if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2) 
	mexType RigidityXY=0.0;
#endif

#if defined(_ST_MAIN_2) 
	mexType TauShearXY=0.0;
#endif
#if defined(_ST_MAIN_1)
	mexType LambdaMiu;
	mexType LambdaMiuComp;
	mexType dFirstPart;
	mexType dFirstPartForR;
	mexType accum_xx=0.0;
	mexType accum_yy=0.0;
	mexType accum_p=0.0;
	_PT source;
	_PT bAttenuating=1;
#endif
#if defined(_ST_MAIN_1) || defined(_ST_MAIN_2) 
	mexType Miu;
	mexType MiuComp;
	mexType OneOverTauSigma;
	mexType NextR;
#endif

#if defined(_ST_MAIN_2)
	mexType accum_xy=0.0;
#endif



#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
//#if defined(_ST_MAIN_1) || defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) ||  defined(_ST_PML_4) ||  defined(_ST_PML_5) ||  defined(_ST_PML_6)  
_PT index2;
//#endif
_PT index;
_PT  MaterialID;

_PT CurZone;

for ( CurZone=0;CurZone<ZoneCount;CurZone++)
  {
	  index=Ind_MaterialMap(i,j);
      MaterialID=ELD(MaterialMap,index);

	  #if  defined(_ST_PML_3) ||  defined(_ST_MAIN_2)  

  		m1=ELD(MiuMatOverH,MaterialID);
  		m2=ELD(MiuMatOverH,EL(MaterialMap,i+1,j));
  		m3=ELD(MiuMatOverH,EL(MaterialMap,i,j+1));
  		m4=ELD(MiuMatOverH,EL(MaterialMap,i+1,j+1));
   		value=m1*m2*m3*m4;
  		RigidityXY =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
      #endif

	  #if  defined(_ST_MAIN_2) 
  		TauShearXY = value!=0.0 ? 0.25*(ELD(TauShear,MaterialID) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j)) +
  							 ELD(TauShear,EL(MaterialMap,i,j+1)) +
  							 ELD(TauShear,EL(MaterialMap,i+1,j+1)))
  							 : ELD(TauShear,MaterialID);

	   #endif
	   
  	
  	if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )//We are in the PML borders
  	 {

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  		if (i<N1-1 && j <N2-1 )
  		{

#if defined(_ST_PML_1) || defined(_ST_PML_2) ||  defined(_ST_PML_3) 
  			Diff= i>1 && i <N1-1 ? CA*(EL(Vx,i,j)-EL(Vx,i-1,j)) -
  			                       CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j))
  			      : i>0 && i <N1 ? (EL(Vx,i,j)-EL(Vx,i-1,j))  :0;

			Diff2= j>1 && j < N2-1 ? CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
  									CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2))
  			        : j>0 && j < N2 ? EL(Vy,i,j)-EL(Vy,i,j-1):0;

#endif

#if defined(_ST_PML_1)
  			
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_x_xx,index2) =InvDXDT_I*(
  											ELD(Sigma_x_xx,index2)*DXDT_I+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff);
			index2=Ind_Sigma_y_xx(i,j);
  			ELD(Sigma_y_xx,index2) =InvDXDT_J*(
  											ELD(Sigma_y_xx,index2)*DXDT_J+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff2);

			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_xx,index)= ELD(Sigma_x_xx,index2) + ELD(Sigma_y_xx,index2);
 #endif 			

 #if defined(_ST_PML_2)			
			index2=Ind_Sigma_x_yy(i,j);
  			ELD(Sigma_x_yy,index2) =InvDXDT_I*(
  											ELD(Sigma_x_yy,index2)*DXDT_I+
  											ELD(LambdaMatOverH,MaterialID)*
  											Diff);

			index2=Ind_Sigma_y_yy(i,j);
  			ELD(Sigma_y_yy,index2) =InvDXDT_J*(
  											ELD(Sigma_y_yy,index2)*DXDT_J+
  											ELD(LambdaMiuMatOverH,MaterialID)*
  											Diff2);


			index=Ind_Sigma_xx(i,j);
  			index2=Ind_Sigma_x_xx(i,j);
  			ELD(Sigma_yy,index)= ELD(Sigma_x_yy,index2) + ELD(Sigma_y_yy,index2);

#endif

#if defined(_ST_PML_3)
  			index2=Ind_Sigma_x_xy(i,j);

  			Diff= i >0 && i<N1-2 ? CA*(EL(Vy,i+1,j)-EL(Vy,i,j)) -
  			                   CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j))
  			                    :i<N1-1 ? EL(Vy,i+1,j)-EL(Vy,i,j):0;

  			ELD(Sigma_x_xy,index2) =InvDXDThp_I*(
  											ELD(Sigma_x_xy,index2)*DXDThp_I+
  											RigidityXY*
  											Diff);
			index2=Ind_Sigma_y_xy(i,j);

  			Diff=j > 0 && j<N2-2 ? CA*(EL(Vx,i,j+1)-EL(Vx,i,j) )-
  			                       CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1) )
  			                       :j<N2-1 ? EL(Vx,i,j+1)-EL(Vx,i,j) :0;

  			ELD(Sigma_y_xy,index2) =InvDXDThp_J*(
  											ELD(Sigma_y_xy,index2)*DXDThp_J+
  											RigidityXY*
  											Diff);
			index=Ind_Sigma_xy(i,j);

			ELD(Sigma_xy,index)= ELD(Sigma_x_xy,Ind_Sigma_x_xy(i,j)) + ELD(Sigma_y_xy,index2);
#endif

		  }	   
#endif
	}
  	else
  	{
#if defined(_ST_MAIN_1)
  		//We are in the center, no need to check any limits, the use of the PML simplify this
  		index=Ind_Sigma_xx(i,j);

		if (REQUIRES_2ND_ORDER_M(X))
			Dx=EL(Vx,i,j)-EL(Vx,i-1,j);
		else
			Dx=CA*(EL(Vx,i,j)-EL(Vx,i-1,j))-
				CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j));

		if REQUIRES_2ND_ORDER_M(Y)
			Dy=EL(Vy,i,j)-EL(Vy,i,j-1);
		else
			Dy=CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
				CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2));

		//We will use the particle displacement to estimate the acoustic pressure, and using the conservation of mass formula
		//We can use the stress kernel as V matrices are not being modified in this kernel,
		// and the spatial derivatives are the same ones required for pressure calculation
        // partial(p)/partial(t) = \rho c^2 div(V)
        //it is important to mention that the Python function will need still to multiply the result for the maps of (speed of sound)^2 and density, 
		// and divide by the spatial step.
		EL(Pressure,i,j)+=DT*(Dx+Dy);
        accum_p+=EL(Pressure,i,j);


  		LambdaMiu=ELD(LambdaMiuMatOverH,MaterialID)*(1.0+ELD(TauLong,MaterialID));
  		Miu=2.0*ELD(MiuMatOverH,MaterialID)*(1.0+ELD(TauShear,MaterialID));
  		OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
		dFirstPart=LambdaMiu*(Dx+Dy);
		
		if (ELD(TauLong,MaterialID)!=0.0 || ELD(TauShear,MaterialID)!=0.0) // We avoid unnecessary calculations if there is no attenuation
		{
			
			LambdaMiuComp=DT*ELD(LambdaMiuMatOverH,MaterialID)*(ELD(TauLong,MaterialID)*OneOverTauSigma);
			dFirstPartForR=LambdaMiuComp*(Dx+Dy);
			MiuComp=DT*2.0*ELD(MiuMatOverH,MaterialID)*(ELD(TauShear,MaterialID)*OneOverTauSigma);
			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxx,index) - dFirstPartForR + MiuComp*(Dy))
  		    	  /(1+DT*0.5*OneOverTauSigma);

			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy) + 0.5*(ELD(Rxx,index) + NextR));
			ELD(Rxx,index)=NextR;
		}
		else
		{
			bAttenuating=0;
			ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy));
		}
  		
	    accum_xx+=ELD(Sigma_xx,index);

		if (bAttenuating==1)
		{
  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryy,index) - dFirstPartForR + MiuComp*(Dx))
  		    	  /(1+DT*0.5*OneOverTauSigma);
				
  			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx) + 0.5*(ELD(Ryy,index) + NextR));
			ELD(Ryy,index)=NextR;
		}
		else
			ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx));
      	
		accum_yy+=ELD(Sigma_yy,index);

  		

#endif		
#if defined(_ST_MAIN_2) ||  defined(_ST_MAIN_3) || defined(_ST_MAIN_4)
  		index=Ind_Sigma_xy(i,j);
#endif
#if defined(_ST_MAIN_2) 
		if (RigidityXY!=0.0)
  		{
			  OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
              if (REQUIRES_2ND_ORDER_P(X))
                  Dx=EL(Vy,i+1,j)-EL(Vy,i,j);
              else
                  Dx=CA*(EL(Vy,i+1,j)-EL(Vy,i,j))-
                     CB*(EL(Vy,i+2,j)-EL(Vy,i-1,j));


              if (REQUIRES_2ND_ORDER_P(Y))
                  Dx+=EL(Vx,i,j+1)-EL(Vx,i,j);
              else
                  Dx+=CA*(EL(Vx,i,j+1)-EL(Vx,i,j))-
                      CB*(EL(Vx,i,j+2)-EL(Vx,i,j-1));

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
        // else
        //     ELD(Rxy,index)=0.0;
#endif


	#if defined(_ST_MAIN_1)
		if ((nStep < LengthSource) && TypeSource>=2) //Source is stress
  		{
  			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
  			  source--; //need to use C index
  			  value=ELD(SourceFunctions,nStep*NumberSources+source); 
				index=Ind_Sigma_xx(i,j);
                if ((TypeSource-2)==0)
                {
                    ELD(Sigma_xx,index)+=value*ELD(Ox,index);
                    ELD(Sigma_yy,index)+=value*ELD(Oy,index);
                }
                else
                {
                   ELD(Sigma_xx,index)=value*ELD(Ox,index);
                   ELD(Sigma_yy,index)=value*ELD(Oy,index);
                }

  			}
  		}
	#endif
  	}
  }
  if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
  {
	#if defined(_ST_MAIN_1) 
    accum_xx/=ZoneCount;
    accum_yy/=ZoneCount;
	#endif
	#if defined(_ST_MAIN_2)
    accum_xy/=ZoneCount;
	#endif


    CurZone=0;
    index=IndN1N2(i,j,0);
    index2=N1*N2;


    if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)+=accum_xx*accum_xx;
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)+=accum_yy*accum_yy;
    
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;
		#endif
		#if defined(_ST_MAIN_2)
        if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)+=accum_xy*accum_xy;
		#endif

		
    }
    if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
        index+=index2*NumberSelRMSPeakMaps;
    if (SelRMSorPeak & SEL_PEAK)
    {
		#if defined(_ST_MAIN_1)
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)=accum_xx>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx) ? accum_xx: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx);
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)=accum_yy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy) ? accum_yy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy);
        if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p :ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);
	    #endif
		#if defined(_ST_MAIN_2)
		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)=accum_xy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy) ? accum_xy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy);
        #endif
		
    }

  }
}
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA))
#undef _ST_PML_1
#undef _ST_PML_2
#undef _ST_PML_3
#endif
#undef _MAIN_KERNEL
#undef _ST_MAIN_1
#undef _ST_MAIN_2



// PML
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#define _PR_PML_1
#define _PR_PML_2
#define _PML_KERNEL_CORNER
#ifdef CUDA
extern "C" __global__ void PML_1_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void PML_1_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
kernel void PML_1_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
	#else	
	#define nN1 (PML_Thickness*2)
	#define nN2 (PML_Thickness*2)
  	_PT j = (_PT) ((gid )/nN1);
  	_PT i = (_PT) (gid -j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	
	// Each i,j go from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1 )
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML

#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size

#endif


#if defined(_MAIN_KERNEL)
i+=PML_Thickness;
j+=PML_Thickness;

#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
	_PT source;
	mexType value;
#endif
#if defined(_PR_PML_1) || defined(_PR_PML_2)  || defined(_PR_MAIN_1) 
	mexType AvgInvRhoI;
#endif

#if defined(_PR_PML_1) || defined(_PR_PML_2)
	mexType Diff;
#endif
#if defined(_PR_MAIN_1) 
	mexType accum_x=0.0;
	mexType Dx;
#endif
#if defined(_PR_MAIN_2)
	mexType accum_y=0.0;
	mexType AvgInvRhoJ;
	mexType Dy;
#endif

_PT index;
_PT index2;
_PT  CurZone;
	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		{
		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )
			{
	#if defined(_PR_PML_1) || defined(_PR_PML_2)  
				index=Ind_MaterialMap(i,j);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 )
				{
    #if defined(_PR_PML_1)
					index=Ind_V_x_x(i,j);


		            Diff= i>0 && i<N1-2 ? CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
		                                  CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j))
					                      :i<N1-1 ? EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j):0;

					ELD(V_x_x,index) =InvDXDThp_I*(ELD(V_x_x,index)*DXDThp_I+
													AvgInvRhoI*
													Diff);

					index=Ind_V_y_x(i,j);
					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
					                      CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2))
					                      :j>0 && j<N2 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1):0;

					ELD(V_y_x,index) =InvDXDT_J*(
													ELD(V_y_x,index)*DXDT_J+
													AvgInvRhoI*
													Diff);

					
					index=Ind_V_x(i,j);
					index2=Ind_V_x_x(i,j);
					ELD(Vx,index)=ELD(V_x_x,index2)+ELD(V_y_x,index2);
		#endif
		#if defined(_PR_PML_2)			

				// For coeffs. for V_y

					index=Ind_V_x_y(i,j);

					Diff= i>1 && i<N1-1 ? CA *(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j)) -
					                      CB *(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j))
					                      :i>0 && i<N1 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j):0;

					ELD(V_x_y,index) =InvDXDT_I*(
													ELD(V_x_y,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_y(i,j);
					Diff= j>0 && j < N2-2 ? CA*( EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j)) -
					                        CB*( EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1))
					                        :j < N2-1 ? EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j):0;

					ELD(V_y_y,index) =InvDXDThp_J*(
												ELD(V_y_y,index)*DXDThp_J+
												AvgInvRhoI*
												Diff);

					index=Ind_V_y(i,j);
					index2=Ind_V_y_y(i,j);
					ELD(Vy,index)=ELD(V_x_y,index2)+ELD(V_y_y,index2);

		#endif
		
				 }
	#endif	
			}
			else
			{
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
				index=Ind_MaterialMap(i,j);
	#if defined(_PR_MAIN_1)
				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dx=CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
						CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j));

				Dx+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
						CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2));

				EL(Vx,i,j)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j);
	#endif
	#if defined(_PR_MAIN_2)
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dy=CA*(EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j) )-
						CB*(EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1));

				Dy+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j))-
						CB*(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j));
				
				EL(Vy,i,j)+=DT*AvgInvRhoJ*Dy;
				accum_y+=EL(Vy,i,j);
	#endif
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#endif
		}
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
  		if ((nStep < LengthSource) && TypeSource<2) //Source is particle displacement
  		{
			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
				source--; //need to use C index
  			  	value=ELD(SourceFunctions,nStep*NumberSources+source);
				if (TypeSource==0)
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)+=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)+=value*ELD(Oy,index);
					#endif

					
				}
				else
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)=value*ELD(Oy,index);
					#endif
				}

  			}
  		}
	#endif
		}
		#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
		if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
	    {
			if (ZoneCount>1)
			{
				#if defined(_PR_MAIN_1)
				accum_x/=ZoneCount;
				#endif
				#if defined(_PR_MAIN_2)
				accum_y/=ZoneCount;
				#endif

			}
			CurZone=0;
			index=IndN1N2(i,j,0);
			index2=N1*N2;
			if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
				#endif


			}
			if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
					index+=index2*NumberSelRMSPeakMaps;
			if (SelRMSorPeak & SEL_PEAK)
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
				#endif
				
			}


		}
		#endif
		
}
#undef _PML_KERNEL_CORNER

#define _PML_KERNEL_LEFT_RIGHT
#ifdef CUDA
extern "C" __global__ void PML_2_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void PML_2_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
kernel void PML_2_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
	#else	
	#define nN1 (PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
  	_PT j = (_PT) ((gid )/nN1);
  	_PT i = (_PT) (gid -j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	
	// Each i,j go from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1 )
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML

#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size

#endif


#if defined(_MAIN_KERNEL)
i+=PML_Thickness;
j+=PML_Thickness;

#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
	_PT source;
	mexType value;
#endif
#if defined(_PR_PML_1) || defined(_PR_PML_2)  || defined(_PR_MAIN_1) 
	mexType AvgInvRhoI;
#endif

#if defined(_PR_PML_1) || defined(_PR_PML_2)
	mexType Diff;
#endif
#if defined(_PR_MAIN_1) 
	mexType accum_x=0.0;
	mexType Dx;
#endif
#if defined(_PR_MAIN_2)
	mexType accum_y=0.0;
	mexType AvgInvRhoJ;
	mexType Dy;
#endif

_PT index;
_PT index2;
_PT  CurZone;
	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		{
		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )
			{
	#if defined(_PR_PML_1) || defined(_PR_PML_2)  
				index=Ind_MaterialMap(i,j);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 )
				{
    #if defined(_PR_PML_1)
					index=Ind_V_x_x(i,j);


		            Diff= i>0 && i<N1-2 ? CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
		                                  CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j))
					                      :i<N1-1 ? EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j):0;

					ELD(V_x_x,index) =InvDXDThp_I*(ELD(V_x_x,index)*DXDThp_I+
													AvgInvRhoI*
													Diff);

					index=Ind_V_y_x(i,j);
					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
					                      CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2))
					                      :j>0 && j<N2 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1):0;

					ELD(V_y_x,index) =InvDXDT_J*(
													ELD(V_y_x,index)*DXDT_J+
													AvgInvRhoI*
													Diff);

					
					index=Ind_V_x(i,j);
					index2=Ind_V_x_x(i,j);
					ELD(Vx,index)=ELD(V_x_x,index2)+ELD(V_y_x,index2);
		#endif
		#if defined(_PR_PML_2)			

				// For coeffs. for V_y

					index=Ind_V_x_y(i,j);

					Diff= i>1 && i<N1-1 ? CA *(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j)) -
					                      CB *(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j))
					                      :i>0 && i<N1 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j):0;

					ELD(V_x_y,index) =InvDXDT_I*(
													ELD(V_x_y,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_y(i,j);
					Diff= j>0 && j < N2-2 ? CA*( EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j)) -
					                        CB*( EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1))
					                        :j < N2-1 ? EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j):0;

					ELD(V_y_y,index) =InvDXDThp_J*(
												ELD(V_y_y,index)*DXDThp_J+
												AvgInvRhoI*
												Diff);

					index=Ind_V_y(i,j);
					index2=Ind_V_y_y(i,j);
					ELD(Vy,index)=ELD(V_x_y,index2)+ELD(V_y_y,index2);

		#endif
		
				 }
	#endif	
			}
			else
			{
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
				index=Ind_MaterialMap(i,j);
	#if defined(_PR_MAIN_1)
				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dx=CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
						CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j));

				Dx+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
						CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2));

				EL(Vx,i,j)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j);
	#endif
	#if defined(_PR_MAIN_2)
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dy=CA*(EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j) )-
						CB*(EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1));

				Dy+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j))-
						CB*(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j));
				
				EL(Vy,i,j)+=DT*AvgInvRhoJ*Dy;
				accum_y+=EL(Vy,i,j);
	#endif
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#endif
		}
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
  		if ((nStep < LengthSource) && TypeSource<2) //Source is particle displacement
  		{
			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
				source--; //need to use C index
  			  	value=ELD(SourceFunctions,nStep*NumberSources+source);
				if (TypeSource==0)
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)+=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)+=value*ELD(Oy,index);
					#endif

					
				}
				else
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)=value*ELD(Oy,index);
					#endif
				}

  			}
  		}
	#endif
		}
		#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
		if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
	    {
			if (ZoneCount>1)
			{
				#if defined(_PR_MAIN_1)
				accum_x/=ZoneCount;
				#endif
				#if defined(_PR_MAIN_2)
				accum_y/=ZoneCount;
				#endif

			}
			CurZone=0;
			index=IndN1N2(i,j,0);
			index2=N1*N2;
			if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
				#endif


			}
			if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
					index+=index2*NumberSelRMSPeakMaps;
			if (SelRMSorPeak & SEL_PEAK)
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
				#endif
				
			}


		}
		#endif
		
}
#undef _PML_KERNEL_LEFT_RIGHT

#define _PML_KERNEL_TOP_BOTTOM
#ifdef CUDA
extern "C" __global__ void PML_3_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,unsigned int nStep, unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void PML_3_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep, unsigned int TypeSource)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
kernel void PML_3_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
	#else	
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (PML_Thickness*2)
  	_PT j = (_PT) ((gid )/nN1);
  	_PT i = (_PT) (gid -j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	
	// Each i,j go from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1 )
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML

#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size

#endif


#if defined(_MAIN_KERNEL)
i+=PML_Thickness;
j+=PML_Thickness;

#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
	_PT source;
	mexType value;
#endif
#if defined(_PR_PML_1) || defined(_PR_PML_2)  || defined(_PR_MAIN_1) 
	mexType AvgInvRhoI;
#endif

#if defined(_PR_PML_1) || defined(_PR_PML_2)
	mexType Diff;
#endif
#if defined(_PR_MAIN_1) 
	mexType accum_x=0.0;
	mexType Dx;
#endif
#if defined(_PR_MAIN_2)
	mexType accum_y=0.0;
	mexType AvgInvRhoJ;
	mexType Dy;
#endif

_PT index;
_PT index2;
_PT  CurZone;
	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		{
		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )
			{
	#if defined(_PR_PML_1) || defined(_PR_PML_2)  
				index=Ind_MaterialMap(i,j);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 )
				{
    #if defined(_PR_PML_1)
					index=Ind_V_x_x(i,j);


		            Diff= i>0 && i<N1-2 ? CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
		                                  CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j))
					                      :i<N1-1 ? EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j):0;

					ELD(V_x_x,index) =InvDXDThp_I*(ELD(V_x_x,index)*DXDThp_I+
													AvgInvRhoI*
													Diff);

					index=Ind_V_y_x(i,j);
					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
					                      CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2))
					                      :j>0 && j<N2 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1):0;

					ELD(V_y_x,index) =InvDXDT_J*(
													ELD(V_y_x,index)*DXDT_J+
													AvgInvRhoI*
													Diff);

					
					index=Ind_V_x(i,j);
					index2=Ind_V_x_x(i,j);
					ELD(Vx,index)=ELD(V_x_x,index2)+ELD(V_y_x,index2);
		#endif
		#if defined(_PR_PML_2)			

				// For coeffs. for V_y

					index=Ind_V_x_y(i,j);

					Diff= i>1 && i<N1-1 ? CA *(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j)) -
					                      CB *(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j))
					                      :i>0 && i<N1 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j):0;

					ELD(V_x_y,index) =InvDXDT_I*(
													ELD(V_x_y,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_y(i,j);
					Diff= j>0 && j < N2-2 ? CA*( EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j)) -
					                        CB*( EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1))
					                        :j < N2-1 ? EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j):0;

					ELD(V_y_y,index) =InvDXDThp_J*(
												ELD(V_y_y,index)*DXDThp_J+
												AvgInvRhoI*
												Diff);

					index=Ind_V_y(i,j);
					index2=Ind_V_y_y(i,j);
					ELD(Vy,index)=ELD(V_x_y,index2)+ELD(V_y_y,index2);

		#endif
		
				 }
	#endif	
			}
			else
			{
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
				index=Ind_MaterialMap(i,j);
	#if defined(_PR_MAIN_1)
				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dx=CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
						CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j));

				Dx+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
						CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2));

				EL(Vx,i,j)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j);
	#endif
	#if defined(_PR_MAIN_2)
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dy=CA*(EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j) )-
						CB*(EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1));

				Dy+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j))-
						CB*(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j));
				
				EL(Vy,i,j)+=DT*AvgInvRhoJ*Dy;
				accum_y+=EL(Vy,i,j);
	#endif
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#endif
		}
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
  		if ((nStep < LengthSource) && TypeSource<2) //Source is particle displacement
  		{
			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
				source--; //need to use C index
  			  	value=ELD(SourceFunctions,nStep*NumberSources+source);
				if (TypeSource==0)
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)+=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)+=value*ELD(Oy,index);
					#endif

					
				}
				else
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)=value*ELD(Oy,index);
					#endif
				}

  			}
  		}
	#endif
		}
		#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
		if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
	    {
			if (ZoneCount>1)
			{
				#if defined(_PR_MAIN_1)
				accum_x/=ZoneCount;
				#endif
				#if defined(_PR_MAIN_2)
				accum_y/=ZoneCount;
				#endif

			}
			CurZone=0;
			index=IndN1N2(i,j,0);
			index2=N1*N2;
			if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
				#endif


			}
			if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
					index+=index2*NumberSelRMSPeakMaps;
			if (SelRMSorPeak & SEL_PEAK)
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
				#endif
				
			}


		}
		#endif
		
}
#undef _PML_KERNEL_TOP_BOTTOM



#undef _PR_PML_1
#undef _PR_PML_2
#endif

#define _PR_MAIN_1
#define _PR_MAIN_2
#define _MAIN_KERNEL
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA))
#define _PR_PML_1
#define _PR_PML_2
#endif
#if defined(CUDA)
extern "C" __global__ void MAIN_1_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
			,unsigned int nStep,unsigned int TypeSource)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
    _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);

#endif
#ifdef OPENCL
__kernel void MAIN_1_ParticleKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	, unsigned int nStep,
	unsigned int TypeSource)
{
	_PT i = (_PT) get_global_id(0);
	_PT j = (_PT) get_global_id(1);
	
#endif
#ifdef METAL
kernel void MAIN_1_ParticleKernel(
	METAL_PARAMS
	#ifndef METALCOMPUTE
  	_PT i = (_PT) gid.x;
  	_PT j = (_PT) gid.y;
  	
	#else
	#define nN1 (N1-PML_Thickness*2)
	#define nN2 (N2-PML_Thickness*2)
	
  	_PT j = (_PT) ((gid )/nN1);
  	_PT i = (_PT) (gid -j*nN1);
	#endif
#endif
#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	
	// Each i,j go from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
if (IsOnPML_J(j)==1 )
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML

#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
if (IsOnPML_I(i)==1 )
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size

#endif


#if defined(_MAIN_KERNEL)
i+=PML_Thickness;
j+=PML_Thickness;

#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
	_PT source;
	mexType value;
#endif
#if defined(_PR_PML_1) || defined(_PR_PML_2)  || defined(_PR_MAIN_1) 
	mexType AvgInvRhoI;
#endif

#if defined(_PR_PML_1) || defined(_PR_PML_2)
	mexType Diff;
#endif
#if defined(_PR_MAIN_1) 
	mexType accum_x=0.0;
	mexType Dx;
#endif
#if defined(_PR_MAIN_2)
	mexType accum_y=0.0;
	mexType AvgInvRhoJ;
	mexType Dy;
#endif

_PT index;
_PT index2;
_PT  CurZone;
	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		{
		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )
			{
	#if defined(_PR_PML_1) || defined(_PR_PML_2)  
				index=Ind_MaterialMap(i,j);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 )
				{
    #if defined(_PR_PML_1)
					index=Ind_V_x_x(i,j);


		            Diff= i>0 && i<N1-2 ? CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
		                                  CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j))
					                      :i<N1-1 ? EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j):0;

					ELD(V_x_x,index) =InvDXDThp_I*(ELD(V_x_x,index)*DXDThp_I+
													AvgInvRhoI*
													Diff);

					index=Ind_V_y_x(i,j);
					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
					                      CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2))
					                      :j>0 && j<N2 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1):0;

					ELD(V_y_x,index) =InvDXDT_J*(
													ELD(V_y_x,index)*DXDT_J+
													AvgInvRhoI*
													Diff);

					
					index=Ind_V_x(i,j);
					index2=Ind_V_x_x(i,j);
					ELD(Vx,index)=ELD(V_x_x,index2)+ELD(V_y_x,index2);
		#endif
		#if defined(_PR_PML_2)			

				// For coeffs. for V_y

					index=Ind_V_x_y(i,j);

					Diff= i>1 && i<N1-1 ? CA *(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j)) -
					                      CB *(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j))
					                      :i>0 && i<N1 ? EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j):0;

					ELD(V_x_y,index) =InvDXDT_I*(
													ELD(V_x_y,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_y(i,j);
					Diff= j>0 && j < N2-2 ? CA*( EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j)) -
					                        CB*( EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1))
					                        :j < N2-1 ? EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j):0;

					ELD(V_y_y,index) =InvDXDThp_J*(
												ELD(V_y_y,index)*DXDThp_J+
												AvgInvRhoI*
												Diff);

					index=Ind_V_y(i,j);
					index2=Ind_V_y_y(i,j);
					ELD(Vy,index)=ELD(V_x_y,index2)+ELD(V_y_y,index2);

		#endif
		
				 }
	#endif	
			}
			else
			{
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
				index=Ind_MaterialMap(i,j);
	#if defined(_PR_MAIN_1)
				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dx=CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
						CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j));

				Dx+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
						CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2));

				EL(Vx,i,j)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j);
	#endif
	#if defined(_PR_MAIN_2)
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dy=CA*(EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j) )-
						CB*(EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1));

				Dy+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j))-
						CB*(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j));
				
				EL(Vy,i,j)+=DT*AvgInvRhoJ*Dy;
				accum_y+=EL(Vy,i,j);
	#endif
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#endif
		}
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
  		if ((nStep < LengthSource) && TypeSource<2) //Source is particle displacement
  		{
			index=IndN1N2(i,j,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
				source--; //need to use C index
  			  	value=ELD(SourceFunctions,nStep*NumberSources+source);
				if (TypeSource==0)
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)+=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)+=value*ELD(Oy,index);
					#endif

					
				}
				else
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j)=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j)=value*ELD(Oy,index);
					#endif
				}

  			}
  		}
	#endif
		}
		#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) 
		if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
	    {
			if (ZoneCount>1)
			{
				#if defined(_PR_MAIN_1)
				accum_x/=ZoneCount;
				#endif
				#if defined(_PR_MAIN_2)
				accum_y/=ZoneCount;
				#endif

			}
			CurZone=0;
			index=IndN1N2(i,j,0);
			index2=N1*N2;
			if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
				#endif


			}
			if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
					index+=index2*NumberSelRMSPeakMaps;
			if (SelRMSorPeak & SEL_PEAK)
			{
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
				#endif
				
			}


		}
		#endif
		
}
#if defined(OPENCL) || (defined(CUDA) && !defined(USE_MINI_KERNELS_CUDA))
#undef _PR_PML_1
#undef _PR_PML_2
#endif
#undef _PR_MAIN_1
#undef _PR_MAIN_2
#undef _MAIN_KERNEL

#if defined(CUDA)
extern "C" __global__ void SnapShot(unsigned int SelK,mexType * Snapshots_pr,mexType * Sigma_xx_pr,mexType * Sigma_yy_pr,unsigned int CurrSnap)
{
	_PT i = (_PT) (blockIdx.x * blockDim.x + threadIdx.x);
  _PT j = (_PT) (blockIdx.y * blockDim.y + threadIdx.y);
#endif
#ifdef OPENCL
__kernel void SnapShot(unsigned int SelK,__global mexType * Snapshots_pr,__global mexType * Sigma_xx_pr,__global mexType * Sigma_yy_pr,unsigned int CurrSnap)
{
  _PT i = (_PT) get_global_id(0);
  _PT j = (_PT) get_global_id(1);
#endif
#ifdef METAL
#define Sigma_xx_pr k_Sigma_xx_pr
#define Sigma_yy_pr k_Sigma_yy_pr

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
	_PT i = (_PT) gid.x;
	_PT j = (_PT) gid.y;
#endif

    if (i>=N1 || j >=N2)
		return;
	// mexType accum=0.0;
	// for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
	// 	{
	// 		_PT index=Ind_Sigma_xx(i,j,(_PT)SelK);
	// 		accum+=(Sigma_xx_pr[index]+Sigma_yy_pr[index]+Sigma_zz_pr[index])/3.0;

	// 	}

	// 	Snapshots_pr[IndN1N2Snap(i,j)+CurrSnap*N1*N2]=accum/ZoneCount;
}

#if defined(CUDA)
extern "C" __global__ void SensorsKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
	,mexType * SensorOutput_pr,
	unsigned int * IndexSensorMap_pr,
	unsigned int nStep)
{
	unsigned int sj =blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef OPENCL
__kernel void SensorsKernel(
#ifdef CUDA
#define __predecorator 
#else
#define __predecorator __global
#endif
__predecorator mexType *V_x_x_pr,
__predecorator mexType *V_y_x_pr,
__predecorator mexType *V_x_y_pr,
__predecorator mexType *V_y_y_pr,
__predecorator mexType *Vx_pr,
__predecorator mexType *Vy_pr,
__predecorator mexType *Rxx_pr,
__predecorator mexType *Ryy_pr,
__predecorator mexType *Rxy_pr,
__predecorator mexType *Sigma_x_xx_pr,
__predecorator mexType *Sigma_y_xx_pr,
__predecorator mexType *Sigma_x_yy_pr,
__predecorator mexType *Sigma_y_yy_pr,
__predecorator mexType *Sigma_x_xy_pr,
__predecorator mexType *Sigma_y_xy_pr,
__predecorator mexType *Sigma_xy_pr,
__predecorator mexType *Sigma_xx_pr,
__predecorator mexType *Sigma_yy_pr,
__predecorator mexType *SourceFunctions_pr,
__predecorator mexType * LambdaMiuMatOverH_pr,
__predecorator mexType * LambdaMatOverH_pr,
__predecorator mexType * MiuMatOverH_pr,
__predecorator mexType * TauLong_pr,
__predecorator mexType * OneOverTauSigma_pr,
__predecorator mexType * TauShear_pr,
__predecorator mexType * InvRhoMatH_pr,
__predecorator mexType * SqrAcc_pr,
__predecorator unsigned int * MaterialMap_pr,
__predecorator unsigned int * SourceMap_pr,
__predecorator mexType * Ox_pr,
__predecorator mexType * Oy_pr,
__predecorator mexType * Pressure_pr
		, __global mexType * SensorOutput_pr,
			__global unsigned int * IndexSensorMap_pr,
			unsigned int nStep)
{
	_PT sj =(_PT) get_global_id(0);
#endif
#ifdef METAL

#define IndexSensorMap_pr k_IndexSensorMap_pr

#ifndef METALCOMPUTE
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
#else
kernel void SensorsKernel(
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],
	const device unsigned int *p_INDEX_MEX [[ buffer(1) ]],
	const device unsigned int *p_INDEX_UINT [[ buffer(2) ]],
	const device unsigned int *p_UINT_BUFFER [[ buffer(3) ]],
	device mexType * p_MEX_BUFFER_0 [[ buffer(4) ]],
	device mexType * p_MEX_BUFFER_1 [[ buffer(5) ]],
	device mexType * p_MEX_BUFFER_2 [[ buffer(6) ]],
	device mexType * p_MEX_BUFFER_3 [[ buffer(7) ]],
	device mexType * p_MEX_BUFFER_4 [[ buffer(8) ]],
	device mexType * p_MEX_BUFFER_5 [[ buffer(9) ]],
	device mexType * p_MEX_BUFFER_6 [[ buffer(10) ]],
	device mexType * p_MEX_BUFFER_7 [[ buffer(11) ]],
	device mexType * p_MEX_BUFFER_8 [[ buffer(12) ]],
	device mexType * p_MEX_BUFFER_9 [[ buffer(13) ]],
	device mexType * p_MEX_BUFFER_10 [[ buffer(14) ]],
	device mexType * p_MEX_BUFFER_11 [[ buffer(15) ]],
	uint gid[[thread_position_in_grid]])
#endif
{
	_PT sj = (_PT) gid;
#endif

	if (sj>=(_PT) NumberSensors)
		return;
_PT index=(((_PT)nStep)/((_PT)SensorSubSampling)-((_PT)SensorStart))*((_PT)NumberSensors)+(_PT)sj;
_PT  i,j;
_PT index2,index3,
    subarrsize=(((_PT)NumberSensors)*(((_PT)TimeSteps)/((_PT)SensorSubSampling)+1-((_PT)SensorStart)));
index2=IndexSensorMap_pr[sj]-1;

mexType accumX=0.0,accumY=0.0,
        accumXX=0.0, accumYY=0.0, 
        accumXY=0.0, accum_p=0;;
for (_PT CurZone=0;CurZone<ZoneCount;CurZone++)
  {
    i=index2%(N1);
    j=index2/N1;

    if ( IS_Vx_SELECTED(SelMapsSensors))
        accumX+=EL(Vx,i,j);
    if ( IS_Vy_SELECTED(SelMapsSensors))
        accumY+=EL(Vy,i,j);

    index3=Ind_Sigma_xx(i,j);
  #ifdef METAL
    //No idea why in this kernel the ELD(SigmaXX...) macros do not expand correctly
    //So we go a bit more manual
  if (IS_Sigmaxx_SELECTED(SelMapsSensors))
      accumXX+=k_Sigma_xx_pr[index3];
  if (IS_Sigmayy_SELECTED(SelMapsSensors))
      accumYY+=k_Sigma_yy_pr[index3];
  if (IS_Pressure_SELECTED(SelMapsSensors))
      accum_p+=k_Pressure_pr[index3];
  index3=Ind_Sigma_xy(i,j);
  if (IS_Sigmaxy_SELECTED(SelMapsSensors))
      accumXY+=k_Sigma_xy_pr[index3];
  
  #else
    if (IS_Sigmaxx_SELECTED(SelMapsSensors))
        accumXX+=ELD(Sigma_xx,index3);
    if (IS_Sigmayy_SELECTED(SelMapsSensors))
        accumYY+=ELD(Sigma_yy,index3);
    if (IS_Pressure_SELECTED(SelMapsSensors))
        accum_p+=ELD(Pressure,index3);
    index3=Ind_Sigma_xy(i,j);
    if (IS_Sigmaxy_SELECTED(SelMapsSensors))
        accumXY+=ELD(Sigma_xy,index3);
   #endif
  }
accumX/=ZoneCount;
accumY/=ZoneCount;
accumXX/=ZoneCount;
accumYY/=ZoneCount;
accumXY/=ZoneCount;
accum_p/=ZoneCount;
//ELD(SensorOutput,index)=accumX*accumX+accumY*accumY+accumZ*accumZ;
if (IS_Vx_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Vx)=accumX;
if (IS_Vy_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Vy)=accumY;
if (IS_Sigmaxx_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmaxx)=accumXX;
if (IS_Sigmayy_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmayy)=accumYY;
if (IS_Sigmaxy_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Sigmaxy)=accumXY;
if (IS_Pressure_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Pressure)=accum_p;

}