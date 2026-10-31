#if !defined(METALCOMPUTE) && !defined(MLX)
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
#else
#define nStep p_CONSTANT_BUFFER_UINT[CInd_nStep]
#define TypeSource p_CONSTANT_BUFFER_UINT[CInd_TypeSource]
#endif

#if defined(METALCOMPUTE)
// Each array is its own Metal buffer. Kernels receive them through three argument buffers grouped by access, as
// Metal limits a kernel to 31 buffer parameters. The host binds each array to the struct entry with the same name,
// reading the order of the [[id(n)]] entries from the kernel source, so these structs are the only place to edit
// when adding an array.
struct FDTD_ArgsIn { // read only
	const device mexType * LambdaMiuMatOverH [[id(0)]];
	const device mexType * LambdaMatOverH [[id(1)]];
	const device mexType * MiuMatOverH [[id(2)]];
	const device mexType * TauLong [[id(3)]];
	const device mexType * OneOverTauSigma [[id(4)]];
	const device mexType * TauShear [[id(5)]];
	const device mexType * InvRhoMatH [[id(6)]];
	const device mexType * Ox [[id(7)]];
	const device mexType * Oy [[id(8)]];
	const device mexType * Oz [[id(9)]];
	const device mexType * SourceFunctions [[id(10)]];
	const device unsigned int * IndexSensorMap [[id(11)]];
	const device unsigned int * SourceMap [[id(12)]];
	const device unsigned int * MaterialMap [[id(13)]];
	const device unsigned int * ReflectorMask [[id(14)]];
};

struct FDTD_ArgsInOut { // read and write
	device mexType * V_x_x [[id(0)]];
	device mexType * V_y_x [[id(1)]];
	device mexType * V_z_x [[id(2)]];
	device mexType * V_x_y [[id(3)]];
	device mexType * V_y_y [[id(4)]];
	device mexType * V_z_y [[id(5)]];
	device mexType * V_x_z [[id(6)]];
	device mexType * V_y_z [[id(7)]];
	device mexType * V_z_z [[id(8)]];
	device mexType * Vx [[id(9)]];
	device mexType * Vy [[id(10)]];
	device mexType * Vz [[id(11)]];
	device mexType * Rxx [[id(12)]];
	device mexType * Ryy [[id(13)]];
	device mexType * Rzz [[id(14)]];
	device mexType * Rxy [[id(15)]];
	device mexType * Rxz [[id(16)]];
	device mexType * Ryz [[id(17)]];
	device mexType * Sigma_x_xx [[id(18)]];
	device mexType * Sigma_y_xx [[id(19)]];
	device mexType * Sigma_z_xx [[id(20)]];
	device mexType * Sigma_x_yy [[id(21)]];
	device mexType * Sigma_y_yy [[id(22)]];
	device mexType * Sigma_z_yy [[id(23)]];
	device mexType * Sigma_x_zz [[id(24)]];
	device mexType * Sigma_y_zz [[id(25)]];
	device mexType * Sigma_z_zz [[id(26)]];
	device mexType * Sigma_x_xy [[id(27)]];
	device mexType * Sigma_y_xy [[id(28)]];
	device mexType * Sigma_x_xz [[id(29)]];
	device mexType * Sigma_z_xz [[id(30)]];
	device mexType * Sigma_y_yz [[id(31)]];
	device mexType * Sigma_z_yz [[id(32)]];
	device mexType * Sigma_xy [[id(33)]];
	device mexType * Sigma_xz [[id(34)]];
	device mexType * Sigma_yz [[id(35)]];
	device mexType * Sigma_xx [[id(36)]];
	device mexType * Sigma_yy [[id(37)]];
	device mexType * Sigma_zz [[id(38)]];
	device mexType * Pressure [[id(39)]];
	device mexType * SqrAcc [[id(40)]];
};

struct FDTD_ArgsOut { // write only
	device mexType * SensorOutput [[id(0)]];
};

#define k_LambdaMiuMatOverH_pr (ArgsIn.LambdaMiuMatOverH)
#define k_LambdaMatOverH_pr (ArgsIn.LambdaMatOverH)
#define k_MiuMatOverH_pr (ArgsIn.MiuMatOverH)
#define k_TauLong_pr (ArgsIn.TauLong)
#define k_OneOverTauSigma_pr (ArgsIn.OneOverTauSigma)
#define k_TauShear_pr (ArgsIn.TauShear)
#define k_InvRhoMatH_pr (ArgsIn.InvRhoMatH)
#define k_Ox_pr (ArgsIn.Ox)
#define k_Oy_pr (ArgsIn.Oy)
#define k_Oz_pr (ArgsIn.Oz)
#define k_SourceFunctions_pr (ArgsIn.SourceFunctions)
#define k_IndexSensorMap_pr (ArgsIn.IndexSensorMap)
#define k_SourceMap_pr (ArgsIn.SourceMap)
#define k_MaterialMap_pr (ArgsIn.MaterialMap)
#define k_ReflectorMask_pr (ArgsIn.ReflectorMask)
#define k_V_x_x_pr (ArgsInOut.V_x_x)
#define k_V_y_x_pr (ArgsInOut.V_y_x)
#define k_V_z_x_pr (ArgsInOut.V_z_x)
#define k_V_x_y_pr (ArgsInOut.V_x_y)
#define k_V_y_y_pr (ArgsInOut.V_y_y)
#define k_V_z_y_pr (ArgsInOut.V_z_y)
#define k_V_x_z_pr (ArgsInOut.V_x_z)
#define k_V_y_z_pr (ArgsInOut.V_y_z)
#define k_V_z_z_pr (ArgsInOut.V_z_z)
#define k_Vx_pr (ArgsInOut.Vx)
#define k_Vy_pr (ArgsInOut.Vy)
#define k_Vz_pr (ArgsInOut.Vz)
#define k_Rxx_pr (ArgsInOut.Rxx)
#define k_Ryy_pr (ArgsInOut.Ryy)
#define k_Rzz_pr (ArgsInOut.Rzz)
#define k_Rxy_pr (ArgsInOut.Rxy)
#define k_Rxz_pr (ArgsInOut.Rxz)
#define k_Ryz_pr (ArgsInOut.Ryz)
#define k_Sigma_x_xx_pr (ArgsInOut.Sigma_x_xx)
#define k_Sigma_y_xx_pr (ArgsInOut.Sigma_y_xx)
#define k_Sigma_z_xx_pr (ArgsInOut.Sigma_z_xx)
#define k_Sigma_x_yy_pr (ArgsInOut.Sigma_x_yy)
#define k_Sigma_y_yy_pr (ArgsInOut.Sigma_y_yy)
#define k_Sigma_z_yy_pr (ArgsInOut.Sigma_z_yy)
#define k_Sigma_x_zz_pr (ArgsInOut.Sigma_x_zz)
#define k_Sigma_y_zz_pr (ArgsInOut.Sigma_y_zz)
#define k_Sigma_z_zz_pr (ArgsInOut.Sigma_z_zz)
#define k_Sigma_x_xy_pr (ArgsInOut.Sigma_x_xy)
#define k_Sigma_y_xy_pr (ArgsInOut.Sigma_y_xy)
#define k_Sigma_x_xz_pr (ArgsInOut.Sigma_x_xz)
#define k_Sigma_z_xz_pr (ArgsInOut.Sigma_z_xz)
#define k_Sigma_y_yz_pr (ArgsInOut.Sigma_y_yz)
#define k_Sigma_z_yz_pr (ArgsInOut.Sigma_z_yz)
#define k_Sigma_xy_pr (ArgsInOut.Sigma_xy)
#define k_Sigma_xz_pr (ArgsInOut.Sigma_xz)
#define k_Sigma_yz_pr (ArgsInOut.Sigma_yz)
#define k_Sigma_xx_pr (ArgsInOut.Sigma_xx)
#define k_Sigma_yy_pr (ArgsInOut.Sigma_yy)
#define k_Sigma_zz_pr (ArgsInOut.Sigma_zz)
#define k_Pressure_pr (ArgsInOut.Pressure)
#define k_SqrAcc_pr (ArgsInOut.SqrAcc)
#define k_SensorOutput_pr (ArgsOut.SensorOutput)
#else
// Arrays packed in shared buffers, with their offsets read from p_INDEX_MEX/p_INDEX_UINT
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
#define k_ReflectorMask_pr	 __def_UINT_VAR(ReflectorMask)
#endif
