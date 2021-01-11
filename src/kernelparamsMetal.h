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
#define NumberSelSensorMaps p_CONSTANT_BUFFER_UINT[CInd_NumberSelSensorMaps]
#define SensorSteps p_CONSTANT_BUFFER_UINT[CInd_SensorSteps]
#define nStep p_CONSTANT_BUFFER_UINT[CInd_nStep]
#define CurrSnap p_CONSTANT_BUFFER_UINT[CInd_CurrSnap]
#define TypeSource p_CONSTANT_BUFFER_UINT[CInd_TypeSource]
#define SelK p_CONSTANT_BUFFER_UINT[CInd_SelK]

#define DT p_CONSTANT_BUFFER_MEX[CInd_DT]
#define InvDXDTplus_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplus)
#define DXDTminus_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminus)
#define InvDXDTplushp_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplushp)
#define DXDTminushp_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminushp)

#define __def_MEX_VAR(__NameVar)  (&p_MEX_BUFFER[p_INDEX_MEX[CInd_ ##__NameVar]])
#define __def_UINT_VAR(__NameVar)  (&p_UINT_BUFFER[p_INDEX_UINT[CInd_ ##__NameVar]])

#define k_V_x_x_pr  __def_MEX_VAR(V_x_x)
#define k_V_y_x_pr  __def_MEX_VAR(V_y_x)
#define k_V_z_x_pr  __def_MEX_VAR(V_z_x)
#define k_V_x_y_pr  __def_MEX_VAR(V_x_y)
#define k_V_y_y_pr  __def_MEX_VAR(V_y_y)
#define k_V_z_y_pr  __def_MEX_VAR(V_z_y)
#define k_V_x_z_pr  __def_MEX_VAR(V_x_z)
#define k_V_y_z_pr  __def_MEX_VAR(V_y_z)
#define k_V_z_z_pr  __def_MEX_VAR(V_z_z)
#define k_Sigma_x_xx_pr  __def_MEX_VAR(Sigma_x_xx)
#define k_Sigma_y_xx_pr  __def_MEX_VAR(Sigma_y_xx)
#define k_Sigma_z_xx_pr  __def_MEX_VAR(Sigma_z_xx)
#define k_Sigma_x_yy_pr  __def_MEX_VAR(Sigma_x_yy)
#define k_Sigma_y_yy_pr  __def_MEX_VAR(Sigma_y_yy)
#define k_Sigma_z_yy_pr  __def_MEX_VAR(Sigma_z_yy)
#define k_Sigma_x_zz_pr  __def_MEX_VAR(Sigma_x_zz)
#define k_Sigma_y_zz_pr  __def_MEX_VAR(Sigma_y_zz)
#define k_Sigma_z_zz_pr  __def_MEX_VAR(Sigma_z_zz)
#define k_Sigma_x_xy_pr  __def_MEX_VAR(Sigma_x_xy)
#define k_Sigma_y_xy_pr  __def_MEX_VAR(Sigma_y_xy)
#define k_Sigma_x_xz_pr  __def_MEX_VAR(Sigma_x_xz)
#define k_Sigma_z_xz_pr  __def_MEX_VAR(Sigma_z_xz)
#define k_Sigma_y_yz_pr  __def_MEX_VAR(Sigma_y_yz)
#define k_Sigma_z_yz_pr  __def_MEX_VAR(Sigma_z_yz)
#define k_Rxx_pr  __def_MEX_VAR(Rxx)
#define k_Ryy_pr  __def_MEX_VAR(Ryy)
#define k_Rzz_pr  __def_MEX_VAR(Rzz)
#define k_Rxy_pr  __def_MEX_VAR(Rxy)
#define k_Rxz_pr  __def_MEX_VAR(Rxz)
#define k_Ryz_pr  __def_MEX_VAR(Ryz)


#define k_LambdaMiuMatOverH_pr  __def_MEX_VAR(LambdaMiuMatOverH)
#define k_LambdaMatOverH_pr  __def_MEX_VAR(LambdaMatOverH)
#define k_MiuMatOverH_pr  __def_MEX_VAR(MiuMatOverH)
#define k_TauLong_pr  __def_MEX_VAR(TauLong)
#define k_OneOverTauSigma_pr  __def_MEX_VAR(OneOverTauSigma)
#define k_TauShear_pr  __def_MEX_VAR(TauShear)
#define k_InvRhoMatH_pr  __def_MEX_VAR(InvRhoMatH)
#define k_Ox_pr  __def_MEX_VAR(Ox)
#define k_Oy_pr  __def_MEX_VAR(Oy)
#define k_Oz_pr  __def_MEX_VAR(Oz)


#define k_Vx_pr  __def_MEX_VAR(Vx)
#define k_Vy_pr  __def_MEX_VAR(Vy)
#define k_Vz_pr  __def_MEX_VAR(Vz)
#define k_Sigma_xx_pr  __def_MEX_VAR(Sigma_xx)
#define k_Sigma_yy_pr  __def_MEX_VAR(Sigma_yy)
#define k_Sigma_zz_pr  __def_MEX_VAR(Sigma_zz)
#define k_Sigma_xy_pr  __def_MEX_VAR(Sigma_xy)
#define k_Sigma_xz_pr  __def_MEX_VAR(Sigma_xz)
#define k_Sigma_yz_pr  __def_MEX_VAR(Sigma_yz)

#define k_SensorOutput_pr  __def_MEX_VAR(SensorOutput)
#define k_SqrAcc_pr  __def_MEX_VAR(SqrAcc)
#define k_SourceFunctions_pr __def_MEX_VAR(SourceFunctions)

#define k_IndexSensorMap_pr  __def_UINT_VAR(IndexSensorMap)
#define k_SourceMap_pr		 __def_UINT_VAR(SourceMap)
#define k_MaterialMap_pr	 __def_UINT_VAR(MaterialMap)
