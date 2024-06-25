_PT index=(((_PT)nStep)/((_PT)SensorSubSampling)-((_PT)SensorStart))*((_PT)NumberSensors)+(_PT)sj;
_PT  i,j;
_PT index2,index3,
    subarrsize=(((_PT)NumberSensors)*(((_PT)TimeSteps)/((_PT)SensorSubSampling)+1-((_PT)SensorStart)));
index2=IndexSensorMap_pr[sj]-1;

mexType accumX=0.0,accumY=0.0,
        accumXX=0.0, accumYY=0.0, 
        accumXY=0.0, accum_p=0, accum_p_gx=0,accum_p_gy=0;
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
  if (IS_Pressure_Gx_SELECTED(SelMapsSensors))
      accum_p_gx+=(k_Pressure_pr[Ind_Sigma_xx(i+1,j)]-k_Pressure_pr[Ind_Sigma_xx(i-1,j)])*0.5;
  if (IS_Pressure_Gy_SELECTED(SelMapsSensors))
      accum_p_gy+=(k_Pressure_pr[Ind_Sigma_xx(i,j+1)]-k_Pressure_pr[Ind_Sigma_xx(i,j-1)])*0.5;
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
    if (IS_Pressure_Gx_SELECTED(SelMapsSensors))
      accum_p_gx+=(Pressure_pr[Ind_Sigma_xx(i+1,j)]-Pressure_pr[Ind_Sigma_xx(i-1,j)])*0.5;
    if (IS_Pressure_Gy_SELECTED(SelMapsSensors))
      accum_p_gy+=(Pressure_pr[Ind_Sigma_xx(i,j+1)]-Pressure_pr[Ind_Sigma_xx(i,j-1)])*0.5;
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
if (IS_Pressure_Gx_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Pressure_gx)=accum_p_gx;
if (IS_Pressure_Gy_SELECTED(SelMapsSensors))
    ELD(SensorOutput,index+subarrsize*IndexSensor_Pressure_gy)=accum_p_gy;

