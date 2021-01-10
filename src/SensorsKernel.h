unsigned int index=(nStep/SensorSteps)*NumberSensors+(unsigned int)sj;
unsigned int i,j,k,index2,index3,
subarrsize=(NumberSensors*(TimeSteps/SensorSteps+1));

index2=IndexSensorMap_pr[sj]-1;

mexType accumX=0.0,accumY=0.0,accumZ=0.0,
        accumXX=0.0, accumYY=0.0, accumZZ=0.0,
        accumXY=0.0, accumXZ=0.0, accumYZ=0.0;
for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
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
    if (IS_Sigmaxx_SELECTED(SelMapsSensors))
        accumXX+=ELD(Sigma_xx,index3);
    if (IS_Sigmayy_SELECTED(SelMapsSensors))
        accumYY+=ELD(Sigma_yy,index3);
    if (IS_Sigmazz_SELECTED(SelMapsSensors))
        accumZZ+=ELD(Sigma_zz,index3);
    index3=Ind_Sigma_xy(i,j,k);
    if (IS_Sigmaxy_SELECTED(SelMapsSensors))
        accumXY+=ELD(Sigma_xy,index3);
    if (IS_Sigmaxz_SELECTED(SelMapsSensors))
        accumXZ+=ELD(Sigma_xz,index3);
    if (IS_Sigmayz_SELECTED(SelMapsSensors))
        accumYZ+=ELD(Sigma_yz,index3);
  }
accumX=accumX/ZoneCount;
accumY=accumY/ZoneCount;
accumZ=accumZ/ZoneCount;
accumXX=accumXX/ZoneCount;
accumYY=accumYY/ZoneCount;
accumZZ=accumZZ/ZoneCount;
accumXY=accumXY/ZoneCount;
accumXZ=accumXZ/ZoneCount;
accumYZ=accumYZ/ZoneCount;
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
