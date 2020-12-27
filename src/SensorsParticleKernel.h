unsigned int index=nStep*NumberSensors+(unsigned int)sj;
unsigned int i,j,k,index2;
index2=IndexSensorMap_pr[sj]-1;

mexType accumX=0.0,accumY=0.0,accumZ=0.0;
for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
  {
    k=index2/(N1*N2);
    j=index2%(N1*N2);
    i=j%(N1);
    j=j/N1;
    accumX+=Vx_pr[Ind_V_x(i,j,k)];
    accumY+=Vy_pr[Ind_V_y(i,j,k)];
    accumZ+=Vz_pr[Ind_V_z(i,j,k)];
  }
accumX=accumX/ZoneCount;
accumY=accumY/ZoneCount;
accumZ=accumZ/ZoneCount;
//SensorOutput_pr[index]=accumX*accumX+accumY*accumY+accumZ*accumZ;
SensorOutput_pr[index]=accumX;
SensorOutput_pr[index+NumberSensors*TimeSteps]=accumY;
SensorOutput_pr[index+NumberSensors*TimeSteps*2]=accumZ;
