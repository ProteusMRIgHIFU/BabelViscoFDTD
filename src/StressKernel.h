    mexType Diff,value,Dx,Dy,Dz,m1,m2,m3,m4,RigidityXY=0.0,RigidityXZ=0.0,
        RigidityYZ=0.0,LambdaMiu,Miu,LambdaMiuComp,MiuComp,
        dFirstPart,OneOverTauSigma,dFirstPartForR,NextR,
            TauShearXY=0.0,TauShearXZ=0.0,TauShearYZ=0.0,
            accum_xx=0.0,accum_yy=0.0,accum_zz=0.0,
            accum_xy=0.0,accum_xz=0.0,accum_yz=0.0;
#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
   	unsigned int index,index2,MaterialID,CurZone;
for ( CurZone=0;CurZone<ZoneCount;CurZone++)
  {
  	if (i<N1 && j<N2 && k<N3)
  	{
	  //EL(Pressure_old,i,j,k)=EL(Pressure,i,j,k);

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

      if REQUIRES_2ND_ORDER_M(Y)
          Dz=EL(Vz,i,j,k)-EL(Vz,i,j,k-1);
      else
          Dz=CA*(EL(Vz,i,j,k)-EL(Vz,i,j,k-1))-
              CB*(EL(Vz,i,j,k+1)-EL(Vz,i,j,k-2));


  		LambdaMiu=ELD(LambdaMiuMatOverH,MaterialID)*(1.0+ELD(TauLong,MaterialID));
  		Miu=2.0*ELD(MiuMatOverH,MaterialID)*(1.0+ELD(TauShear,MaterialID));
  		OneOverTauSigma=ELD(OneOverTauSigma,MaterialID);
  		LambdaMiuComp=DT*ELD(LambdaMiuMatOverH,MaterialID)*(ELD(TauLong,MaterialID)*OneOverTauSigma);
  		MiuComp=DT*2.0*ELD(MiuMatOverH,MaterialID)*(ELD(TauShear,MaterialID)*OneOverTauSigma);


  		dFirstPart=LambdaMiu*(Dx+Dy+Dz);
  		dFirstPartForR=LambdaMiuComp*(Dx+Dy+Dz);

  		NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxx,index) - dFirstPartForR + MiuComp*(Dy+Dz))
  		      /(1+DT*0.5*OneOverTauSigma);
  		ELD(Sigma_xx,index)+=	DT*(dFirstPart - Miu*(Dy+Dz) + 0.5*(ELD(Rxx,index) + NextR));
      accum_xx+=ELD(Sigma_xx,index);

  		ELD(Rxx,index)=NextR;

  		NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryy,index) - dFirstPartForR + MiuComp*(Dx+Dz))
  		      /(1+DT*0.5*OneOverTauSigma);
  		ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx+Dz) + 0.5*(ELD(Ryy,index) + NextR));
      accum_yy+=ELD(Sigma_yy,index);

  		ELD(Ryy,index)=NextR;

  		NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rzz,index) - dFirstPartForR +MiuComp*(Dx+Dy))
  		      /(1+DT*0.5*OneOverTauSigma);
  		ELD(Sigma_zz,index)+=	DT*(dFirstPart - Miu*(Dx+Dy) + 0.5*(ELD(Rzz,index) + NextR));
      accum_zz+=ELD(Sigma_zz,index);

  		ELD(Rzz,index)=NextR;

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
  			MiuComp=RigidityXY*(TauShearXY*OneOverTauSigma);

  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxy,index) - DT*MiuComp*Dx)
  		          /(1+DT*0.5*OneOverTauSigma);

  			ELD(Sigma_xy,index)+= DT*(Miu*Dx + 0.5*(ELD(Rxy,index) +NextR));
        accum_xy+=ELD(Sigma_xy,index);

  			ELD(Rxy,index)=NextR;
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
  			MiuComp=RigidityXZ*(TauShearXZ*OneOverTauSigma);

  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rxz,index) - DT*MiuComp*Dx)
  		          /(1+DT*0.5*OneOverTauSigma);

  			ELD(Sigma_xz,index)+= DT*(Miu*Dx + 0.5*(ELD(Rxz,index) +NextR));
        accum_xz+=ELD(Sigma_xz,index);

  			ELD(Rxz,index)=NextR;
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
  			MiuComp=RigidityYZ*(TauShearYZ*OneOverTauSigma);

  			NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryz,index) - DT*MiuComp*Dy)
  		          /(1+DT*0.5*OneOverTauSigma);

  			ELD(Sigma_yz,index)+= DT*(Miu*Dy + 0.5*(ELD(Ryz,index) +NextR));
        accum_yz+=ELD(Sigma_yz,index);

  			ELD(Ryz,index)=NextR;

  		}
          else
              ELD(Ryz,index)=0.0;

  	}
  }
  if (i<N1 && j <N2 && k < N3 )
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
    }

  }
