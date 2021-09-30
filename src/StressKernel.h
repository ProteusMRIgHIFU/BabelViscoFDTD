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
   	_PT index,index2, MaterialID,source,bAttenuating=1;
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

//#define _DebugPrint
#ifdef _DebugPrint
	int bDoPrint=0;
	if (i==64 && j==64 && k==117)
		bDoPrint=1;
#endif
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
			#ifdef _DebugPrint
			#ifdef OPENCL
			if (bDoPrint)
				printf("Capturing RMS  Pressure %g,%g,%lu,%lu\n",ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure),
							DT,index,index2*IndexRMSPeak_Pressure);
			#endif
			#endif
		}
		#ifdef _DebugPrint
		#ifdef OPENCL
		else{
		if (bDoPrint)
			printf("Capturing Pressure RMS not enabled \n");
		}
		#endif
		#endif
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
			#ifdef _DebugPrint
			#ifdef OPENCL
			if (bDoPrint)
				printf("Capturing peak  Pressure %g,%g\n",ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure),DT);
			#endif
			#endif
		}
		#ifdef _DebugPrint
		#ifdef OPENCL
		else
		{
		if (bDoPrint)
			printf("Capturing Pressure Peak not enabled \n");
		}
		#endif
		#endif
    }

  }
