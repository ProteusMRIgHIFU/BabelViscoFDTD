
#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
	
	mexType Diff;
	mexType Diff2;
	mexType Dx;
	mexType Dy;
	mexType value;
	mexType m1;
	mexType m2;
	mexType m3;
	mexType m4;
	mexType RigidityXY=0.0;
	mexType TauShearXY=0.0;
	mexType LambdaMiu;
	mexType LambdaMiuComp;
	mexType dFirstPart;
	mexType dFirstPartForR;
	mexType accum_xx=0.0;
	mexType accum_yy=0.0;
	mexType accum_p=0.0;
	_PT source;
	_PT bAttenuating=1;
	mexType Miu;
	mexType MiuComp;
	mexType OneOverTauSigma;
	mexType NextR;

	mexType accum_xy=0.0;



#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
_PT index2;
_PT index;
_PT  MaterialID;
_PT CurZone;

for ( CurZone=0;CurZone<ZoneCount;CurZone++)
  {
	index=Ind_MaterialMap(i,j);
	MaterialID=ELD(MaterialMap,index);
	m1=ELD(MiuMatOverH,MaterialID);
	if (i<N1-1)
		m2=ELD(MiuMatOverH,EL(MaterialMap,i+1,j));
	else
		m2=m1;
	if (j<N2-1)
		m3=ELD(MiuMatOverH,EL(MaterialMap,i,j+1));
	else
		m3=m1;
	if (i<N1-1 && j<N2-1 )
		m4=ELD(MiuMatOverH,EL(MaterialMap,i+1,j+1));
	else
		m4 = m1;
	value=m1*m2*m3*m4;
	RigidityXY =value !=0.0 ? 4.0/(1.0/m1+1.0/m2+1.0/m3+1.0/m4):0.0;
	if (value ==0.0)
		TauShearXY = ELD(TauShear,MaterialID);
	else
	{
		m1 = ELD(TauShear,MaterialID);
		if (i<N1-1)
			m2=ELD(TauShear,EL(MaterialMap,i+1,j));
		else
			m2=m1;
		if (j<N2-1)
			m3=ELD(TauShear,EL(MaterialMap,i,j+1));
		else
			m3=m1;
		if (i<N1-1 && j<N2-1 )
			m4=ELD(TauShear,EL(MaterialMap,i+1,j+1));
		else
			m4=m1;

		TauShearXY=0.25*(m1+m2+m3+m4);
		
	}	   
  	
  	if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )//We are in the PML borders
  	 {

  		if (i<N1-1 && j <N2-1 )
  		{

  			Diff= i>1 && i <N1-1 ? CA*(EL(Vx,i,j)-EL(Vx,i-1,j)) -
  			                       CB*(EL(Vx,i+1,j)-EL(Vx,i-2,j))
  			      : i>0 && i <N1 ? (EL(Vx,i,j)-EL(Vx,i-1,j))  :0;

			Diff2= j>1 && j < N2-1 ? CA*(EL(Vy,i,j)-EL(Vy,i,j-1))-
  									CB*(EL(Vy,i,j+1)-EL(Vy,i,j-2))
  			        : j>0 && j < N2 ? EL(Vy,i,j)-EL(Vy,i,j-1):0;

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


		  }	   
	}
  	else
  	{
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

  		index=Ind_Sigma_xy(i,j);
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
	
  	}
  }
  if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
  {
    accum_xx/=ZoneCount;
    accum_yy/=ZoneCount;
    accum_xy/=ZoneCount;


    CurZone=0;
    index=IndN1N2(i,j,0);
    index2=N1*N2;


    if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
    {
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)+=accum_xx*accum_xx;
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)+=accum_yy*accum_yy;
    
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;
        if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)+=accum_xy*accum_xy;

		
    }
    if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
        index+=index2*NumberSelRMSPeakMaps;
    if (SelRMSorPeak & SEL_PEAK)
    {
        if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx)=accum_xx>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx) ? accum_xx: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxx);
        if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy)=accum_yy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy) ? accum_yy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmayy);
        if (IS_Pressure_SELECTED(SelMapsRMSPeak))
			ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p :ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);
		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
            ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy)=accum_xy>ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy) ? accum_xy: ELD(SqrAcc,index+index2*IndexRMSPeak_Sigmaxy);
    }

  }
