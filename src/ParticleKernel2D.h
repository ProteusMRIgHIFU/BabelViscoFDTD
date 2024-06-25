
#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  )
	return;
#endif
	_PT source;
	mexType value;
	mexType AvgInvRhoI;
	mexType Diff;
	mexType accum_x=0.0;
	mexType Dx;
	mexType accum_y=0.0;
	mexType AvgInvRhoJ;
	mexType Dy;

_PT index;
_PT index2;
_PT  CurZone;
	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		{
		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 )
			{
				index=Ind_MaterialMap(i,j);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 )
				{
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


		
				 }

			}
			else
			{
				index=Ind_MaterialMap(i,j);
				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dx=CA*(EL(Sigma_xx,i+1,j)-EL(Sigma_xx,i,j))-
						CB*(EL(Sigma_xx,i+2,j)-EL(Sigma_xx,i-1,j));

				Dx+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i,j-1))-
						CB*(EL(Sigma_xy,i,j+1)-EL(Sigma_xy,i,j-2));

				EL(Vx,i,j)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j);
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dy=CA*(EL(Sigma_yy,i,j+1)-EL(Sigma_yy,i,j) )-
						CB*(EL(Sigma_yy,i,j+2)-EL(Sigma_yy,i,j-1));

				Dy+=CA*(EL(Sigma_xy,i,j)-EL(Sigma_xy,i-1,j))-
						CB*(EL(Sigma_xy,i+1,j)-EL(Sigma_xy,i-2,j));
				
				EL(Vy,i,j)+=DT*AvgInvRhoJ*Dy;
				accum_y+=EL(Vy,i,j);
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		}
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
					EL(Vx,i,j)+=value*ELD(Ox,index);
					EL(Vy,i,j)+=value*ELD(Oy,index);
				
				}
				else
				{
					EL(Vx,i,j)=value*ELD(Ox,index);
					EL(Vy,i,j)=value*ELD(Oy,index);
				}

  			}
  		}
		}
		if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0  && nStep>=SensorStart*SensorSubSampling)
	    {
			if (ZoneCount>1)
			{
				accum_x/=ZoneCount;
				accum_y/=ZoneCount;
			}
			CurZone=0;
			index=IndN1N2(i,j,0);
			index2=N1*N2;
			if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
			{
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
			}
			if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
					index+=index2*NumberSelRMSPeakMaps;
			if (SelRMSorPeak & SEL_PEAK)
			{
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
				
			}


		}
		
		
