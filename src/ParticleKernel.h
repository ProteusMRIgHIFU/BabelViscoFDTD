#if defined(METAL) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_CORNER) 
	i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
	j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
	k=k>Limit_K_low_PML ? k -Limit_K_low_PML-1+Limit_K_up_PML:k;
	// Each i,j,k go from 0 -> 2 x PML size
#endif
#if defined(_PML_KERNEL_LEFT_RIGHT)
j+=PML_Thickness;
k+=PML_Thickness;
if (IsOnPML_J(j)==1 ||  IsOnPML_K(k)==1)
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
//  i go from 0 -> 2 x PML size
//  j go from  PML size to N2 - PML
//  k go from  PML size to N3 - PML
#endif

#if defined(_PML_KERNEL_TOP_BOTTOM)
i+=PML_Thickness;
k+=PML_Thickness;
if (IsOnPML_I(i)==1 ||  IsOnPML_K(k)==1)
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from  PML size to N1 - PML
//  j go from 0 -> 2 x PML size
//  k go from  PML size to N3 - PML
#endif

#if defined(_PML_KERNEL_FRONT_BACK)
i+=PML_Thickness;
j+=PML_Thickness;
if (IsOnPML_I(i)==1 ||  IsOnPML_J(j)==1)
	return;
k=k>Limit_K_low_PML ? k -Limit_K_low_PML-1+Limit_K_up_PML:k;
//  i go from  PML size to N1 - PML
//  j go from  PML size to N2 - PML
//  K go from 0 -> 2 x PML size
#endif

#if defined(_PML_KERNEL_LEFT_RIGHT_RODS)
k+=PML_Thickness;
if (IsOnPML_K(k)==1)
	return;
i=i>Limit_I_low_PML ? i -Limit_I_low_PML-1+Limit_I_up_PML:i;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
//  i go from 0 -> 2 x PML size
//  j go from  0 -> 2 x PML size
//  k go from  PML size to N3 - PML
#endif

#if defined(_PML_KERNEL_BOTTOM_TOP_RODS)
i+=PML_Thickness;
if (IsOnPML_I(i)==1)
	return;
j=j>Limit_J_low_PML ? j -Limit_J_low_PML-1+Limit_J_up_PML:j;
k=k>Limit_K_low_PML ? k -Limit_K_low_PML-1+Limit_K_up_PML:k;
//  i go from PML size to N1 - PML
//  j go from  0 -> 2 x PML size
//  k go from  0 -> 2 x PML size
#endif

#if defined(_MAIN_KERNEL)
i+=PML_Thickness;
j+=PML_Thickness;
k+=PML_Thickness;
#endif
#endif

#if defined(OPENCL) || defined(METAL) || defined(CUDA)
if (i>=N1 || j >=N2  || k>=N3)
	return;
#endif
#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) ||  defined(_PR_MAIN_3)
	_PT source;
	mexType value;
#endif
#if defined(_PR_PML_1) || defined(_PR_PML_2) ||  defined(_PR_PML_3) || defined(_PR_MAIN_1) 
	mexType AvgInvRhoI;
#endif

#if defined(_PR_PML_1) || defined(_PR_PML_2) ||  defined(_PR_PML_3)
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
#if defined(_PR_MAIN_3)
	mexType accum_z=0.0;
	mexType AvgInvRhoK;
	mexType Dz;
#endif
_PT index;
_PT index2;
_PT  CurZone;
	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		{
		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 || IsOnPML_K(k)==1)
			{
	#if defined(_PR_PML_1) || defined(_PR_PML_2) ||  defined(_PR_PML_3)
				index=Ind_MaterialMap(i,j,k);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 && k<N3-1)
				{
    #if defined(_PR_PML_1)
					index=Ind_V_x_x(i,j,k);


		            Diff= i>0 && i<N1-2 ? CA*(EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k))-
		                                  CB*(EL(Sigma_xx,i+2,j,k)-EL(Sigma_xx,i-1,j,k))
					                      :i<N1-1 ? EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k):0;

					ELD(V_x_x,index) =InvDXDThp_I*(ELD(V_x_x,index)*DXDThp_I+
													AvgInvRhoI*
													Diff);

					index=Ind_V_y_x(i,j,k);
					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k))-
					                      CB*(EL(Sigma_xy,i,j+1,k)-EL(Sigma_xy,i,j-2,k))
					                      :j>0 && j<N2 ? EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k):0;

					ELD(V_y_x,index) =InvDXDT_J*(
													ELD(V_y_x,index)*DXDT_J+
													AvgInvRhoI*
													Diff);
					index=Ind_V_z_x(i,j,k);
					Diff= k >1 && k < N3-1 ? CA*( EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1))-
					                         CB*( EL(Sigma_xz,i,j,k+1)-EL(Sigma_xz,i,j,k-2)) :
		                                     k >0 && k < N3 ?  EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1):0;

					ELD(V_z_x,index) =InvDXDT_K*(
													ELD(V_z_x,index)*DXDT_K+
													AvgInvRhoI*
													Diff);

					
					index=Ind_V_x(i,j,k);
					index2=Ind_V_x_x(i,j,k);
					ELD(Vx,index)=ELD(V_x_x,index2)+ELD(V_y_x,index2)+ELD(V_z_x,index2);
		#endif
		#if defined(_PR_PML_2)			

				// For coeffs. for V_y

					index=Ind_V_x_y(i,j,k);

					Diff= i>1 && i<N1-1 ? CA *(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k)) -
					                      CB *(EL(Sigma_xy,i+1,j,k)-EL(Sigma_xy,i-2,j,k))
					                      :i>0 && i<N1 ? EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k):0;

					ELD(V_x_y,index) =InvDXDT_I*(
													ELD(V_x_y,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_y(i,j,k);
					Diff= j>0 && j < N2-2 ? CA*( EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k)) -
					                        CB*( EL(Sigma_yy,i,j+2,k)-EL(Sigma_yy,i,j-1,k))
					                        :j < N2-1 ? EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k):0;

					ELD(V_y_y,index) =InvDXDThp_J*(
												ELD(V_y_y,index)*DXDThp_J+
												AvgInvRhoI*
												Diff);
					index=Ind_V_y_z(i,j,k);

					Diff= k>1  && k <N3-1 ? CA*(EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1) )-
					                        CB*(EL(Sigma_yz,i,j,k+1)-EL(Sigma_yz,i,j,k-2) )
					                        :k>0  && k <N3 ? EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1):0;

					ELD(V_z_y,index) =InvDXDT_K*(
												ELD(V_z_y,index)*DXDT_K+
												AvgInvRhoI*
												Diff);

					index=Ind_V_y(i,j,k);
					index2=Ind_V_y_y(i,j,k);
					ELD(Vy,index)=ELD(V_x_y,index2)+ELD(V_y_y,index2)+ELD(V_z_y,index2);

		#endif
		#if defined(_PR_PML_3)		

					index=Ind_V_x_z(i,j,k);

					Diff= i> 1 && i <N1-1 ? CA*( EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k)) -
					                        CB*( EL(Sigma_xz,i+1,j,k)-EL(Sigma_xz,i-2,j,k))
					                        :i> 0 && i <N1 ? EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k):0;

					ELD(V_x_z,index) =InvDXDT_I*(
													ELD(V_x_z,index)*DXDT_I+
													AvgInvRhoI*
													Diff);
					index=Ind_V_y_z(i,j,k);

					Diff= j>1 && j<N2-1 ? CA*(EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k)) -
					                      CB*(EL(Sigma_yz,i,j+1,k)-EL(Sigma_yz,i,j-2,k)):
		                                  j>0 && j<N2 ? EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k):0;

					ELD(V_y_z,index) =InvDXDT_J*(
												ELD(V_y_z,index)*DXDT_J+
												AvgInvRhoI*
												Diff);
					index=Ind_V_z_z(i,j,k);

					Diff= k>0 && k< N3-2 ? CA*(EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k) )-
					                       CB*(EL(Sigma_zz,i,j,k+2)-EL(Sigma_zz,i,j,k-1) ):
		                                   k< N3-1 ? EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k) :0;

					ELD(V_z_z,index) =InvDXDThp_K*(
												ELD(V_z_z,index)*DXDThp_K+
												AvgInvRhoI*
												Diff);

					index=Ind_V_z(i,j,k);
					index2=Ind_V_z_z(i,j,k);
					ELD(Vz,index)=ELD(V_x_z,index2)+ELD(V_y_z,index2)+ELD(V_z_z,index2);
		#endif
				 }
	#endif	
			}
			else
			{
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) ||  defined(_PR_MAIN_3)
				index=Ind_MaterialMap(i,j,k);
	#if defined(_PR_MAIN_1)
				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j,k))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dx=CA*(EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k))-
						CB*(EL(Sigma_xx,i+2,j,k)-EL(Sigma_xx,i-1,j,k));

				Dx+=CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k))-
						CB*(EL(Sigma_xy,i,j+1,k)-EL(Sigma_xy,i,j-2,k));

				Dx+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1))-
						CB*(EL(Sigma_xz,i,j,k+1)-EL(Sigma_xz,i,j,k-2));

				EL(Vx,i,j,k)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j,k);
	#endif
	#if defined(_PR_MAIN_2)
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1,k))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dy=CA*(EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k) )-
						CB*(EL(Sigma_yy,i,j+2,k)-EL(Sigma_yy,i,j-1,k));

				Dy+=CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k))-
						CB*(EL(Sigma_xy,i+1,j,k)-EL(Sigma_xy,i-2,j,k));

				Dy+=CA*( EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1))-
					CB*(EL(Sigma_yz,i,j,k+1)-EL(Sigma_yz,i,j,k-2));
				
				EL(Vy,i,j,k)+=DT*AvgInvRhoJ*Dy;
				accum_y+=EL(Vy,i,j,k);
	#endif
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#if defined(_PR_MAIN_3)
				AvgInvRhoK=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j,k+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				Dz=CA*(EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k))-
						CB*( EL(Sigma_zz,i,j,k+2)-EL(Sigma_zz,i,j,k-1));

				Dz+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k))-
					CB*(EL(Sigma_xz,i+1,j,k)-EL(Sigma_xz,i-2,j,k));

				Dz+=CA*( EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k))-
					CB*(EL(Sigma_yz,i,j+1,k)-EL(Sigma_yz,i,j-2,k));

				EL(Vz,i,j,k)+=DT*AvgInvRhoK*Dz;
				accum_z+=EL(Vz,i,j,k);
	#endif
	#endif
		}
	#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) ||  defined(_PR_MAIN_3)
  		if ((nStep < LengthSource) && TypeSource<2) //Source is particle displacement
  		{
			index=IndN1N2N3(i,j,k,0);
  			source=ELD(SourceMap,index);
  			if (source>0)
  			{
				source--; //need to use C index
  			  	value=ELD(SourceFunctions,nStep*NumberSources+source);
				if (TypeSource==0)
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j,k)+=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j,k)+=value*ELD(Oy,index);
					#endif
					#if defined(_PR_MAIN_3)
					EL(Vz,i,j,k)+=value*ELD(Oz,index);
					#endif
					
				}
				else
				{
					#if defined(_PR_MAIN_1)
					EL(Vx,i,j,k)=value*ELD(Ox,index);
					#endif
					#if defined(_PR_MAIN_2)
					EL(Vy,i,j,k)=value*ELD(Oy,index);
					#endif
					#if defined(_PR_MAIN_3)
					EL(Vz,i,j,k)=value*ELD(Oz,index);
					#endif
				}

  			}
  		}
	#endif
		}
		#if defined(_PR_MAIN_1) || defined(_PR_MAIN_2) || defined(_PR_MAIN_3)
		if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0 && IsOnPML_K(k)==0 && nStep>=SensorStart*SensorSubSampling)
	    {
			if (ZoneCount>1)
			{
				#if defined(_PR_MAIN_1)
				accum_x/=ZoneCount;
				#endif
				#if defined(_PR_MAIN_2)
				accum_y/=ZoneCount;
				#endif
				#if defined(_PR_MAIN_3)
				accum_z/=ZoneCount;
				#endif
			}
			CurZone=0;
			index=IndN1N2N3(i,j,k,0);
			index2=N1*N2*N3;
			if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
			{
				// if (IS_ALLV_SELECTED(SelMapsRMSPeak)) #we need later to see how to tackle this in case of need
				// 	ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV)+=accum_x*accum_x  +  accum_y*accum_y  +  accum_z*accum_z;
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
				#endif
				#if defined(_PR_MAIN_3)
				if (IS_Vz_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vz)+=accum_z*accum_z;
				#endif

			}
			if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
					index+=index2*NumberSelRMSPeakMaps;
			if (SelRMSorPeak & SEL_PEAK)
			{
				// if (IS_ALLV_SELECTED(SelMapsRMSPeak))
				// {
				// 	value=accum_x*accum_x  +  accum_y*accum_y  +  accum_z*accum_z; //in the Python function we will do the final sqr root`
				// 	ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV)=value > ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV) ? value : ELD(SqrAcc,index+index2*IndexRMSPeak_ALLV);
				// }
				#if defined(_PR_MAIN_1)
				if (IS_Vx_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
				#endif
				#if defined(_PR_MAIN_2)
				if (IS_Vy_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
				#endif
				#if defined(_PR_MAIN_3)
				if (IS_Vz_SELECTED(SelMapsRMSPeak))
						ELD(SqrAcc,index+index2*IndexRMSPeak_Vz)=accum_z > ELD(SqrAcc,index+index2*IndexRMSPeak_Vz) ? accum_z : ELD(SqrAcc,index+index2*IndexRMSPeak_Vz);
				#endif
			}


		}
		#endif
		
