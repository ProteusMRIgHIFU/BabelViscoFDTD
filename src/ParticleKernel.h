#if (defined(METAL) && !defined(METAL_SINGLE_KERNEL)) || defined(USE_MINI_KERNELS_CUDA)
#if defined(_PML_KERNEL_I_BOTTOM) 
//  i go from 0 to PML 
//  j go from  0 to  N2
//  k go from  0 to N3
// so no initial corrections on indices
#endif
#if defined(_PML_KERNEL_I_TOP)
i+=Limit_I_up_PML;
//  i go from  N1 - PML to N1
//  j go from  0 to N2
//  k go from  0 to N3
#endif

#if defined(_PML_KERNEL_J_BOTTOM)
i+=PML_Thickness;
//  i go from  PML to N1 - PML
//  j go from  0 to PML
//  k go from  0 to N3

#endif

#if defined(_PML_KERNEL_J_TOP)
i+=PML_Thickness;
j+=Limit_J_up_PML;
//  i go from  PML to N1 - PML
//  j go from  N2 - PML to N2
//  k go from  0 to N3
#endif

#if defined(_PML_KERNEL_K_BOTTOM)
i+=PML_Thickness;
j+=PML_Thickness;
//  i go from  PML to N1 - PML
//  j go from  PML to N2 - PML
//  k go from  0 to PML

#endif

#if defined(_PML_KERNEL_K_TOP)
i+=PML_Thickness;
j+=PML_Thickness;
k+=Limit_K_up_PML;
//  i go from  PML to N1 - PML
//  j go from  PML to N2 - PML
//  k go from  N3 - PML to N3
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
#if defined(_PR_MAIN)
	_PT source;
	mexType value;
#endif

	mexType AvgInvRhoI;

#if defined(_PR_PML)
	mexType Diff;
#endif
#if defined(_PR_MAIN) 
	mexType accum_x=0.0;
	mexType Dx;
	mexType accum_y=0.0;
	mexType AvgInvRhoJ;
	mexType Dy;
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
			#if defined(_PR_PML)	
				index=Ind_MaterialMap(i,j,k);
				AvgInvRhoI=ELD(InvRhoMatH,ELD(MaterialMap,index));
				//In the PML
				// For coeffs. for V_x
				if (i<N1-1 && j <N2-1 && k<N3-1)
				{
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
		
				 }
			#endif
			}
		else
			{

	#if defined(_PR_MAIN) 
				index=Ind_MaterialMap(i,j,k);
				AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j,k))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				
				Dx=CA*(EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k))-
						CB*(EL(Sigma_xx,i+2,j,k)-EL(Sigma_xx,i-1,j,k));

				Dx+=CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k))-
						CB*(EL(Sigma_xy,i,j+1,k)-EL(Sigma_xy,i,j-2,k));

				Dx+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1))-
						CB*(EL(Sigma_xz,i,j,k+1)-EL(Sigma_xz,i,j,k-2));

				EL(Vx,i,j,k)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j,k);

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

				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				AvgInvRhoK=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j,k+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				Dz=CA*(EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k))-
						CB*( EL(Sigma_zz,i,j,k+2)-EL(Sigma_zz,i,j,k-1));

				Dz+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k))-
					CB*(EL(Sigma_xz,i+1,j,k)-EL(Sigma_xz,i-2,j,k));

				Dz+=CA*( EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k))-
					CB*(EL(Sigma_yz,i,j+1,k)-EL(Sigma_yz,i,j-2,k));

				EL(Vz,i,j,k)+=DT*AvgInvRhoK*Dz;
				accum_z+=EL(Vz,i,j,k);
			
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
							EL(Vx,i,j,k)+=value*ELD(Ox,index);
							EL(Vy,i,j,k)+=value*ELD(Oy,index);
							EL(Vz,i,j,k)+=value*ELD(Oz,index);
							
						}
						else
						{
							EL(Vx,i,j,k)=value*ELD(Ox,index);
							EL(Vy,i,j,k)=value*ELD(Oy,index);
							EL(Vz,i,j,k)=value*ELD(Oz,index);
							
						}
  					}		
				}
		#endif
			}
		}
	#if defined(_PR_MAIN) 
	if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0 && IsOnPML_K(k)==0 && nStep>=SensorStart*SensorSubSampling)
	{
		if (ZoneCount>1)
		{
			accum_x/=ZoneCount;
			accum_y/=ZoneCount;
			accum_z/=ZoneCount;
		}
		CurZone=0;
		index=IndN1N2N3(i,j,k,0);
		index2=N1*N2*N3;
		if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
		{
			if (IS_Vx_SELECTED(SelMapsRMSPeak))
				ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)+=accum_x*accum_x;
			if (IS_Vy_SELECTED(SelMapsRMSPeak))
				ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)+=accum_y*accum_y;
			if (IS_Vz_SELECTED(SelMapsRMSPeak))
				ELD(SqrAcc,index+index2*IndexRMSPeak_Vz)+=accum_z*accum_z;
			
		}
		if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
				index+=index2*NumberSelRMSPeakMaps;
		if (SelRMSorPeak & SEL_PEAK)
		{
			if (IS_Vx_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vx)=accum_x > ELD(SqrAcc,index+index2*IndexRMSPeak_Vx) ? accum_x : ELD(SqrAcc,index+index2*IndexRMSPeak_Vx);
			if (IS_Vy_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vy)=accum_y > ELD(SqrAcc,index+index2*IndexRMSPeak_Vy) ? accum_y : ELD(SqrAcc,index+index2*IndexRMSPeak_Vy);
			if (IS_Vz_SELECTED(SelMapsRMSPeak))
					ELD(SqrAcc,index+index2*IndexRMSPeak_Vz)=accum_z > ELD(SqrAcc,index+index2*IndexRMSPeak_Vz) ? accum_z : ELD(SqrAcc,index+index2*IndexRMSPeak_Vz);
			
		}

	}
	#endif

		
		
