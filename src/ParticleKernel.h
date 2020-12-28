#ifdef USE_2ND_ORDER_EDGES
	interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
    unsigned int index,index2,CurZone,source;
	mexType AvgInvRhoI,AvgInvRhoJ,AvgInvRhoK,Dx,Diff,value,accum_x=0.0,accum_y=0.0,accum_z=0.0;

	for (   CurZone=0;CurZone<ZoneCount;CurZone++)
		if (i<N1 && j<N2 && k<N3)
			{

		  if (IsOnPML_I(i)==1 || IsOnPML_J(j)==1 || IsOnPML_K(k)==1)
			{
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
				 }

				 //We add now for Vx, Vy, Vz

				if ( j <N2 && k < N3)
				{
				   index=Ind_V_x(i,j,k);
				   index2=Ind_V_x_x(i,j,k);
				   ELD(Vx,index)=ELD(V_x_x,index2)+ELD(V_y_x,index2)+ELD(V_z_x,index2);
				}
				if (i<N1 && k < N3 )
				{
					index=Ind_V_y(i,j,k);
					index2=Ind_V_y_y(i,j,k);
					ELD(Vy,index)=ELD(V_x_y,index2)+ELD(V_y_y,index2)+ELD(V_z_y,index2);
				}
				if (i<N1 && j < N2 )
				{
					index=Ind_V_z(i,j,k);
					index2=Ind_V_z_z(i,j,k);
					ELD(Vz,index)=ELD(V_x_z,index2)+ELD(V_y_z,index2)+ELD(V_z_z,index2);
				}
			}
			else
			{
				index=Ind_MaterialMap(i,j,k);

		#ifdef USE_2ND_ORDER_EDGES
						unsigned int m1=ELD(MiuMatOverH,EL(MaterialMap,i,j,k));
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i+2,j,k))==0.0,m1== 0.0)
		            interfaceX=interfaceX|frontLinep1;
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i+1,j,k))==0.0,m1 ==0.0)
		            interfaceX=interfaceX|frontLine;
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i-1,j,k))==0.0,m1 ==0.0)
		            interfaceX=interfaceX|backLine;
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j+2,k))==0.0,m1== 0.0)
		            interfaceY=interfaceY|frontLinep1;
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j+1,k))==0.0,m1 ==0.0)
		            interfaceY=interfaceY|frontLine;
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j-1,k))==0.0,m1 ==0.0)
		            interfaceY=interfaceY|backLine;

		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k+2))==0.0 , m1== 0.0)
		            interfaceZ=interfaceZ|frontLinep1;
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k+1))==0.0, m1 ==0.0)
		            interfaceZ=interfaceZ|frontLine;
		        if XOR(ELD(MiuMatOverH,EL(MaterialMap,i,j,k-1))==0.0, m1 ==0.0)
		            interfaceZ=interfaceZ|backLine;
		#endif

		    AvgInvRhoI=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i+1,j,k))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				AvgInvRhoJ=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j+1,k))+ELD(InvRhoMatH,ELD(MaterialMap,index)));
				AvgInvRhoK=0.5*(ELD(InvRhoMatH,EL(MaterialMap,i,j,k+1))+ELD(InvRhoMatH,ELD(MaterialMap,index)));


	      if REQUIRES_2ND_ORDER_P(X)
	          Dx=EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k);
	      else
	          Dx=CA*(EL(Sigma_xx,i+1,j,k)-EL(Sigma_xx,i,j,k))-
	             CB*(EL(Sigma_xx,i+2,j,k)-EL(Sigma_xx,i-1,j,k));

	      if REQUIRES_2ND_ORDER_P(Y)
	          Dx+=EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k);
	      else
	          Dx+=CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i,j-1,k))-
	              CB*(EL(Sigma_xy,i,j+1,k)-EL(Sigma_xy,i,j-2,k));

	      if REQUIRES_2ND_ORDER_P(Z)
	          Dx+=EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1);
	      else
	          Dx+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i,j,k-1))-
	              CB*(EL(Sigma_xz,i,j,k+1)-EL(Sigma_xz,i,j,k-2));


	      EL(Vx,i,j,k)+=DT*AvgInvRhoI*Dx;
				accum_x+=EL(Vx,i,j,k);
	      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if REQUIRES_2ND_ORDER_P(Y)
            Dx=EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k);
        else
            Dx=CA*(EL(Sigma_yy,i,j+1,k)-EL(Sigma_yy,i,j,k) )-
                CB*(EL(Sigma_yy,i,j+2,k)-EL(Sigma_yy,i,j-1,k));

        if REQUIRES_2ND_ORDER_P(X)
            Dx+=EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k);
        else
            Dx+=CA*(EL(Sigma_xy,i,j,k)-EL(Sigma_xy,i-1,j,k))-
                CB*(EL(Sigma_xy,i+1,j,k)-EL(Sigma_xy,i-2,j,k));

        if REQUIRES_2ND_ORDER_P(Z)
            Dx+=EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1);
        else
            Dx+=CA*( EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j,k-1))-
            CB*(EL(Sigma_yz,i,j,k+1)-EL(Sigma_yz,i,j,k-2));

        EL(Vy,i,j,k)+=DT*AvgInvRhoJ*Dx;
				accum_y+=EL(Vy,i,j,k);
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        if REQUIRES_2ND_ORDER_P(Z)
            Dx=EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k);
        else
            Dx=CA*(EL(Sigma_zz,i,j,k+1)-EL(Sigma_zz,i,j,k))-
                CB*( EL(Sigma_zz,i,j,k+2)-EL(Sigma_zz,i,j,k-1));

        if REQUIRES_2ND_ORDER_P(X)
            Dx+=EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k);
        else
            Dx+=CA*(EL(Sigma_xz,i,j,k)-EL(Sigma_xz,i-1,j,k))-
            CB*(EL(Sigma_xz,i+1,j,k)-EL(Sigma_xz,i-2,j,k));

        if REQUIRES_2ND_ORDER_P(Y)
            Dx+=EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k);
        else
            Dx+=CA*( EL(Sigma_yz,i,j,k)-EL(Sigma_yz,i,j-1,k))-
            CB*(EL(Sigma_yz,i,j+1,k)-EL(Sigma_yz,i,j-2,k));

        EL(Vz,i,j,k)+=DT*AvgInvRhoK*Dx;
				accum_z+=EL(Vz,i,j,k);
			}

  		if (nStep < LengthSource)
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

		}
		if (i<N1 && j <N2 && k < N3 )
	  {
	    accum_x/=ZoneCount;
	    accum_y/=ZoneCount;
	    accum_z/=ZoneCount;
	    CurZone=0;
	    index=Ind_Sigma_xx(i,j,k);
	    ELD(SqrAcc,index)+=accum_x*accum_x  +  accum_y*accum_y  +  accum_z*accum_z;
	  }
