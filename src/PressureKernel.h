    unsigned int index,index2,CurZone;
	mexType accum_p=0, Dx,Dy,Dz;
    if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0 && IsOnPML_K(k)==0)
    {
        //we can just avoid calculating the pressure field in the PML as it is not usesful anyway

        //We will use the particle displacement to estimate the acoustic pressure, and using the conservation of mass formula
        // partial(p)/partial(t) = \rho c^2 div(V)
        //it is important to mention that the Python function will need still to multiply the result for the maps of speed of sound and density.

	    for (   CurZone=0;CurZone<ZoneCount;CurZone++)
        {
            Dx=EL(Vx,i,j,k)-EL(Vx,i-1,j,k);
            Dy=EL(Vy,i,j,k)-EL(Vy,i,j-1,k);
            Dz=EL(Vz,i,j,k)-EL(Vz,i,j,k-1);
            EL(Pressure,i,j,k)+=DT*(Dx+Dy+Dz);
            accum_p+=EL(Pressure,i,j,k);

        }
        if (ZoneCount>1)
				accum_p/=ZoneCount;
        CurZone=0;
        index=IndN1N2N3(i,j,k,0);
        index2=N1*N2*N3;
        if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
            ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;

        if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
                index+=index2*NumberSelRMSPeakMaps;
        if (SelRMSorPeak & SEL_PEAK)
            ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p :ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);

    }
