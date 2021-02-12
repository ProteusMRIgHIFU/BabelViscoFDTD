    unsigned int index,index2,CurZone,source;
	mexType accum_p=0, Dv;
    if (IsOnPML_I(i)==0 && IsOnPML_J(j)==0 && IsOnPML_K(k)==0)
    {
        //we can just avoid calculating the pressure field in the PML as it is not usesful anyway

        //We will use the particle displacement to estimate the acoustic pressure, 
        //it is important to mention that the Python function will need still to multiply the result for the maps of speed of sound and density.
    
	    for (   CurZone=0;CurZone<ZoneCount;CurZone++) 
        {
            if REQUIRES_2ND_ORDER_P(X)
                Dv=EL(Vx,i+1,j,k)-EL(Vx,i,j,k);
            else
                Dv=CA*(EL(Vx,i+1,j,k)-EL(Vx,i,j,k))-
                    CB*(EL(Vx,i+2,j,k)-EL(Vx,i-1,j,k));

            if REQUIRES_2ND_ORDER_P(X)
                Dv+=EL(Vy,i,j+1,k)-EL(Vy,i,j,k);
            else
                Dv+=CA*(EL(Vy,i,j+1,k)-EL(Vy,i,j,k))-
                    CB*(EL(Vy,i,j+2,k)-EL(Vy,i,j-1,k));

            if REQUIRES_2ND_ORDER_P(X)
                Dv+=EL(Vz,i,j,k+1)-EL(Vz,i,j,k);
            else
                Dv+=CA*(EL(Vz,i,j,k+1)-EL(Vz,i,j,k))-
                    CB*(EL(Vz,i,j,k+2)-EL(Vz,i,j,k-1));
            accum_p+=-Dv;

        }
        accum_p/=ZoneCount;
        index=IndN1N2N3(i,j,k,0);
		index2=N1*N2*N3;
        if ((SelRMSorPeak & SEL_RMS) ) //RMS was selected, and it is always at the location 0 of dim 5
		    ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)+=accum_p*accum_p;
        
        if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK) ) //If both PEAK and RMS were selected we save in the far part of the array
			index+=index2*NumberSelRMSPeakMaps;
		if (SelRMSorPeak & SEL_PEAK)
            ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure)=accum_p > ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure) ? accum_p : ELD(SqrAcc,index+index2*IndexRMSPeak_Pressure);
    }
    

        
