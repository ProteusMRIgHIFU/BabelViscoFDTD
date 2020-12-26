#if defined(CUDA)
__global__ void StressKernel(InputDataKernel *p,unsigned int nStep)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
__kernel void StressKernel(
__global mexType *V_x_x_pr,
__global mexType *V_y_x_pr,
__global mexType *V_z_x_pr,
__global mexType *V_x_y_pr,
__global mexType *V_y_y_pr,
__global mexType *V_z_y_pr,
__global mexType *V_x_z_pr,
__global mexType *V_y_z_pr,
__global mexType *V_z_z_pr,
__global mexType *Vx_pr,
__global mexType *Vy_pr,
__global mexType *Vz_pr,
__global mexType *Rxx_pr,
__global mexType *Ryy_pr,
__global mexType *Rzz_pr,
__global mexType *Rxy_pr,
__global mexType *Rxz_pr,
__global mexType *Ryz_pr,
__global mexType *Sigma_x_xx_pr,
__global mexType *Sigma_y_xx_pr,
__global mexType *Sigma_z_xx_pr,
__global mexType *Sigma_x_yy_pr,
__global mexType *Sigma_y_yy_pr,
__global mexType *Sigma_z_yy_pr,
__global mexType *Sigma_x_zz_pr,
__global mexType *Sigma_y_zz_pr,
__global mexType *Sigma_z_zz_pr,
__global mexType *Sigma_x_xy_pr,
__global mexType *Sigma_y_xy_pr,
__global mexType *Sigma_x_xz_pr,
__global mexType *Sigma_z_xz_pr,
__global mexType *Sigma_y_yz_pr,
__global mexType *Sigma_z_yz_pr,
__global mexType *Sigma_xy_pr,
__global mexType *Sigma_xz_pr,
__global mexType *Sigma_yz_pr,
__global mexType *Sigma_xx_pr,
__global mexType *Sigma_yy_pr,
__global mexType *Sigma_zz_pr,
__global mexType *Snapshots_pr,
__global mexType *SourceFunctions_pr,
__global mexType * LambdaMiuMatOverH_pr,
__global mexType * LambdaMatOverH_pr,
__global mexType * MiuMatOverH_pr,
__global mexType * TauLong_pr,
__global mexType * OneOverTauSigma_pr,
__global mexType * TauShear_pr,
__global mexType * InvRhoMatH_pr,
__global mexType * SqrAcc_pr,
__global unsigned int * MaterialMap_pr,
__global unsigned int * SourceMap_pr
	, unsigned int nStep)
{
  const unsigned int i = get_global_id(0);
  const unsigned int j = get_global_id(1);
  const unsigned int k = get_global_id(2);
#endif

    if (i>N1 || j >N2  || k>N3)
		return;

    mexType Diff,value,Dx,Dy,Dz,m1,m2,m3,m4,RigidityXY=0.0,RigidityXZ=0.0,
        RigidityYZ=0.0,LambdaMiu,Miu,LambdaMiuComp,MiuComp,
        dFirstPart,OneOverTauSigma,dFirstPartForR,NextR,
            TauShearXY=0.0,TauShearXZ=0.0,TauShearYZ=0.0;
#ifdef USE_2ND_ORDER_EDGES
    interface_t interfaceZ=inside, interfaceY=inside, interfaceX=inside;
#endif
   	unsigned int index,index2,MaterialID,CurZone;
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
  		ELD(Rxx,index)=NextR;

  		NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Ryy,index) - dFirstPartForR + MiuComp*(Dx+Dz))
  		      /(1+DT*0.5*OneOverTauSigma);
  		ELD(Sigma_yy,index)+=	DT*(dFirstPart - Miu*(Dx+Dz) + 0.5*(ELD(Ryy,index) + NextR));
  		ELD(Ryy,index)=NextR;

  		NextR=( (1-DT*0.5*OneOverTauSigma)*ELD(Rzz,index) - dFirstPartForR +MiuComp*(Dx+Dy))
  		      /(1+DT*0.5*OneOverTauSigma);
  		ELD(Sigma_zz,index)+=	DT*(dFirstPart - Miu*(Dx+Dy) + 0.5*(ELD(Rzz,index) + NextR));
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
  			ELD(Ryz,index)=NextR;

  		}
          else
              ELD(Ryz,index)=0.0;

  	}

  }
}

#if defined(CUDA)
__global__ void ParticleKernel(InputDataKernel * p, unsigned int nStep, int CurrSnap,unsigned int NextSnap,unsigned int TypeSource)
{
	  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
__kernel void ParticleKernel(
__global mexType *V_x_x_pr,
__global mexType *V_y_x_pr,
__global mexType *V_z_x_pr,
__global mexType *V_x_y_pr,
__global mexType *V_y_y_pr,
__global mexType *V_z_y_pr,
__global mexType *V_x_z_pr,
__global mexType *V_y_z_pr,
__global mexType *V_z_z_pr,
__global mexType *Vx_pr,
__global mexType *Vy_pr,
__global mexType *Vz_pr,
__global mexType *Rxx_pr,
__global mexType *Ryy_pr,
__global mexType *Rzz_pr,
__global mexType *Rxy_pr,
__global mexType *Rxz_pr,
__global mexType *Ryz_pr,
__global mexType *Sigma_x_xx_pr,
__global mexType *Sigma_y_xx_pr,
__global mexType *Sigma_z_xx_pr,
__global mexType *Sigma_x_yy_pr,
__global mexType *Sigma_y_yy_pr,
__global mexType *Sigma_z_yy_pr,
__global mexType *Sigma_x_zz_pr,
__global mexType *Sigma_y_zz_pr,
__global mexType *Sigma_z_zz_pr,
__global mexType *Sigma_x_xy_pr,
__global mexType *Sigma_y_xy_pr,
__global mexType *Sigma_x_xz_pr,
__global mexType *Sigma_z_xz_pr,
__global mexType *Sigma_y_yz_pr,
__global mexType *Sigma_z_yz_pr,
__global mexType *Sigma_xy_pr,
__global mexType *Sigma_xz_pr,
__global mexType *Sigma_yz_pr,
__global mexType *Sigma_xx_pr,
__global mexType *Sigma_yy_pr,
__global mexType *Sigma_zz_pr,
__global mexType *Snapshots_pr,
__global mexType *SourceFunctions_pr,
__global mexType * LambdaMiuMatOverH_pr,
__global mexType * LambdaMatOverH_pr,
__global mexType * MiuMatOverH_pr,
__global mexType * TauLong_pr,
__global mexType * OneOverTauSigma_pr,
__global mexType * TauShear_pr,
__global mexType * InvRhoMatH_pr,
__global mexType * SqrAcc_pr,
__global unsigned int * MaterialMap_pr,
__global unsigned int * SourceMap_pr
	, unsigned int nStep,
	int CurrSnap,
	unsigned int NextSnap, 
	unsigned int TypeSource)
{
		const unsigned int i = get_global_id(0);
	  const unsigned int j = get_global_id(1);
	  const unsigned int k = get_global_id(2);
#endif
    if (i>N1 || j >N2  || k>N3)
		return;


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

  		index=Ind_Sigma_xx(i,j,k);
  		if (nStep < LengthSource)
  		{
  			//NOW we add the sources, we assume a "soft" source ,maybe later we can add a parameter to decide if the source is "hard"
  			source=ELD(SourceMap,IndN1N2N3(i,j,k,0));
  			if (source>0)
  			{
  			  source--; //need to use C index
  			  value=ELD(SourceFunctions,nStep*NumberSources+source);
                if (TypeSource==0)
                {
                    EL(Vx,i,j,k)+=value*Ox;
                    EL(Vy,i,j,k)+=value*Oy;
                    EL(Vz,i,j,k)+=value*Oz;
                }
                else
                {
                    EL(Vx,i,j,k)=value*Ox;
                    EL(Vy,i,j,k)=value*Oy;
                    EL(Vz,i,j,k)=value*Oz;
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
}


#if defined(CUDA)
__global__ void SnapShot(unsigned int SelK,mexType * Snapshots_pr,mexType * Sigma_xx_pr,mexType * Sigma_yy_pr,mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
#else
__kernel void SnapShot(unsigned int SelK,__global mexType * Snapshots_pr,__global mexType * Sigma_xx_pr,__global mexType * Sigma_yy_pr,__global mexType * Sigma_zz_pr,unsigned int CurrSnap)
{
  const unsigned int i = get_global_id(0);
  const unsigned int j = get_global_id(1);
#endif

    if (i>=N1 || j >=N2)
		return;
	mexType accum=0.0;
	for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
		{
			unsigned int index=Ind_Sigma_xx(i,j,SelK);
			accum+=(Sigma_xx_pr[index]+Sigma_yy_pr[index]+Sigma_zz_pr[index])/3.0;

		}

		Snapshots_pr[IndN1N2Snap(i,j)+CurrSnap*N1*N2]=accum/ZoneCount;
}

#if defined(CUDA)
__global__ void SensorsKernel(mexType * SensorOutput_pr,mexType * Sigma_xx_pr,mexType * Sigma_yy_pr,mexType * Sigma_zz_pr,unsigned int * IndexSensorMap_pr, unsigned int nStep,unsigned int NumberSensors)
{
	unsigned int j =blockIdx.x * blockDim.x + threadIdx.x;
#else
__kernel void SensorsKernel(__global mexType * SensorOutput_pr,__global mexType * Sigma_xx_pr,__global mexType * Sigma_yy_pr,__global mexType * Sigma_zz_pr,__global unsigned int * IndexSensorMap_pr, unsigned int nStep,unsigned int NumberSensors)
{
	unsigned int j =get_global_id(0);
#endif
	if (j>=	NumberSensors)
		return;

	unsigned int index=nStep*NumberSensors+j;

	mexType accum=0.0;
	for (unsigned int CurZone=0;CurZone<ZoneCount;CurZone++)
		{
			unsigned int index2=IndexSensorMap_pr[j]-1 + Ind_Sigma_xx(0,0,0);
			accum+=(Sigma_xx_pr[index2]+Sigma_yy_pr[index2]+Sigma_zz_pr[index2])/3.0;

		}
	SensorOutput_pr[index]=accum/ZoneCount;

}
