    //Only use these for the PML

	LOCAL_CALLOC(Vx,GET_NUMBER_ELEMS(Vx_res));
  LOCAL_CALLOC(Vy,GET_NUMBER_ELEMS(Vy_res));
  LOCAL_CALLOC(Vz,GET_NUMBER_ELEMS(Vz_res));
  LOCAL_CALLOC(Sigma_xx,GET_NUMBER_ELEMS(Sigma_xx_res));
  LOCAL_CALLOC(Sigma_yy,GET_NUMBER_ELEMS(Sigma_yy_res));
  LOCAL_CALLOC(Sigma_zz,GET_NUMBER_ELEMS(Sigma_zz_res));
  LOCAL_CALLOC(Sigma_xy,GET_NUMBER_ELEMS(Sigma_xy_res));
  LOCAL_CALLOC(Sigma_xz,GET_NUMBER_ELEMS(Sigma_xz_res));
  LOCAL_CALLOC(Sigma_yz,GET_NUMBER_ELEMS(Sigma_yz_res));
  LOCAL_CALLOC(Pressure,GET_NUMBER_ELEMS(Pressure_res));
  
	LOCAL_CALLOC(V_x_x,INHOST(SizePMLxp1));
	LOCAL_CALLOC(V_y_x,INHOST(SizePMLxp1));
	LOCAL_CALLOC(V_z_x,INHOST(SizePMLxp1));
	LOCAL_CALLOC(V_x_y,INHOST(SizePMLyp1));
	LOCAL_CALLOC(V_y_y,INHOST(SizePMLyp1));
	LOCAL_CALLOC(V_z_y,INHOST(SizePMLyp1));
	LOCAL_CALLOC(V_x_z,INHOST(SizePMLzp1));
	LOCAL_CALLOC(V_y_z,INHOST(SizePMLzp1));
	LOCAL_CALLOC(V_z_z,INHOST(SizePMLzp1));

	LOCAL_CALLOC(Sigma_x_xx,INHOST(SizePML));
	LOCAL_CALLOC(Sigma_y_xx,INHOST(SizePML));
	LOCAL_CALLOC(Sigma_z_xx,INHOST(SizePML));

	LOCAL_CALLOC(Sigma_x_yy,INHOST(SizePML));
	LOCAL_CALLOC(Sigma_y_yy,INHOST(SizePML));
	LOCAL_CALLOC(Sigma_z_yy,INHOST(SizePML));

	LOCAL_CALLOC(Sigma_x_zz,INHOST(SizePML));
	LOCAL_CALLOC(Sigma_y_zz,INHOST(SizePML));
	LOCAL_CALLOC(Sigma_z_zz,INHOST(SizePML));

	LOCAL_CALLOC(Sigma_x_xy,INHOST(SizePMLxp1yp1zp1));
	LOCAL_CALLOC(Sigma_y_xy,INHOST(SizePMLxp1yp1zp1));

	LOCAL_CALLOC(Sigma_x_xz,INHOST(SizePMLxp1yp1zp1));
	LOCAL_CALLOC(Sigma_z_xz,INHOST(SizePMLxp1yp1zp1));

	LOCAL_CALLOC(Sigma_y_yz,INHOST(SizePMLxp1yp1zp1));
	LOCAL_CALLOC(Sigma_z_yz,INHOST(SizePMLxp1yp1zp1));

	LOCAL_CALLOC(Rxx,GET_NUMBER_ELEMS(Sigma_xx_res));
	LOCAL_CALLOC(Ryy,GET_NUMBER_ELEMS(Sigma_xx_res));
	LOCAL_CALLOC(Rzz,GET_NUMBER_ELEMS(Sigma_xx_res));
	LOCAL_CALLOC(Rxy,GET_NUMBER_ELEMS(Sigma_xy_res));
	LOCAL_CALLOC(Rxz,GET_NUMBER_ELEMS(Sigma_xy_res));
	LOCAL_CALLOC(Ryz,GET_NUMBER_ELEMS(Sigma_xy_res));



   #if  defined(USE_OPENMP)
   int ntx= omp_get_max_threads();

  #ifdef MATLAB_MEX
  	PRINTF("Max number of threads =%i\n",ntx);
  #else
  	PySys_WriteStdout("Max number of threads =%i\n",ntx);
  #endif
#endif
#if  defined(USE_OPENMP)
    omp_set_num_threads(ntx);
#endif

	//Lets roll the time
	int ii,jj,kk,CurZone;

  #ifdef CHECK_FOR_NANs
    int bNanDetected =0;
    int indexforNaN=0;
  #endif

	// unsigned int MaxIndex = INHOST(InternalIndexCount)> INHOST(EdgeIndexCount) ? INHOST(InternalIndexCount):INHOST(EdgeIndexCount);


	// fprintf(FDEBUG,"TotalIndexCount,InternalIndexCount,EdgeIndexCount,ZoneCount =%i,%i,%i,%i\n",INHOST(TotalIndexCount),
	// 			INHOST(InternalIndexCount),INHOST(EdgeIndexCount),INHOST(ZoneCount));

	//fflush( FDEBUG );

	for (unsigned int nStep=0;nStep<INHOST(TimeSteps);nStep++)
	{
		if ((nStep % 200)==0)
		{
			PRINTF("Doing step %i of %i\n",nStep,INHOST(TimeSteps));
			fprintf(FDEBUG,"Doing step %i of %i\n",nStep,INHOST(TimeSteps));
			fflush(FDEBUG);
		}
		//********************************
		//First we do the constrains tensors
		//********************************
		#pragma omp parallel for private(jj,ii,CurZone)
		for(kk=0; kk<( int)(N3+1); kk++)
		{
			unsigned int k= (unsigned int)kk;
			for(jj=0; jj<( int)N2+1; jj++)
			{
				unsigned int j= (unsigned int)jj;
				for(ii=0; ii<( int)N1+1; ii++)
				{
					unsigned int i= (unsigned int)ii;

					#include "StressKernel.h"

				}
			}
		}
		#ifdef CHECK_FOR_NANs
			#pragma omp parallel for private(jj,ii,CurZone)
			for(kk=0; kk<( int)N3; kk++)
			{
				unsigned int k= (unsigned int)kk;
				for(jj=0; jj<( int)N2; jj++)
				{
					unsigned int j= (unsigned int)jj;
					for(ii=0; ii<( int)N1; ii++)
					{
						unsigned int i= (unsigned int)ii;
						for ( CurZone=0;CurZone<( int)INHOST(ZoneCount);CurZone++)
						{
							indexforNaN=Ind_Sigma_xx(i,j,k);
							if (isnan(ELD(Sigma_xx,indexforNaN)) ||
								isnan(ELD(Sigma_yy,indexforNaN)) ||
								isnan(ELD(Sigma_zz,indexforNaN)) ||
								isnan(ELD(Rxx,indexforNaN)) ||
								isnan(ELD(Ryy,indexforNaN)) ||
								isnan(ELD(Rzz,indexforNaN)))
								{
								bNanDetected=1;
												break;
							}
						}
					}
				}
			}

			if (bNanDetected==1)
			{
					PRINTF("***** FOUND NAN AFTER Stress Kernel at step %i\n",nStep);
				break;
			}
			#endif
			//********************************
			//Then we do the particle displacements
			//********************************
			#pragma omp parallel for private(jj,ii,CurZone)
			for(kk=0; kk<( int)N3+1; kk++)
			{
				unsigned int k= (unsigned int)kk;
				for(jj=0; jj<( int)N2+1; jj++)
				{
					unsigned int j= (unsigned int)jj;
					for(ii=0; ii<( int)N1+1; ii++)
					{
						unsigned int i= (unsigned int)ii;
						#include "ParticleKernel.h"

					}
				}
			}

			#ifdef CHECK_FOR_NANs
			#pragma omp parallel for private(jj,ii,CurZone)
			for(kk=0; kk<( int)N3; kk++)
			{
				unsigned int k= (unsigned int)kk;
				for(jj=0; jj<( int)N2; jj++)
				{
					unsigned int j= (unsigned int)jj;
					for(ii=0; ii<( int)N1; ii++)
					{
						unsigned int i= (unsigned int)ii;
						for ( CurZone=0;CurZone<( int)INHOST(ZoneCount);CurZone++)
						{
							if (isnan(EL(Vx,i,j,k)))
							{
								bNanDetected=1;
													break;
							}else if (isnan(EL(Vy,i,j,k)))
							{
								bNanDetected=1;
													break;
							}else if (isnan(EL(Vz,i,j,k)))
							{
								bNanDetected=1;
													break;
							}
						}
					}
				}
			}
			if (bNanDetected==1)
			{
						PRINTF("***** FOUND NAN AFTER Particle Kernel in step %i\n",nStep);
				break;
			}
		#endif

		if (INHOST(CurrSnap)>=0 && INHOST(CurrSnap) <NumberSnapshots)
			if(nStep==SnapshotsPos_pr[INHOST(CurrSnap)]-1 )
			{

				#pragma omp parallel for private(jj,ii,CurZone)
				for(jj=0; jj<( int)N2; jj++)
				{
					unsigned int j= (unsigned int)jj;
					for(ii=0; ii<( int)N1; ii++)
					{
						unsigned int i= (unsigned int)ii;
						mexType accum=0.0;
						for ( CurZone=0;CurZone<( int)INHOST(ZoneCount);CurZone++)
						{
							unsigned int index=Ind_Sigma_xx(i,j,N3/2);
							accum+=(Sigma_xx_pr[index]+Sigma_yy_pr[index]+Sigma_zz_pr[index])/3.0;
						}
						Snapshots_pr[IndN1N2Snap(i,j)+INHOST(CurrSnap)*N1*N2]=accum/INHOST(ZoneCount);
					}
				}
				INHOST(CurrSnap)++;
			}

		//Finally, the sensors
		if (((nStep % INHOST(SensorSubSampling))==0) && ((nStep / INHOST(SensorSubSampling))>=INHOST(SensorStart)))
		{
			int sj;
			#pragma omp parallel for private(CurZone)
			for(sj=0; sj<( int)INHOST(NumberSensors); sj++)
			{
					#include"SensorsKernel.h"
			}
		}

	}
	//DONE, it looks easy but it took a couple weeks taking it to this simple implementation


	fprintf(FDEBUG,"after main loop\n");
	fflush(FDEBUG);

	{
	#pragma omp parallel for private(jj,ii,CurZone)
	for(kk=0; kk<( int)N3; kk++)
	{
		unsigned int k= (unsigned int)kk;
		for(jj=0; jj<( int)N2; jj++)
		{
			unsigned int j= (unsigned int)jj;
			for(ii=0; ii<( int)N1; ii++)
			{
				unsigned int i= (unsigned int)ii;
				ASSIGN_RES(Vx);
				ASSIGN_RES(Vy);
				ASSIGN_RES(Vz);
				ASSIGN_RES(Sigma_xx);
				ASSIGN_RES(Sigma_yy);
				ASSIGN_RES(Sigma_zz);
				ASSIGN_RES(Sigma_xy);
				ASSIGN_RES(Sigma_xz);
				ASSIGN_RES(Sigma_yz);
				ASSIGN_RES(Pressure);
			}
		}
	}
	}
fprintf(FDEBUG,"After finishing assigning results\n");
fflush(FDEBUG);
free(Vx_pr);
free(Vy_pr);
free(Vz_pr);
free(Pressure_pr);
free(Sigma_xx_pr);
free(Sigma_yy_pr);
free(Sigma_zz_pr);
free(Sigma_xy_pr);
free(Sigma_xz_pr);
free(Sigma_yz_pr);


free(V_x_x_pr);
free(V_y_x_pr);
free(V_z_x_pr);
free(V_x_y_pr);
free(V_y_y_pr);
free(V_z_y_pr);
free(V_x_z_pr);
free(V_y_z_pr);
free(V_z_z_pr);
free(Sigma_x_xx_pr);
free(Sigma_y_xx_pr);
free(Sigma_z_xx_pr);
free(Sigma_x_yy_pr);
free(Sigma_y_yy_pr);
free(Sigma_z_yy_pr);
free(Sigma_x_zz_pr);
free(Sigma_y_zz_pr);
free(Sigma_z_zz_pr);
free(Sigma_x_xy_pr);
free(Sigma_y_xy_pr);
free(Sigma_x_xz_pr);
free(Sigma_z_xz_pr);
free(Sigma_y_yz_pr);
free(Sigma_z_yz_pr);
free(Rxx_pr);
free(Ryy_pr);
free(Rzz_pr);
free(Rxy_pr);
free(Rxz_pr);
free(Ryz_pr);

fprintf(FDEBUG,"After releasing all memory\n");
fflush(FDEBUG);
