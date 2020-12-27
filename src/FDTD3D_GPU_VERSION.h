int NumberAlloc=0;

#if defined(CUDA)
    cudaDeviceProp deviceProperties;
    int *sm13DeviceList; // Device handles for each CUDA >=1.3 capable device
    int deviceCount = 0; // Total number of devices
    int sm13DeviceCount = 0; // Number of devices that are CUDA >=1.3 capable

    //---------------------------------------------------------------//
    // Find all CUDA >=1.3-capable devices                           //
    //---------------------------------------------------------------//

    // Check for number of devices total
    PRINTF("before cudaGetDeviceCount \n");
    cudaGetDeviceCount(&deviceCount);
    PRINTF("after cudaGetDeviceCount \n");

    if(deviceCount == 0)
    {
        ERROR_STRING("There are no CUDA devices.\n");
    }
    else
    {
        PRINTF("There %s %i device%s.\n", deviceCount > 1 ? "are" : "is",
                                            deviceCount,
                                            deviceCount > 1 ? "s" : "");
    }

    // Make list of devices
    sm13DeviceList = (int*) calloc(deviceCount, sizeof(int));

    for(int deviceID = 0; deviceID < deviceCount; deviceID++)
    {
        // Check device properties
        if(cudaGetDeviceProperties(&deviceProperties, deviceID) == cudaSuccess)
        {

            PRINTF("Found device [%d:%d]:\n", sm13DeviceCount, deviceID);
            PRINTF("  Name: %s\n", deviceProperties.name);
            PRINTF("  Compute capability: %i.%i\n", deviceProperties.major, deviceProperties.minor);
            PRINTF("  Total memory: %li bytes\n", deviceProperties.totalGlobalMem);
            PRINTF("  Threads per block: %i\n",deviceProperties.maxThreadsPerBlock);
            PRINTF("  Max block dimensions: %i x %i x %i\n", deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
            PRINTF("  Max grid size: %i x %i x %i\n", deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
            PRINTF("  Shared memory per block: %i bytes\n", (int) deviceProperties.sharedMemPerBlock);
            PRINTF("  Registers per block: %i\n", deviceProperties.regsPerBlock);
            // Check major/minor CUDA versions
            if(deviceProperties.major >=3 && strstr(deviceProperties.name, DefaultGPUDeviceName_pr) )
            {
               PRINTF("  Selecting device [%s] for calculations \n", deviceProperties.name);
                // Add device to 3-capable list
                sm13DeviceList[sm13DeviceCount] = deviceID;
                sm13DeviceCount++;
                break;
            }
        }
    }

    // Were any of the devices 1.3-capable?
    if(sm13DeviceCount == 0)
    {
        ERROR_STRING("There are no devices supporting CUDA or that matches selected device.\n");
    }

    mxcheckGPUErrors(cudaSetDevice(sm13DeviceList[0]));

    mxcheckGPUErrors(cudaDeviceSetCacheConfig (cudaFuncCachePreferL1));

#else
  cl_uint numPlatforms;
  int err;
  cl_device_id     device_id[10];     // compute device id
  cl_context       context;       // compute context
  cl_command_queue commands;      // compute command queue
  cl_program       program;       // compute program
  cl_kernel        StressKernel;       // compute kernel
  cl_kernel        ParticleKernel;       // compute kernel
  cl_kernel        SnapShot;       // compute kernel
  cl_kernel        SensorsKernel;       // compute kernel
  cl_char device_name[1024];

  // Find number of platforms
  mxcheckGPUErrors(clGetPlatformIDs(0, NULL, &numPlatforms));
  if (numPlatforms == 0)
  {
      ERROR_STRING("Found 0 OPENCL platforms!\n");
  }

  // Get all platforms
  cl_platform_id Platform[numPlatforms];
  mxcheckGPUErrors(clGetPlatformIDs(numPlatforms, Platform, NULL));

  // Secure a GPU

  unsigned int total_devices;
  int SelDevice=0;

// Create a compute context
  for (unsigned int icpu = 0; icpu < numPlatforms; icpu++)
  {
      err = clGetDeviceIDs(Platform[icpu], CL_DEVICE_TYPE_ALL, 10, device_id, &total_devices);
      if (err == CL_SUCCESS)
      {
          break;
      }
  }

  if (device_id[0] == NULL)
      ERROR_STRING("Found 0 OPENCL devices!\n");

  for (unsigned int icpu = 0; icpu <total_devices;icpu++)
  {
    mxcheckGPUErrors(output_device_info(device_id[icpu]));
    err = clGetDeviceInfo(device_id[icpu], CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    PRINTF("GPU device = %s\n",device_name);
    if (NULL!=strstr((char *)device_name,DefaultGPUDeviceName_pr))
    {
      PRINTF("Found %s device!\n",DefaultGPUDeviceName_pr);
      SelDevice=icpu;
    }
  }


  context = clCreateContext(0, 1, &device_id[SelDevice], NULL, NULL, &err);
  mxcheckGPUErrors(err);

  // Create a command queue

  commands = clCreateCommandQueue(context, device_id[SelDevice], 0, &err);
  mxcheckGPUErrors(err);

  sprintf(BUFFER_FOR_OPENCL_CODE,"\n#define mexType %s\n#define OPENCL\n",MEX_STR);
  char * indexingSource = load_file("_indexing.h");
  if (indexingSource==0)
  {
    ERROR_STRING("Unable to read _indexing.h file!!")
  }
  strncat(BUFFER_FOR_OPENCL_CODE,indexingSource,MAXP_BUFFER_OPENCL);
  strncat(BUFFER_FOR_OPENCL_CODE,"\n",MAXP_BUFFER_OPENCL);
  free(indexingSource);

#endif

//initilizing constant memory variables


InitSymbol(DT,mexType,G_FLOAT);
InitSymbol(N1,unsigned int,G_INT);
InitSymbol(N2,unsigned int,G_INT);
InitSymbol(N3,unsigned int,G_INT);
InitSymbol(Limit_I_low_PML,unsigned int,G_INT);
InitSymbol(Limit_J_low_PML,unsigned int,G_INT);
InitSymbol(Limit_K_low_PML,unsigned int,G_INT);
InitSymbol(Limit_I_up_PML,unsigned int,G_INT);
InitSymbol(Limit_J_up_PML,unsigned int,G_INT);
InitSymbol(Limit_K_up_PML,unsigned int,G_INT);
InitSymbol(SizeCorrI,unsigned int,G_INT);
InitSymbol(SizeCorrJ,unsigned int,G_INT);
InitSymbol(SizeCorrK,unsigned int,G_INT);
InitSymbol(PML_Thickness,unsigned int,G_INT);
InitSymbol(NumberSources,unsigned int,G_INT);
InitSymbol(LengthSource,unsigned int,G_INT);
InitSymbol(ZoneCount,unsigned int,G_INT);
InitSymbol(SizePMLxp1,unsigned int,G_INT);
InitSymbol(SizePMLyp1,unsigned int,G_INT);
InitSymbol(SizePMLzp1,unsigned int,G_INT);
InitSymbol(SizePML,unsigned int,G_INT);
InitSymbol(SizePMLxp1yp1zp1,unsigned int,G_INT);
InitSymbol(Ox,mexType,G_FLOAT);
InitSymbol(Oy,mexType,G_FLOAT);
InitSymbol(Oz,mexType,G_FLOAT);

//~
#ifdef CUDA //CUDA specifics

	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuInvDXDTpluspr,InvDXDTplus_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));
	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuDXDTminuspr,DXDTminus_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));
	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuInvDXDTplushppr,InvDXDTplushp_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));
	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuDXDTminushppr,DXDTminushp_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));

#else //OPENCL specifics
  InitSymbolArray(InvDXDTplus,G_FLOAT,PML_Thickness+1);
  InitSymbolArray(DXDTminus,G_FLOAT,PML_Thickness+1);
  InitSymbolArray(InvDXDTplushp,G_FLOAT,PML_Thickness+1);
  InitSymbolArray(DXDTminushp,G_FLOAT,PML_Thickness+1);

  char * KernelSource = load_file("_opencl_kernel.c");
  if (KernelSource==0)
  {
    ERROR_STRING("Unable to read _opencl_kernel.c file!!")
  }
  strncat(BUFFER_FOR_OPENCL_CODE,KernelSource,MAXP_BUFFER_OPENCL);
  strncat(BUFFER_FOR_OPENCL_CODE,"\n",MAXP_BUFFER_OPENCL);
  free(KernelSource);
  //PRINTF("%s",BUFFER_FOR_OPENCL_CODE);

  // program = clCreateProgramWithSource(context, 1, (const char **) & BUFFER_FOR_OPENCL_CODE, NULL, &err);
  // mxcheckGPUErrors(err);


  FILE * TempKernel;
  TempKernel=fopen("kernel.cu", "w");
  fprintf(TempKernel,"%s",BUFFER_FOR_OPENCL_CODE);
  fclose(TempKernel);
  char scmd [80];
  sprintf(scmd,"./pi_ocl --device %i",SelDevice);
  system(scmd);


  char * binary;
  size_t binary_size;
  long l_szie;
  cl_int binary_status;
  binary = common_read_file("KERNEL.BIN", &l_szie);
  binary_size=l_szie;
  program = clCreateProgramWithBinary(
        context, 1, &device_id[SelDevice], &binary_size,
        (const unsigned char **)&binary, &binary_status, &err
    );
  mxcheckGPUErrors(err);
  free(binary);


  PRINTF("After clCreateProgramWithSource\n");
  // Build the program
  err = clBuildProgram(program, 1, &device_id[SelDevice], NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      size_t len;
      char buffer[200048];

      PRINTF("Error: Failed to build program executable!\n%s\n", opencl_err_code(err));
      clGetProgramBuildInfo(program, device_id[SelDevice], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      PRINTF("%s\n", buffer);
      ERROR_STRING("Unable to build program");
  }

  // Create the compute kernel from the program
    StressKernel = clCreateKernel(program, "StressKernel", &err);
    mxcheckGPUErrors(err);
    ParticleKernel = clCreateKernel(program, "ParticleKernel", &err);
    mxcheckGPUErrors(err);

    SnapShot = clCreateKernel(program, "SnapShot", &err);
    mxcheckGPUErrors(err);
    SensorsKernel = clCreateKernel(program, "SensorsKernel", &err);
    mxcheckGPUErrors(err);

#endif

    //Only used these for the PML
	unsigned int SizeCopy;
	ownCudaCalloc(V_x_x,mexType,INHOST(SizePMLxp1));
	ownCudaCalloc(V_y_x,mexType,INHOST(SizePMLxp1));
	ownCudaCalloc(V_z_x,mexType,INHOST(SizePMLxp1));
	ownCudaCalloc(V_x_y,mexType,INHOST(SizePMLyp1));
	ownCudaCalloc(V_y_y,mexType,INHOST(SizePMLyp1));
	ownCudaCalloc(V_z_y,mexType,INHOST(SizePMLyp1));
	ownCudaCalloc(V_x_z,mexType,INHOST(SizePMLzp1));
	ownCudaCalloc(V_y_z,mexType,INHOST(SizePMLzp1));
	ownCudaCalloc(V_z_z,mexType,INHOST(SizePMLzp1));
	ownCudaCalloc(Sigma_x_xx,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_y_xx,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_z_xx,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_x_yy,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_y_yy,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_z_yy,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_x_zz,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_y_zz,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_z_zz,mexType,INHOST(SizePML));
	ownCudaCalloc(Sigma_x_xy,mexType,INHOST(SizePMLxp1yp1zp1));
	ownCudaCalloc(Sigma_y_xy,mexType,INHOST(SizePMLxp1yp1zp1));
	ownCudaCalloc(Sigma_x_xz,mexType,INHOST(SizePMLxp1yp1zp1));
	ownCudaCalloc(Sigma_z_xz,mexType,INHOST(SizePMLxp1yp1zp1));
	ownCudaCalloc(Sigma_y_yz,mexType,INHOST(SizePMLxp1yp1zp1));
	ownCudaCalloc(Sigma_z_yz,mexType,INHOST(SizePMLxp1yp1zp1));

  SizeCopy = GET_NUMBER_ELEMS(Sigma_xx_res);
	ownCudaCalloc(Rxx,mexType,SizeCopy);
	ownCudaCalloc(Ryy,mexType,SizeCopy);
	ownCudaCalloc(Rzz,mexType,SizeCopy);
	SizeCopy = GET_NUMBER_ELEMS(Sigma_xy_res);
	ownCudaCalloc(Rxy,mexType,SizeCopy);
	ownCudaCalloc(Rxz,mexType,SizeCopy);
	ownCudaCalloc(Ryz,mexType,SizeCopy);

	//These come from the user input
	CreateAndCopyFromMXVarOnGPU(LambdaMiuMatOverH,mexType);
	CreateAndCopyFromMXVarOnGPU(LambdaMatOverH	,mexType);
	CreateAndCopyFromMXVarOnGPU(MiuMatOverH,mexType);
	CreateAndCopyFromMXVarOnGPU(TauLong,mexType);
	CreateAndCopyFromMXVarOnGPU(OneOverTauSigma	,mexType);
	CreateAndCopyFromMXVarOnGPU(TauShear,mexType);
	CreateAndCopyFromMXVarOnGPU(InvRhoMatH		,mexType);
  CreateAndCopyFromMXVarOnGPU(Ox		,mexType);
  CreateAndCopyFromMXVarOnGPU(Oy		,mexType);
  CreateAndCopyFromMXVarOnGPU(Oz		,mexType);
	CreateAndCopyFromMXVarOnGPU(IndexSensorMap	,unsigned int);
	CreateAndCopyFromMXVarOnGPU(SourceFunctions	,mexType);
	CreateAndCopyFromMXVarOnGPU(SourceMap		,unsigned int);
	CreateAndCopyFromMXVarOnGPU(MaterialMap		,unsigned int);

  ownCudaCalloc(Vx,mexType,GET_NUMBER_ELEMS(Vx_res));
  ownCudaCalloc(Vy,mexType,GET_NUMBER_ELEMS(Vy_res));
  ownCudaCalloc(Vz,mexType,GET_NUMBER_ELEMS(Vz_res));
  ownCudaCalloc(Sigma_xx,mexType,GET_NUMBER_ELEMS(Sigma_xx_res));
  ownCudaCalloc(Sigma_yy,mexType,GET_NUMBER_ELEMS(Sigma_yy_res));
  ownCudaCalloc(Sigma_zz,mexType,GET_NUMBER_ELEMS(Sigma_zz_res));
  ownCudaCalloc(Sigma_xy,mexType,GET_NUMBER_ELEMS(Sigma_xy_res));
  ownCudaCalloc(Sigma_xz,mexType,GET_NUMBER_ELEMS(Sigma_xz_res));
  ownCudaCalloc(Sigma_yz,mexType,GET_NUMBER_ELEMS(Sigma_yz_res));

#ifdef CUDA
    mexType * gpu_Snapshots_pr=NULL;
    InputDataKernel pHost;
#else
   cl_mem gpu_Snapshots_pr;
#endif

	CreateAndCopyFromMXVarOnGPU2(Snapshots,mexType);

  CreateAndCopyFromMXVarOnGPU(SensorOutput,mexType);
  CreateAndCopyFromMXVarOnGPU(SqrAcc,mexType);

	//putting Pointers in structure
	InParamP(V_x_x);
	InParamP(V_y_x);
	InParamP(V_z_x);
	InParamP(V_x_y);
	InParamP(V_y_y);
	InParamP(V_z_y);
	InParamP(V_x_z);
  InParamP(V_y_z);
	InParamP(V_z_z);
	InParamP(Sigma_x_xx);
	InParamP(Sigma_y_xx);
	InParamP(Sigma_z_xx);
	InParamP(Sigma_x_yy);
	InParamP(Sigma_y_yy);
	InParamP(Sigma_z_yy);
	InParamP(Sigma_x_zz);
	InParamP(Sigma_y_zz);
	InParamP(Sigma_z_zz);
	InParamP(Sigma_x_xy);
	InParamP(Sigma_y_xy);
	InParamP(Sigma_x_xz);
	InParamP(Sigma_z_xz);
	InParamP(Sigma_y_yz);
	InParamP(Sigma_z_yz);
	InParamP(LambdaMiuMatOverH);
	InParamP(LambdaMatOverH);
	InParamP(MiuMatOverH);
	InParamP(TauLong);
	InParamP(OneOverTauSigma);
	InParamP(TauShear);
	InParamP(InvRhoMatH);
	InParamP(SourceFunctions);
	InParamP(SourceMap);
	InParamP(MaterialMap);
  InParamP(Vx);
  InParamP(Vy);
  InParamP(Vz);
  InParamP(Rxx);
  InParamP(Ryy);
  InParamP(Rzz);
  InParamP(Rxy);
  InParamP(Rxz);
  InParamP(Ryz);
  InParamP(Sigma_xx);
  InParamP(Sigma_yy);
  InParamP(Sigma_zz);
  InParamP(Sigma_xy);
  InParamP(Sigma_xz);
  InParamP(Sigma_yz);
  InParamP(Snapshots);
  InParamP(SqrAcc);
  InParamP(Ox);
  InParamP(Oy);
  InParamP(Oz);


//unsigned int MaxIndex = INHOST(InternalIndexCount)> INHOST(EdgeIndexCount) ? INHOST(InternalIndexCount):INHOST(EdgeIndexCount);

#if defined(CUDA)


  //copying the structure to GPU memory
    InputDataKernel * pGPU;
	mxcheckGPUErrors(cudaMalloc((void **)&pGPU,sizeof(InputDataKernel)));
	NumberAlloc++;
	PRINTF("size of  InputDataKernel =%ld\n", sizeof(InputDataKernel));
	mxcheckGPUErrors(cudaMemcpy(pGPU, &pHost, sizeof(InputDataKernel), cudaMemcpyHostToDevice));


  struct cudaFuncAttributes funcAttrib;
  checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, StressKernel));

  int blockSizeStress;   // The launch configurator returned block size
  int minGridSizeStress; // The minimum grid size needed to achieve the
  int blockSizeParticle;   // The launch configurator returned block size
  int minGridSizeParticle; // The minimum grid size needed to achieve the
  int blockSizeSnap;   // The launch configurator returned block size
  int minGridSizeSnap; // The minimum grid size needed to achieve the
  int blockSizeSensor;   // The launch configurator returned block size
  int minGridSizeSensor; // The minimum grid size needed to achieve the
               // maximum occupancy for a full device launch

   //Calculate the block dimensions
 	dim3              dimBlockStress;
  dim3              dimGridStress;
  dim3              dimBlockParticle;
  dim3              dimGridParticle;
  dim3              dimBlockSnap;
  dim3              dimGridSnap;
  dim3              dimBlockSensors;
  dim3              dimGridSensors;
  // dimBlockStress.x = k_blockDimX;
  // dimBlockStress.y= k_blockDimY;
  // dimBlockStress.z= k_blockDimZ;

  cudaOccupancyMaxPotentialBlockSize( &minGridSizeStress, &blockSizeStress,
                                  StressKernel, 0, 0);
  PRINTF("minGridSize and Blocksize from API for stress = %i and %i\n",minGridSizeStress,blockSizeStress);
  dimBlockStress.x=8;
  dimBlockStress.y=8;
  dimBlockStress.z=(unsigned int)floor(blockSizeStress/(dimBlockStress.y*dimBlockStress.x));

  dimGridStress.x  = (unsigned int)ceil((float)(INHOST(N1)+1) / dimBlockStress.x);
  dimGridStress.y  = (unsigned int)ceil((float)(INHOST(N2)+1) / dimBlockStress.y);
  dimGridStress.z  = (unsigned int)ceil((float)(INHOST(N3)+1) / dimBlockStress.z);
  PRINTF(" Stress block size to %dx%dx%d\n", dimBlockStress.x, dimBlockStress.y,dimBlockStress.z);
  PRINTF(" Stress grid size to %dx%dx%d\n", dimGridStress.x, dimGridStress.y,dimGridStress.z);

  cudaOccupancyMaxPotentialBlockSize( &minGridSizeParticle, &blockSizeParticle,
                                  ParticleKernel, 0, 0);
  PRINTF("minGridSize and Blocksize from API for Particle = %i and %i\n",minGridSizeParticle,blockSizeParticle);
  dimBlockParticle.x=8;
  dimBlockParticle.y=8;
  dimBlockParticle.z=(unsigned int)floor(blockSizeParticle/(dimBlockParticle.y*dimBlockParticle.x));

  dimGridParticle.x  = (unsigned int)ceil((float)(INHOST(N1)+1) / dimBlockParticle.x);
  dimGridParticle.y  = (unsigned int)ceil((float)(INHOST(N2)+1) / dimBlockParticle.y);
  dimGridParticle.z  = (unsigned int)ceil((float)(INHOST(N3)+1) / dimBlockParticle.z);
  PRINTF(" Particle block size to %dx%dx%d\n", dimBlockParticle.x, dimBlockParticle.y,dimBlockParticle.z);
  PRINTF(" Particle grid size to %dx%dx%d\n", dimGridParticle.x, dimGridParticle.y,dimGridParticle.z);

  cudaOccupancyMaxPotentialBlockSize( &minGridSizeSnap, &blockSizeSnap,
                                  SnapShot, 0, 0);
  PRINTF("N1:minGridSize and Blocksize from API for SnapShot = %i and %i\n",minGridSizeSnap,blockSizeSnap);
  dimBlockSnap.x=8;
  dimBlockSnap.y=(unsigned int)floor(blockSizeSnap/(dimBlockSnap.x));

  dimGridSnap.x  = (unsigned int)ceil((float)(INHOST(N1)+1) / dimBlockSnap.x);
  dimGridSnap.y  = (unsigned int)ceil((float)(INHOST(N2)+1) / dimBlockSnap.y);

  PRINTF(" Snapshot block size to %dx%d\n", dimBlockSnap.x, dimBlockSnap.y);
  PRINTF(" Snapshot grid size to %dx%d\n", dimGridSnap.x, dimGridSnap.y);

  cudaOccupancyMaxPotentialBlockSize( &minGridSizeSensor, &blockSizeSensor,
                                  SensorsKernel, 0, 0);
  PRINTF("minGridSize and Blocksize from API for SensorsKernel = %i and %i\n",minGridSizeSensor,blockSizeSensor);
  dimBlockSensors.x=blockSizeSensor;
  dimBlockSensors.y=1;
  dimGridSensors.x  = (unsigned int)ceil((float)(NumberSensors) / dimBlockSensors.x);
  dimGridSensors.y=1;


  PRINTF(" set sensor block size to %dx%d\n", dimBlockSensors.x, dimBlockSensors.y);
  PRINTF(" set sensor grid size to %dx%d\n", dimGridSensors.x, dimGridSensors.y);

  size_t free_byte ;
  size_t total_byte ;

  mxcheckGPUErrors(cudaMemGetInfo( &free_byte, &total_byte ));
  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;

  PRINTF("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
  #define TOTAL_streams 12
  cudaStream_t streams[TOTAL_streams];
  for (unsigned n =0;n<TOTAL_streams;n++)
  mxcheckGPUErrors(cudaStreamCreate ( &streams[n])) ;


#else
  const  size_t global_stress_particle[3] ={N1,N2,N3};
  const  size_t global_sensors[1] ={NumberSensors};

  mxcheckGPUErrors(clSetKernelArg(SnapShot, 1, sizeof(cl_mem), &gpu_Snapshots_pr));
  mxcheckGPUErrors(clSetKernelArg(SnapShot, 2, sizeof(cl_mem), &gpu_Sigma_xx_pr));
  mxcheckGPUErrors(clSetKernelArg(SnapShot, 3, sizeof(cl_mem), &gpu_Sigma_yy_pr));
  mxcheckGPUErrors(clSetKernelArg(SnapShot, 4, sizeof(cl_mem), &gpu_Sigma_zz_pr));

  mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 0, sizeof(cl_mem), &gpu_SensorOutput_pr));
  mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 1, sizeof(cl_mem), &gpu_Vx_pr));
  mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 2, sizeof(cl_mem), &gpu_Vy_pr));
  mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 3, sizeof(cl_mem), &gpu_Vz_pr));
  mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 4, sizeof(cl_mem), &gpu_IndexSensorMap_pr));


#endif


	for (unsigned int nStep=0;nStep<TimeSteps;nStep++)
	{

#if defined(CUDA)
	      StressKernel<<<dimGridStress, dimBlockStress>>>(pGPU,nStep);
        mxcheckGPUErrors(cudaDeviceSynchronize());

        //~ //********************************
        //********************************
        //Then we do the particle displacements
        //********************************
        ParticleKernel<<<dimGridParticle, dimBlockParticle,0,streams[0]>>>(pGPU,nStep,CurrSnap, SnapshotsPos_pr[CurrSnap]-1,INHOST(TypeSource));
        mxcheckGPUErrors(cudaDeviceSynchronize());

#else
        int nextSnap=SnapshotsPos_pr[CurrSnap]-1;
        mxcheckGPUErrors(clSetKernelArg(StressKernel, 54, sizeof(unsigned int), &nStep));


        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 54, sizeof(unsigned int), &nStep));
        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 55, sizeof(unsigned int), &CurrSnap));
        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 56, sizeof(unsigned int), &nextSnap));
        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 57, sizeof(unsigned int), &INHOST(TypeSource)));
        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, StressKernel, 3, NULL, global_stress_particle, NULL, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));
        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, ParticleKernel, 3, NULL, global_stress_particle, NULL, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));

#endif


   // Snapshots
		if (CurrSnap <NumberSnapshots)
			if(nStep==SnapshotsPos_pr[CurrSnap]-1)
			{
  #if defined(CUDA)
				SnapShot<<<dimGridSnap,dimBlockSnap,0,streams[6]>>>(INHOST(N3)/2,gpu_Snapshots_pr,gpu_Sigma_xx_pr,gpu_Sigma_yy_pr,gpu_Sigma_zz_pr,CurrSnap);
				mxcheckGPUErrors(cudaDeviceSynchronize());
  #else
        int selfSlice=INHOST(N3)/2;
        mxcheckGPUErrors(clSetKernelArg(SnapShot, 0, sizeof(unsigned int), &selfSlice));
        mxcheckGPUErrors(clSetKernelArg(SnapShot, 5, sizeof(unsigned int), &CurrSnap));

        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, SnapShot, 2, NULL, global_stress_particle, NULL, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));
  #endif
				CurrSnap++;
			}
		//~ //Finally, the sensors
#if defined(CUDA)
     SensorsKernel<<<dimGridSensors,dimBlockSensors,0,streams[7]>>>(gpu_SensorOutput_pr,gpu_Vx_pr,
       gpu_Vy_pr,gpu_Vz_pr,gpu_IndexSensorMap_pr,nStep,NumberSensors,TimeSteps);
		 mxcheckGPUErrors(cudaDeviceSynchronize());
#else
      mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 5, sizeof(unsigned int), &nStep));
      mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 6, sizeof(unsigned int), &NumberSensors));
      mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 7, sizeof(unsigned int), &TimeSteps));

      mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, SensorsKernel, 1, NULL, global_sensors, NULL, 0, NULL, NULL));
      mxcheckGPUErrors(clFinish(commands));
#endif

	}
  LOCAL_CALLOC(Vx,GET_NUMBER_ELEMS(Vx_res));
  LOCAL_CALLOC(Vy,GET_NUMBER_ELEMS(Vy_res));
  LOCAL_CALLOC(Vz,GET_NUMBER_ELEMS(Vz_res));
  LOCAL_CALLOC(Sigma_xx,GET_NUMBER_ELEMS(Sigma_xx_res));
  LOCAL_CALLOC(Sigma_yy,GET_NUMBER_ELEMS(Sigma_yy_res));
  LOCAL_CALLOC(Sigma_zz,GET_NUMBER_ELEMS(Sigma_zz_res));
  LOCAL_CALLOC(Sigma_xy,GET_NUMBER_ELEMS(Sigma_xy_res));
  LOCAL_CALLOC(Sigma_xz,GET_NUMBER_ELEMS(Sigma_xz_res));
  LOCAL_CALLOC(Sigma_yz,GET_NUMBER_ELEMS(Sigma_yz_res));

	//DONE, just to copy to the host the results
	CopyFromGPUToMX3(SensorOutput,mexType);
	CopyFromGPUToMX(Vx,mexType);
  CopyFromGPUToMX(Vy,mexType);
  CopyFromGPUToMX(Vz,mexType);
  CopyFromGPUToMX(Sigma_xx,mexType);
  CopyFromGPUToMX(Sigma_yy,mexType);
  CopyFromGPUToMX(Sigma_xy,mexType);
  CopyFromGPUToMX(Sigma_xz,mexType);
  CopyFromGPUToMX(Sigma_yz,mexType);
  CopyFromGPUToMX3(SqrAcc,mexType);

  {
    unsigned i,j,k,CurZone;
  #pragma omp parallel for private(j,i,CurZone)
  for(k=0; k<INHOST(N3); k++)
    for(j=0; j<INHOST(N2); j++)
      for(i=0; i<INHOST(N1); i++)
      {
        ASSIGN_RES(Vx);
        ASSIGN_RES(Vy);
        ASSIGN_RES(Vz);
        ASSIGN_RES(Sigma_xx);
        ASSIGN_RES(Sigma_yy);
        ASSIGN_RES(Sigma_zz);
        ASSIGN_RES(Sigma_xy);
        ASSIGN_RES(Sigma_xz);
        ASSIGN_RES(Sigma_yz);
      }
 }

#if defined(CUDA)
    for (unsigned n =0;n<TOTAL_streams;n++)
		    mxcheckGPUErrors(cudaStreamDestroy ( streams[n])) ;
#endif

    if (NumberSnapshots>0)
    {
		CopyFromGPUToMX3(Snapshots,mexType);
	}
#if defined(CUDA)
	mxcheckGPUErrors(cudaFree(pGPU)); NumberAlloc--;
  	free(sm13DeviceList);
#endif
  free(Vx_pr);
  free(Vy_pr);
  free(Vz_pr);
  free(Sigma_xx_pr);
  free(Sigma_yy_pr);
  free(Sigma_zz_pr);
  free(Sigma_xy_pr);
  free(Sigma_xz_pr);
  free(Sigma_yz_pr);


	ownCudaFree(SensorOutput);
	ownCudaFree(V_x_x);
  ownCudaFree(V_y_x);
  ownCudaFree(V_z_x);
  ownCudaFree(V_x_y);
  ownCudaFree(V_y_y);
  ownCudaFree(V_z_y);
  ownCudaFree(V_x_z);
  ownCudaFree(V_y_z);
  ownCudaFree(V_z_z);
  ownCudaFree(Sigma_x_xx);
  ownCudaFree(Sigma_y_xx);
  ownCudaFree(Sigma_z_xx);
  ownCudaFree(Sigma_x_yy);
  ownCudaFree(Sigma_y_yy);
  ownCudaFree(Sigma_z_yy);
  ownCudaFree(Sigma_x_zz);
  ownCudaFree(Sigma_y_zz);
  ownCudaFree(Sigma_z_zz);
  ownCudaFree(Sigma_x_xy);
  ownCudaFree(Sigma_y_xy);
  ownCudaFree(Sigma_x_xz);
  ownCudaFree(Sigma_z_xz);
  ownCudaFree(Sigma_y_yz);
  ownCudaFree(Sigma_z_yz);
  ownCudaFree(LambdaMiuMatOverH);
	ownCudaFree(LambdaMatOverH);
	ownCudaFree(MiuMatOverH);
	ownCudaFree(TauLong);
	ownCudaFree(OneOverTauSigma);
	ownCudaFree(TauShear);
	ownCudaFree(InvRhoMatH);
	ownCudaFree(IndexSensorMap);
	ownCudaFree(SourceFunctions);
	ownCudaFree(SourceMap);
  ownCudaFree(Ox);
  ownCudaFree(Oy);
  ownCudaFree(Oz);
	ownCudaFree(MaterialMap);
  ownCudaFree(Vx);
  ownCudaFree(Vy);
  ownCudaFree(Vz);
  ownCudaFree(Sigma_xx);
  ownCudaFree(Sigma_yy);
  ownCudaFree(Sigma_zz);
  ownCudaFree(Sigma_xy);
  ownCudaFree(Sigma_xz);
  ownCudaFree(Sigma_yz);
  ownCudaFree(Rxx);
  ownCudaFree(Ryy);
  ownCudaFree(Rzz);
  ownCudaFree(Rxy);
  ownCudaFree(Rxz);
  ownCudaFree(Ryz);
  ownCudaFree(Snapshots);
  ownCudaFree(SqrAcc);


#if defined(CUDA)
	mxcheckGPUErrors(cudaMemGetInfo( &free_byte, &total_byte ));
    free_db = (double)free_byte ;
    total_db = (double)total_byte ;
    used_db = total_db - free_db ;

    PRINTF("GPU memory remaining (free should be equal to total): used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
#else
    clReleaseProgram(program);
    clReleaseKernel(StressKernel);
    clReleaseKernel(ParticleKernel);
    clReleaseKernel(SensorsKernel);
    clReleaseKernel(SnapShot);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
#endif
PRINTF("Number of unfreed allocs (it should be 0):%i\n",NumberAlloc);
