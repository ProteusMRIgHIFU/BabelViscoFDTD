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
               PRINTF("  At least one device available [%s] for calculations \n", deviceProperties.name);
                // Add device to 3-capable list
                sm13DeviceList[sm13DeviceCount] = deviceID;
                sm13DeviceCount++;
            }
        }
    }

    // Were any of the devices 1.3-capable?
    if(sm13DeviceCount == 0)
    {
        ERROR_STRING("There are no devices supporting CUDA or that matches selected device.\n");
    }

    if (INHOST(DefaultGPUDeviceNumber)>= sm13DeviceCount)
    {
      PRINTF("The requested device [%i] (0-base index) is more than the number of devices available [%i] \n",INHOST(DefaultGPUDeviceNumber),sm13DeviceCount);
      ERROR_STRING("Unable to select requested device.\n");
    }

    PRINTF("Selecting device [%s] with number [%i] for calculations\n", DefaultGPUDeviceName_pr,INHOST(DefaultGPUDeviceNumber));

    mxcheckGPUErrors(cudaSetDevice(sm13DeviceList[INHOST(DefaultGPUDeviceNumber)]));

    mxcheckGPUErrors(cudaDeviceSetCacheConfig (cudaFuncCachePreferL1));
#endif

#ifdef OPENCL
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
    int SelDevice=-1;

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
        break;
      }
    }

    if (SelDevice==-1)
    {
      PRINTF("Device requested %s \n",DefaultGPUDeviceName_pr);
      ERROR_STRING("Device requested was not found!\n");
    }


    context = clCreateContext(0, 1, &device_id[SelDevice], NULL, NULL, &err);
    mxcheckGPUErrors(err);

    // Create a command queue

    commands = clCreateCommandQueue(context, device_id[SelDevice], 0, &err);
    mxcheckGPUErrors(err);

    sprintf(BUFFER_FOR_GPU_CODE,"\n#define mexType %s\n#define OPENCL\n",MEX_STR);
    char * indexingSource = load_file("_indexing.h");
    if (indexingSource==0)
    {
      ERROR_STRING("Unable to read _indexing.h file!!")
    }
    strncat(BUFFER_FOR_GPU_CODE,indexingSource,MAXP_BUFFER_GPU_CODE);
    strncat(BUFFER_FOR_GPU_CODE,"\n",MAXP_BUFFER_GPU_CODE);
    free(indexingSource);

#endif

#ifdef METAL
    unsigned int _c_mex_type = 0;
    unsigned int _c_uint_type = 0;
    unsigned int HOST_INDEX_MEX[LENGTH_INDEX_MEX][2];

    unsigned int HOST_INDEX_UINT[LENGTH_INDEX_MEX][2];

    ns::Array<mtlpp::Device>  AllDev= mtlpp::Device::CopyAllDevices();
    mxcheckGPUErrors(((int)AllDev));


    if (AllDev.GetSize()==0)
    {
      ERROR_STRING("Found 0 METAL platforms!\n");
    }
    unsigned int SelDevice=0;
    {
      for (int _n = 0;_n<AllDev.GetSize();_n++)
      {
        PRINTF("Meta device available: %i %s\n",_n,AllDev[_n].GetName().GetCStr());
        if (NULL!=strstr(AllDev[_n].GetName().GetCStr(),DefaultGPUDeviceName_pr))
        {
          PRINTF("Found %s device!\n",DefaultGPUDeviceName_pr);
          SelDevice=_n;
          break;
        }
      }
    }
    if (SelDevice==-1)
    {
      PRINTF("Device requested %s \n",DefaultGPUDeviceName_pr);
      ERROR_STRING("Device requested was not found!\n");
    }

    mtlpp::Device device= AllDev[SelDevice];

    sprintf(BUFFER_FOR_GPU_CODE,"\n#define mexType %s\n#define METAL\n"
                                "#include <metal_stdlib>\nusing namespace metal;\n"
                                "#define MAX_SIZE_PML %i\n",MEX_STR,MAX_SIZE_PML);
    char * indexingSource = load_file("_indexing.h");
    if (indexingSource==0)
    {
      ERROR_STRING("Unable to read _indexing.h file!!")
    }
    strncat(BUFFER_FOR_GPU_CODE,indexingSource,MAXP_BUFFER_GPU_CODE);
    strncat(BUFFER_FOR_GPU_CODE,"\n",MAXP_BUFFER_GPU_CODE);
    free(indexingSource);

    char * KernelSource = load_file("_gpu_kernel.c");
    if (KernelSource==0)
    {
      ERROR_STRING("Unable to read _gpu_kernel.c file!!")
    }
    strncat(BUFFER_FOR_GPU_CODE,KernelSource,MAXP_BUFFER_GPU_CODE);
    strncat(BUFFER_FOR_GPU_CODE,"\n",MAXP_BUFFER_GPU_CODE);
    free(KernelSource);

    PRINTF("After reading files\n");

    ns::Error error;

    mtlpp::Library library = device.NewLibrary(BUFFER_FOR_GPU_CODE, mtlpp::CompileOptions(), &error);
    if (((int)library)==0)
    {
      FILE * TempKernel;
      TempKernel=fopen("__For_Analysis_kernel.m", "w");
      fprintf(TempKernel,"%s",BUFFER_FOR_GPU_CODE);
      fclose(TempKernel);
        PRINTF("GetLocalizedDescription = %s\n",error.GetLocalizedDescription().GetCStr());
        PRINTF("GetLocalizedFailureReason = %s\n",error.GetLocalizedFailureReason().GetCStr());
        PRINTF("GetLocalizedRecoverySuggestion = %s\n",error.GetLocalizedRecoverySuggestion().GetCStr());
        PRINTF("GetLocalizedRecoveryOptions = %s\n",error.GetLocalizedRecoveryOptions().GetCStr());
        PRINTF("GetHelpAnchor = %s\n",error.GetHelpAnchor().GetCStr());
        ERROR_STRING("Error in compilation, see also file __For_Analysis_kernel.m that was generated with the metal code in the current directory")
    }
    mxcheckGPUErrors(((int)library));

    PRINTF("After compiling code \n");

    mtlpp::Function ParticleKernelFunc = library.NewFunction("ParticleKernel");
    mxcheckGPUErrors(((int)ParticleKernelFunc));
    mtlpp::ComputePipelineState computePipelineStateParticle = device.NewComputePipelineState(ParticleKernelFunc, nullptr);
    mxcheckGPUErrors(((int)computePipelineStateParticle));

    mtlpp::Function StressKernelFunc = library.NewFunction("StressKernel");
    mxcheckGPUErrors(((int)StressKernelFunc));
    mtlpp::ComputePipelineState computePipelineStateStress = device.NewComputePipelineState(StressKernelFunc, nullptr);
    mxcheckGPUErrors(((int)computePipelineStateStress));

    mtlpp::Function SnapShotFunc = library.NewFunction("SnapShot");
    mxcheckGPUErrors(((int)SnapShotFunc));
    mtlpp::ComputePipelineState computePipelineStateSnapShot = device.NewComputePipelineState(SnapShotFunc, nullptr);
    mxcheckGPUErrors(((int)computePipelineStateSnapShot));

    mtlpp::Function SensorsKernelFunc = library.NewFunction("SensorsKernel");
    mxcheckGPUErrors(((int)SensorsKernelFunc));
    mtlpp::ComputePipelineState computePipelineStateSensors = device.NewComputePipelineState(SensorsKernelFunc, nullptr);
    mxcheckGPUErrors(((int)computePipelineStateSensors));

    PRINTF("After getting all functions code \n");

    mtlpp::CommandQueue commandQueue = device.NewCommandQueue();
    mxcheckGPUErrors(((int)commandQueue));

    mtlpp::Buffer _CONSTANT_BUFFER_UINT = device.NewBuffer(sizeof(unsigned int) * LENGTH_CONST_UINT, mtlpp::ResourceOptions::StorageModeManaged);
    mxcheckGPUErrors(((int)_CONSTANT_BUFFER_UINT));

    mtlpp::Buffer _CONSTANT_BUFFER_MEX = device.NewBuffer(sizeof(mexType) * LENGTH_CONST_MEX, mtlpp::ResourceOptions::StorageModeManaged);
    mxcheckGPUErrors(((int)_CONSTANT_BUFFER_MEX));

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

InitSymbol(NumberSensors,unsigned int,G_INT);
InitSymbol(TimeSteps,unsigned int,G_INT);
InitSymbol(SelRMSorPeak,unsigned int,G_INT);

InitSymbol(SelMapsRMSPeak,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_ALLV,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Vx,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Vy,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Vz,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Sigmaxx,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Sigmayy,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Sigmazz,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Sigmaxy,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Sigmaxz,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Sigmayz,unsigned int,G_INT);
InitSymbol(IndexRMSPeak_Pressure,unsigned int,G_INT);
InitSymbol(NumberSelRMSPeakMaps,unsigned int,G_INT);

InitSymbol(SelMapsSensors,unsigned int,G_INT);
InitSymbol(IndexSensor_ALLV,unsigned int,G_INT);
InitSymbol(IndexSensor_Vx,unsigned int,G_INT);
InitSymbol(IndexSensor_Vy,unsigned int,G_INT);
InitSymbol(IndexSensor_Vz,unsigned int,G_INT);
InitSymbol(IndexSensor_Sigmaxx,unsigned int,G_INT);
InitSymbol(IndexSensor_Sigmayy,unsigned int,G_INT);
InitSymbol(IndexSensor_Sigmazz,unsigned int,G_INT);
InitSymbol(IndexSensor_Sigmaxy,unsigned int,G_INT);
InitSymbol(IndexSensor_Sigmaxz,unsigned int,G_INT);
InitSymbol(IndexSensor_Sigmayz,unsigned int,G_INT);
InitSymbol(IndexSensor_Pressure,unsigned int,G_INT);
InitSymbol(NumberSelSensorMaps,unsigned int,G_INT);
InitSymbol(SensorSubSampling,unsigned int,G_INT);
InitSymbol(SensorStart,unsigned int,G_INT);

//~
#ifdef CUDA //CUDA specifics

	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuInvDXDTpluspr,InvDXDTplus_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));
	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuDXDTminuspr,DXDTminus_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));
	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuInvDXDTplushppr,InvDXDTplushp_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));
	mxcheckGPUErrors(cudaMemcpyToSymbol(gpuDXDTminushppr,DXDTminushp_pr,(INHOST(PML_Thickness)+1)*sizeof(mexType)));
#endif

#if defined(OPENCL) || defined(METAL)
  InitSymbolArray(InvDXDTplus,G_FLOAT,INHOST(PML_Thickness)+1);
  InitSymbolArray(DXDTminus,G_FLOAT,INHOST(PML_Thickness)+1);
  InitSymbolArray(InvDXDTplushp,G_FLOAT,INHOST(PML_Thickness)+1);
  InitSymbolArray(DXDTminushp,G_FLOAT,INHOST(PML_Thickness)+1);

#endif

#ifdef OPENCL
  char * KernelSource = load_file("_gpu_kernel.c");
  if (KernelSource==0)
  {
    ERROR_STRING("Unable to read _gpu_kernel.c file!!")
  }
  strncat(BUFFER_FOR_GPU_CODE,KernelSource,MAXP_BUFFER_GPU_CODE);
  strncat(BUFFER_FOR_GPU_CODE,"\n",MAXP_BUFFER_GPU_CODE);
  free(KernelSource);
  //PRINTF("%s",BUFFER_FOR_GPU_CODE);

  // program = clCreateProgramWithSource(context, 1, (const char **) & BUFFER_FOR_GPU_CODE, NULL, &err);
  // mxcheckGPUErrors(err);


  FILE * TempKernel;
  TempKernel=fopen("kernel.cu", "w");
  fprintf(TempKernel,"%s",BUFFER_FOR_GPU_CODE);
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
	_PT SizeCopy;
	ownGpuCalloc(V_x_x,mexType,INHOST(SizePMLxp1));
	ownGpuCalloc(V_y_x,mexType,INHOST(SizePMLxp1));
	ownGpuCalloc(V_z_x,mexType,INHOST(SizePMLxp1));
	ownGpuCalloc(V_x_y,mexType,INHOST(SizePMLyp1));
	ownGpuCalloc(V_y_y,mexType,INHOST(SizePMLyp1));
	ownGpuCalloc(V_z_y,mexType,INHOST(SizePMLyp1));
	ownGpuCalloc(V_x_z,mexType,INHOST(SizePMLzp1));
	ownGpuCalloc(V_y_z,mexType,INHOST(SizePMLzp1));
	ownGpuCalloc(V_z_z,mexType,INHOST(SizePMLzp1));
	ownGpuCalloc(Sigma_x_xx,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_y_xx,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_z_xx,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_x_yy,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_y_yy,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_z_yy,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_x_zz,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_y_zz,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_z_zz,mexType,INHOST(SizePML));
	ownGpuCalloc(Sigma_x_xy,mexType,INHOST(SizePMLxp1yp1zp1));
	ownGpuCalloc(Sigma_y_xy,mexType,INHOST(SizePMLxp1yp1zp1));
	ownGpuCalloc(Sigma_x_xz,mexType,INHOST(SizePMLxp1yp1zp1));
	ownGpuCalloc(Sigma_z_xz,mexType,INHOST(SizePMLxp1yp1zp1));
	ownGpuCalloc(Sigma_y_yz,mexType,INHOST(SizePMLxp1yp1zp1));
	ownGpuCalloc(Sigma_z_yz,mexType,INHOST(SizePMLxp1yp1zp1));

  SizeCopy = GET_NUMBER_ELEMS(Sigma_xx_res);
	ownGpuCalloc(Rxx,mexType,SizeCopy);
	ownGpuCalloc(Ryy,mexType,SizeCopy);
	ownGpuCalloc(Rzz,mexType,SizeCopy);
	SizeCopy = GET_NUMBER_ELEMS(Sigma_xy_res);
	ownGpuCalloc(Rxy,mexType,SizeCopy);
	ownGpuCalloc(Rxz,mexType,SizeCopy);
	ownGpuCalloc(Ryz,mexType,SizeCopy);

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
	CreateAndCopyFromMXVarOnGPU(SourceFunctions	,mexType);
  CreateAndCopyFromMXVarOnGPU(IndexSensorMap	,unsigned int);
	CreateAndCopyFromMXVarOnGPU(SourceMap		,unsigned int);
	CreateAndCopyFromMXVarOnGPU(MaterialMap		,unsigned int);

  ownGpuCalloc(Vx,mexType,GET_NUMBER_ELEMS(Vx_res));
  ownGpuCalloc(Vy,mexType,GET_NUMBER_ELEMS(Vy_res));
  ownGpuCalloc(Vz,mexType,GET_NUMBER_ELEMS(Vz_res));
  ownGpuCalloc(Sigma_xx,mexType,GET_NUMBER_ELEMS(Sigma_xx_res));
  ownGpuCalloc(Sigma_yy,mexType,GET_NUMBER_ELEMS(Sigma_yy_res));
  ownGpuCalloc(Sigma_zz,mexType,GET_NUMBER_ELEMS(Sigma_zz_res));
  ownGpuCalloc(Sigma_xy,mexType,GET_NUMBER_ELEMS(Sigma_xy_res));
  ownGpuCalloc(Sigma_xz,mexType,GET_NUMBER_ELEMS(Sigma_xz_res));
  ownGpuCalloc(Sigma_yz,mexType,GET_NUMBER_ELEMS(Sigma_yz_res));
  ownGpuCalloc(Pressure,mexType,GET_NUMBER_ELEMS(Pressure_res));

#ifdef CUDA
    mexType * gpu_Snapshots_pr=NULL;
    InputDataKernel pHost;
#endif
#ifdef OPENCL
   cl_mem gpu_Snapshots_pr;
#endif
#ifdef METAL
   mtlpp::Buffer gpu_Snapshots_pr;
#endif



  CreateAndCopyFromMXVarOnGPU2(Snapshots,mexType);
  CreateAndCopyFromMXVarOnGPU(SensorOutput,mexType);
  CreateAndCopyFromMXVarOnGPU(SqrAcc,mexType);

#ifdef METAL


  mtlpp::Buffer _MEX_BUFFER = device.NewBuffer(sizeof(mexType) *_c_mex_type,
            mtlpp::ResourceOptions::StorageModeManaged);
  mxcheckGPUErrors(((int)_MEX_BUFFER));

  mtlpp::Buffer _UINT_BUFFER = device.NewBuffer(sizeof(unsigned int) *_c_uint_type,
            mtlpp::ResourceOptions::StorageModeManaged);
  mxcheckGPUErrors(((int)_UINT_BUFFER));

  mtlpp::Buffer _INDEX_MEX = device.NewBuffer(sizeof(unsigned int) *
            LENGTH_INDEX_MEX,
            mtlpp::ResourceOptions::StorageModeManaged);
  mxcheckGPUErrors(((int)_INDEX_MEX));

  mtlpp::Buffer _INDEX_UINT = device.NewBuffer(sizeof(unsigned int) *
            LENGTH_INDEX_UINT,
            mtlpp::ResourceOptions::StorageModeManaged);
  mxcheckGPUErrors(((int)_INDEX_UINT));



  {
      unsigned int * inData = static_cast<unsigned int*>(_INDEX_MEX.GetContents());
      for (uint32_t j=0; j<LENGTH_INDEX_MEX; j++)
          inData[j] = HOST_INDEX_MEX[j][0];
      _INDEX_MEX.DidModify(ns::Range(0, sizeof(unsigned int) * LENGTH_INDEX_MEX));
  }

  {
      unsigned int * inData = static_cast< unsigned int *>(_INDEX_UINT.GetContents());
      for (uint32_t j=0; j<LENGTH_INDEX_UINT; j++)
          inData[j] = HOST_INDEX_UINT[j][0];
      _INDEX_UINT.DidModify(ns::Range(0, sizeof(unsigned int) * LENGTH_INDEX_UINT));
  }

  _CONSTANT_BUFFER_UINT.DidModify(ns::Range(0, sizeof(unsigned int)*LENGTH_CONST_UINT));
  _CONSTANT_BUFFER_MEX.DidModify(ns::Range(0,sizeof(mexType) * LENGTH_CONST_MEX));


  CompleteCopyToGpu(LambdaMiuMatOverH,mexType);
  CompleteCopyToGpu(LambdaMatOverH	,mexType);
  CompleteCopyToGpu(MiuMatOverH,mexType);
  CompleteCopyToGpu(TauLong,mexType);
  CompleteCopyToGpu(OneOverTauSigma	,mexType);
  CompleteCopyToGpu(TauShear,mexType);
  CompleteCopyToGpu(InvRhoMatH		,mexType);
  CompleteCopyToGpu(Ox		,mexType);
  CompleteCopyToGpu(Oy		,mexType);
  CompleteCopyToGpu(Oz		,mexType);
  CompleteCopyToGpu(SourceFunctions	,mexType);
  CompleteCopyToGpu(IndexSensorMap	,unsigned int);
  CompleteCopyToGpu(SourceMap		,unsigned int);
  CompleteCopyToGpu(MaterialMap		,unsigned int);

  _MEX_BUFFER.DidModify(ns::Range(0,sizeof(mexType) *_c_mex_type));

  _UINT_BUFFER.DidModify(ns::Range(0,sizeof(unsigned int) *_c_uint_type));

#endif

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
  InParamP(SqrAcc);
  InParamP(Pressure);
#ifdef CUDA
  InParamP(SensorOutput);
#endif
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
  dimGridSensors.x  = (unsigned int)ceil((float)(INHOST(NumberSensors)) / dimBlockSensors.x);
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
  #define TOTAL_streams 1
  cudaStream_t streams[TOTAL_streams];
  for (unsigned n =0;n<TOTAL_streams;n++)
  mxcheckGPUErrors(cudaStreamCreate ( &streams[n])) ;

#endif

#ifdef OPENCL
  const  size_t global_stress_particle[3] ={N1,N2,N3};
  const  size_t global_sensors[1] ={INHOST(NumberSensors)};
  if (NumberSnapshots>0)
  {
      mxcheckGPUErrors(clSetKernelArg(SnapShot, 1, sizeof(cl_mem), &gpu_Snapshots_pr));
      mxcheckGPUErrors(clSetKernelArg(SnapShot, 2, sizeof(cl_mem), &gpu_Sigma_xx_pr));
      mxcheckGPUErrors(clSetKernelArg(SnapShot, 3, sizeof(cl_mem), &gpu_Sigma_yy_pr));
      mxcheckGPUErrors(clSetKernelArg(SnapShot, 4, sizeof(cl_mem), &gpu_Sigma_zz_pr));
  }

  mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 54, sizeof(cl_mem), &gpu_SensorOutput_pr));
  mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 55, sizeof(cl_mem), &gpu_IndexSensorMap_pr));
#endif

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

  unsigned int INHOST(nStep)=0;
  while(INHOST(nStep)<INHOST(TimeSteps))
	{
#if defined(CUDA)
        unsigned int nCurStream=0;
        unsigned int maxStream=TOTAL_streams;
        if ((INHOST(TimeSteps)-INHOST(nStep))<maxStream)
            maxStream=INHOST(TimeSteps)-INHOST(nStep);
        while((INHOST(nStep)<INHOST(TimeSteps))&&(nCurStream<TOTAL_streams))
        {

            StressKernel<<<dimGridStress, dimBlockStress,0,streams[nCurStream]>>>(pGPU,INHOST(nStep),INHOST(TypeSource));
            // We let for future reference in case we want to offload the sensor task via memory transfer
            // if (((INHOST(nStep) % INHOST(SensorSubSampling))==0) && ((INHOST(nStep) / INHOST(SensorSubSampling))>=INHOST(SensorStart)))
		        // {
            //   //We copy pressure to start accumulating over time
            //   CopyFromGPUToMXAsync(Pressure,mexType,streams[nCurStream]);
            // }
            //~ //********************************
            //********************************
            //Then we do the particle displacements
            //********************************
            ParticleKernel<<<dimGridParticle, dimBlockParticle,0,streams[nCurStream]>>>(pGPU,INHOST(nStep),INHOST(TypeSource));
        
#endif
#ifdef OPENCL
        int nextSnap=-1;
        if (NumberSnapshots>0)
            nextSnap=SnapshotsPos_pr[INHOST(CurrSnap)]-1;
        mxcheckGPUErrors(clSetKernelArg(StressKernel, 54, sizeof(unsigned int), &INHOST(nStep)));
        mxcheckGPUErrors(clSetKernelArg(StressKernel, 55, sizeof(unsigned int), &INHOST(TypeSource)));
        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 54, sizeof(unsigned int), &INHOST(nStep)));
        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 55, sizeof(unsigned int), &INHOST(TypeSource)));
        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, StressKernel, 3, NULL, global_stress_particle, NULL, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));
        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, ParticleKernel, 3, NULL, global_stress_particle, NULL, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));

#endif
#ifdef METAL
        InitSymbol(nStep,unsigned int,G_INT);
        InitSymbol(TypeSource,unsigned int,G_INT);
        InitSymbol(SelK,unsigned int,G_INT);


        mtlpp::CommandBuffer commandBufferStress = commandQueue.CommandBuffer();
        mxcheckGPUErrors(((int)commandBufferStress));

        mtlpp::ComputeCommandEncoder commandEncoderStress = commandBufferStress.ComputeCommandEncoder();
        COMMON_METAL_PARAMS;
        commandEncoderStress.SetBuffer(gpu_Snapshots_pr, 0, 6);
        commandEncoderStress.SetComputePipelineState(computePipelineStateStress);
        commandEncoderStress.DispatchThreadgroups(
            mtlpp::Size(
              (unsigned int)ceil((float)(INHOST(N1)+1) / 4),
              (unsigned int)ceil((float)(INHOST(N2)+1) / 4),
              (unsigned int)ceil((float)(INHOST(N3)+1) / 4)),
            mtlpp::Size(4, 4, 4));
        commandEncoderStress.EndEncoding();

        mtlpp::BlitCommandEncoder blitCommandEncoderStress = commandBufferStress.BlitCommandEncoder();
        blitCommandEncoderStress.EndEncoding();
        commandBufferStress.Commit();
        commandBufferStress.WaitUntilCompleted();

        mtlpp::CommandBuffer commandBufferParticle = commandQueue.CommandBuffer();
        mxcheckGPUErrors(((int)commandBufferParticle));

        mtlpp::ComputeCommandEncoder commandEncoderParticle = commandBufferParticle.ComputeCommandEncoder();
        COMMON_METAL_PARAMS;
        commandEncoderParticle.SetComputePipelineState(computePipelineStateParticle);
        commandEncoderParticle.DispatchThreadgroups(
            mtlpp::Size(
              (unsigned int)ceil((float)(INHOST(N1)+1) / 4),
              (unsigned int)ceil((float)(INHOST(N2)+1) / 4),
              (unsigned int)ceil((float)(INHOST(N3)+1) / 4)),
            mtlpp::Size(4, 4, 4));
        commandEncoderParticle.EndEncoding();

        mtlpp::BlitCommandEncoder blitCommandEncoderParticle = commandBufferParticle.BlitCommandEncoder();
        blitCommandEncoderParticle.EndEncoding();
        commandBufferParticle.Commit();
        commandBufferParticle.WaitUntilCompleted();

#endif

   // Snapshots
		if (INHOST(CurrSnap) <NumberSnapshots)
			if(INHOST(nStep)==SnapshotsPos_pr[INHOST(CurrSnap)]-1)
			{
  #if defined(CUDA)
				SnapShot<<<dimGridSnap,dimBlockSnap,0,streams[nCurStream]>>>(INHOST(SelK),gpu_Snapshots_pr,gpu_Sigma_xx_pr,gpu_Sigma_yy_pr,gpu_Sigma_zz_pr,INHOST(CurrSnap));
				
  #endif
  #if defined(OPENCL)
        int selfSlice=INHOST(SelK);
        mxcheckGPUErrors(clSetKernelArg(SnapShot, 0, sizeof(unsigned int), &selfSlice));
        mxcheckGPUErrors(clSetKernelArg(SnapShot, 5, sizeof(unsigned int), &INHOST(CurrSnap)));

        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, SnapShot, 2, NULL, global_stress_particle, NULL, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));
  #endif
  #if defined(METAL)
        InitSymbol(CurrSnap,unsigned int,G_INT);
        mtlpp::CommandBuffer commandBufferSnapShot = commandQueue.CommandBuffer();
        mxcheckGPUErrors(((int)commandBufferSnapShot));

        mtlpp::ComputeCommandEncoder commandEncoderSnapShot = commandBufferSnapShot.ComputeCommandEncoder();
        COMMON_METAL_PARAMS;
        commandEncoderSnapShot.SetBuffer(gpu_Snapshots_pr, 0, 6);
        commandEncoderSnapShot.SetComputePipelineState(computePipelineStateSnapShot);
        commandEncoderSnapShot.DispatchThreadgroups(
            mtlpp::Size(
              (unsigned int)ceil((float)(INHOST(N1)+1) / 8),
              (unsigned int)ceil((float)(INHOST(N2)+1) / 8),
              1),
            mtlpp::Size(8, 8,1));
        commandEncoderSnapShot.EndEncoding();

        mtlpp::BlitCommandEncoder blitCommandEncoderSnapShot = commandBufferSnapShot.BlitCommandEncoder();
        blitCommandEncoderSnapShot.EndEncoding();
        commandBufferSnapShot.Commit();
        commandBufferSnapShot.WaitUntilCompleted();
  #endif

				INHOST(CurrSnap)++;
			}

		//~ //Finally, the sensors
    if (((((_PT)INHOST(nStep)) % ((_PT)INHOST(SensorSubSampling)))==0) && ((((_PT)INHOST(nStep)) / ((_PT)INHOST(SensorSubSampling)))>=((_PT)INHOST(SensorStart))))
		{
#if defined(CUDA)
      SensorsKernel<<<dimGridSensors,dimBlockSensors,0,streams[nCurStream]>>>(pGPU,gpu_IndexSensorMap_pr,INHOST(nStep));
#endif
#if defined(OPENCL)
      mxcheckGPUErrors(clSetKernelArg(SensorsKernel, 56, sizeof(unsigned int), &INHOST(nStep)));
      mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, SensorsKernel, 1, NULL, global_sensors, NULL, 0, NULL, NULL));
      mxcheckGPUErrors(clFinish(commands));
#endif
#if defined(METAL)
      mtlpp::CommandBuffer commandBufferSensors = commandQueue.CommandBuffer();
      mxcheckGPUErrors(((int)commandBufferSensors));

      mtlpp::ComputeCommandEncoder commandEncoderSensors = commandBufferSensors.ComputeCommandEncoder();
      COMMON_METAL_PARAMS;
      commandEncoderSensors.SetComputePipelineState(computePipelineStateSensors);
      commandEncoderSensors.DispatchThreadgroups(
          mtlpp::Size(
            (unsigned int)ceil((float)(INHOST(NumberSensors)) / 32),
            1,
            1),
          mtlpp::Size(32, 1, 1));
      commandEncoderSensors.EndEncoding();

      mtlpp::BlitCommandEncoder blitCommandEncoderSensors = commandBufferSensors.BlitCommandEncoder();
      if (INHOST(nStep)==(INHOST(TimeSteps)-1))
      {  //just in the very last step we synchronize
        blitCommandEncoderSensors.Synchronize(_MEX_BUFFER);
      }
      blitCommandEncoderSensors.EndEncoding();
      commandBufferSensors.Commit();
      commandBufferSensors.WaitUntilCompleted();
#endif
    }
    INHOST(nStep)++;
    nCurStream++;
  #if defined(CUDA)
    } //this one closes the bracket for the streams
    for(unsigned int nSyncStream=0;nSyncStream<nCurStream;nSyncStream++)
        cudaStreamSynchronize(streams[nSyncStream]);
   #endif
	}




	//DONE, just to copy to the host the results
  #if defined(CUDA) || defined(OPENCL)
	CopyFromGPUToMX3(SensorOutput,mexType);
  CopyFromGPUToMX3(SqrAcc,mexType);
  #else
  CopyFromGPUToMX4(SensorOutput,mexType);
  CopyFromGPUToMX4(SqrAcc,mexType);
  #endif

	CopyFromGPUToMX(Vx,mexType);
  CopyFromGPUToMX(Vy,mexType);
  CopyFromGPUToMX(Vz,mexType);
  CopyFromGPUToMX(Sigma_xx,mexType);
  CopyFromGPUToMX(Sigma_yy,mexType);
  CopyFromGPUToMX(Sigma_xy,mexType);
  CopyFromGPUToMX(Sigma_xz,mexType);
  CopyFromGPUToMX(Sigma_yz,mexType);
  CopyFromGPUToMX(Pressure,mexType);

  {
    _PT i,j,k,CurZone;
  #pragma omp parallel for private(j,i,CurZone)
  for(k=0; k<((_PT)INHOST(N3)); k++)
    for(j=0; j<((_PT)INHOST(N2)); j++)
      for(i=0; i<((_PT)INHOST(N1)); i++)
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
        ASSIGN_RES(Pressure);
      }
 }

#if defined(CUDA)
    for (unsigned n =0;n<TOTAL_streams;n++)
		    mxcheckGPUErrors(cudaStreamDestroy ( streams[n])) ;
#endif

	CopyFromGPUToMX3(Snapshots,mexType);

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
  free(Pressure_pr);

	ownGPUFree(SensorOutput);
	ownGPUFree(V_x_x);
  ownGPUFree(V_y_x);
  ownGPUFree(V_z_x);
  ownGPUFree(V_x_y);
  ownGPUFree(V_y_y);
  ownGPUFree(V_z_y);
  ownGPUFree(V_x_z);
  ownGPUFree(V_y_z);
  ownGPUFree(V_z_z);
  ownGPUFree(Sigma_x_xx);
  ownGPUFree(Sigma_y_xx);
  ownGPUFree(Sigma_z_xx);
  ownGPUFree(Sigma_x_yy);
  ownGPUFree(Sigma_y_yy);
  ownGPUFree(Sigma_z_yy);
  ownGPUFree(Sigma_x_zz);
  ownGPUFree(Sigma_y_zz);
  ownGPUFree(Sigma_z_zz);
  ownGPUFree(Sigma_x_xy);
  ownGPUFree(Sigma_y_xy);
  ownGPUFree(Sigma_x_xz);
  ownGPUFree(Sigma_z_xz);
  ownGPUFree(Sigma_y_yz);
  ownGPUFree(Sigma_z_yz);
  ownGPUFree(LambdaMiuMatOverH);
	ownGPUFree(LambdaMatOverH);
	ownGPUFree(MiuMatOverH);
	ownGPUFree(TauLong);
	ownGPUFree(OneOverTauSigma);
	ownGPUFree(TauShear);
	ownGPUFree(InvRhoMatH);
	ownGPUFree(IndexSensorMap);
	ownGPUFree(SourceFunctions);
	ownGPUFree(SourceMap);
  ownGPUFree(Ox);
  ownGPUFree(Oy);
  ownGPUFree(Oz);
	ownGPUFree(MaterialMap);
  ownGPUFree(Vx);
  ownGPUFree(Vy);
  ownGPUFree(Vz);
  ownGPUFree(Sigma_xx);
  ownGPUFree(Sigma_yy);
  ownGPUFree(Sigma_zz);
  ownGPUFree(Sigma_xy);
  ownGPUFree(Sigma_xz);
  ownGPUFree(Sigma_yz);
  ownGPUFree(Pressure);
  ownGPUFree(Rxx);
  ownGPUFree(Ryy);
  ownGPUFree(Rzz);
  ownGPUFree(Rxy);
  ownGPUFree(Rxz);
  ownGPUFree(Ryz);
  ownGPUFree(Snapshots);
  ownGPUFree(SqrAcc);

#if defined(CUDA)
	mxcheckGPUErrors(cudaMemGetInfo( &free_byte, &total_byte ));
    free_db = (double)free_byte ;
    total_db = (double)total_byte ;
    used_db = total_db - free_db ;

    PRINTF("GPU memory remaining (free should be equal to total): used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
#endif
#if defined(OPENCL)
    clReleaseProgram(program);
    clReleaseKernel(StressKernel);
    clReleaseKernel(ParticleKernel);
    clReleaseKernel(SensorsKernel);
    clReleaseKernel(SnapShot);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
#endif
PRINTF("Number of unfreed allocs (it should be 0):%i\n",NumberAlloc);
