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
    cl_platform_id * Platform = (cl_platform_id*) malloc(numPlatforms*sizeof(cl_platform_id));
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
    size_t size_bits;
    cl_uint address_bits;
    clGetDeviceInfo(device_id[SelDevice], CL_DEVICE_ADDRESS_BITS, 0, NULL, &size_bits);
    clGetDeviceInfo(device_id[SelDevice], CL_DEVICE_ADDRESS_BITS, size_bits, &address_bits, NULL);
    PRINTF("size: %lu , bits: %u\n", size_bits, address_bits);

    if (address_bits==32)
    {
      PRINTF("********************************\n");
      PRINTF("WARNING - OpenCL driver only supports 32 bits, simulations only will be\n");
      PRINTF("WARNING - useful for domains that uses less than 4 GB in total memory\n");
      PRINTF("WARNING - Program may crash without an easy way to catch the error\n");
      PRINTF("WARNING - Consider using CUDA backend if NVIDIA GPU in Win or Linux, or METAL backend in MacOS\n");
    }

    context = clCreateContext(0, 1, &device_id[SelDevice], NULL, NULL, &err);
    mxcheckGPUErrors(err);

    // Create a command queue

    commands = clCreateCommandQueue(context, device_id[SelDevice], 0, &err);
    mxcheckGPUErrors(err);
    

#endif

#ifdef METAL
    _PT _c_mex_type[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    _PT  _c_uint_type = 0;
    _PT  HOST_INDEX_MEX[LENGTH_INDEX_MEX][2]; //need to encode 64 bits numbers in 32 arrays...

    _PT  HOST_INDEX_UINT[LENGTH_INDEX_UINT][2];

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
        PRINTF("Metal device available: %i %s\n",_n,AllDev[_n].GetName().GetCStr());
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

    ns::Error error;
    ns::String PathToLib(kernbinfile_pr);
    mtlpp::Library library = device.NewLibrary(PathToLib, &error);
    if (((int)library)==0 )
    {
        PRINTF("GetLocalizedDescription = %s\n",error.GetLocalizedDescription().GetCStr());
        PRINTF("GetLocalizedFailureReason = %s\n",error.GetLocalizedFailureReason().GetCStr());
        PRINTF("GetLocalizedRecoverySuggestion = %s\n",error.GetLocalizedRecoverySuggestion().GetCStr());
        PRINTF("GetLocalizedRecoveryOptions = %s\n",error.GetLocalizedRecoveryOptions().GetCStr());
        PRINTF("GetHelpAnchor = %s\n",error.GetHelpAnchor().GetCStr());
        PRINTF("GetCode = %i\n",error.GetCode());
        ERROR_STRING("Error loading Metal library")
    }
    mxcheckGPUErrors(((int)library));

    PRINTF("After compiling code \n");


    GET_KERNEL_STRESS_FUNCTION(PML_1) 
    GET_KERNEL_STRESS_FUNCTION(PML_2) 
    GET_KERNEL_STRESS_FUNCTION(PML_3) 
    GET_KERNEL_STRESS_FUNCTION(PML_4) 
    GET_KERNEL_STRESS_FUNCTION(PML_5) 
    GET_KERNEL_STRESS_FUNCTION(PML_6)    
    GET_KERNEL_STRESS_FUNCTION(MAIN_1)     

    GET_KERNEL_PARTICLE_FUNCTION(PML_1) 
    GET_KERNEL_PARTICLE_FUNCTION(PML_2) 
    GET_KERNEL_PARTICLE_FUNCTION(PML_3) 
    GET_KERNEL_PARTICLE_FUNCTION(PML_4) 
    GET_KERNEL_PARTICLE_FUNCTION(PML_5) 
    GET_KERNEL_PARTICLE_FUNCTION(PML_6)    
    GET_KERNEL_PARTICLE_FUNCTION(MAIN_1)  

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

  //PRINTF("%s",BUFFER_FOR_GPU_CODE);
  //   size_t szKernelLength = strlen(BUFFER_FOR_GPU_CODE);
  //   program = clCreateProgramWithSource(context, 1, (const char **) & BUFFER_FOR_GPU_CODE, &szKernelLength, &err);
  //   mxcheckGPUErrors(err);

    char scmd [8000];
    snprintf(scmd,8000,"\"%s\" --input \"%s\" --output \"%s\" --device %i",PI_OCL_PATH_pr,
                    kernelfile_pr,kernbinfile_pr,
                    SelDevice);
    PRINTF("compiling kernel with \"%s\"\n",scmd)
    if (system(scmd)!=0)
    {
      ERROR_STRING("Error when trying to compile program");
    }

    char * binary;
    size_t binary_size;
    long l_szie;
    cl_int binary_status;
    binary = common_read_file(kernbinfile_pr, &l_szie);
    if (binary==NULL)
    {
         PRINTF("problem when trying to open kernel binary [%s]\n",kernbinfile_pr);
         ERROR_STRING("Stopping execution")
    }
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


  mtlpp::Buffer _MEX_BUFFER[12];
  for (_PT ii=0;ii<12;ii++)
  {
     PRINTF("Allocating Buffer %i with %lu float entries\n",ii,_c_mex_type[ii]);
    _MEX_BUFFER[ii]= device.NewBuffer(sizeof(mexType) *_c_mex_type[ii],
            mtlpp::ResourceOptions::StorageModeManaged);
    mxcheckGPUErrors(((int)_MEX_BUFFER[ii]));
    if (_MEX_BUFFER[ii].GetLength() != sizeof(mexType) *_c_mex_type[ii])
    {
        PRINTF("ERROR, size of buffer is not what is expected %lu, %lu\n",_MEX_BUFFER[ii].GetLength(),sizeof(mexType) *_c_mex_type[ii]);
        ERROR_STRING("Stopping simulation");
    }
  }

  mtlpp::Buffer _UINT_BUFFER = device.NewBuffer(sizeof(unsigned int) *_c_uint_type,
            mtlpp::ResourceOptions::StorageModeManaged);
  mxcheckGPUErrors(((int)_UINT_BUFFER));

  mtlpp::Buffer _INDEX_MEX = device.NewBuffer(sizeof(unsigned int) *
            LENGTH_INDEX_MEX*2,
            mtlpp::ResourceOptions::StorageModeManaged);
  mxcheckGPUErrors(((int)_INDEX_MEX));

  mtlpp::Buffer _INDEX_UINT = device.NewBuffer(sizeof(unsigned int) *
            LENGTH_INDEX_UINT*2,
            mtlpp::ResourceOptions::StorageModeManaged);
  mxcheckGPUErrors(((int)_INDEX_UINT));

  {
      unsigned int * inData = static_cast<unsigned int *>(_INDEX_MEX.GetContents());
      for (uint32_t j=0; j<LENGTH_INDEX_MEX; j++)
      {
          inData[j*2] =  (unsigned int) (0xFFFFFFFF & HOST_INDEX_MEX[j][0]);
          inData[j*2+1] = (unsigned int) (HOST_INDEX_MEX[j][0]>>32);
          
          
      }
      _INDEX_MEX.DidModify(ns::Range(0, sizeof(unsigned int) * LENGTH_INDEX_MEX*2));
      
  }

  {
      unsigned int * inData = static_cast< unsigned int *>(_INDEX_UINT.GetContents());
      for (uint32_t j=0; j<LENGTH_INDEX_UINT; j++)
      {
           inData[j*2] =  (unsigned int) (0xFFFFFFFF & HOST_INDEX_UINT[j][0]);
           inData[j*2+1] = (unsigned int) (HOST_INDEX_UINT[j][0]>>32);
          
      }
      _INDEX_UINT.DidModify(ns::Range(0, sizeof(unsigned int) * LENGTH_INDEX_UINT*2));
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
  _PT totalfloat=0;
  for (_PT ii=0;ii<12;ii++)
  {
    _MEX_BUFFER[ii].DidModify(ns::Range(0,sizeof(mexType) *_c_mex_type[ii]));
    totalfloat+=_c_mex_type[ii];
  }

  _UINT_BUFFER.DidModify(ns::Range(0,sizeof(unsigned int) *_c_uint_type));
  
  PRINTF("Total float entries %lu and int entries %lu\n",totalfloat,_c_uint_type);

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

  int minBlockSize;
  int minGridSize;

  CUDA_GRID_BLOC_CALC_MAIN(MAIN_1_StressKernel);
  CUDA_GRID_BLOC_CALC_MAIN(MAIN_1_ParticleKernel);

  #if defined(USE_MINI_KERNELS_CUDA)
  CUDA_GRID_BLOC_CALC_PML(Stress);
  CUDA_GRID_BLOC_CALC_PML(Particle);
  #endif

  //We handle the case the user wants to specify manually the computing grid sizes
  if (ManualLocalSize_pr[0] != -1)
  {
      dimBlockMAIN_1_StressKernel.x=(unsigned int)ManualLocalSize_pr[0];
      dimBlockMAIN_1_StressKernel.y=(unsigned int)ManualLocalSize_pr[1];
      dimBlockMAIN_1_StressKernel.z=(unsigned int)ManualLocalSize_pr[2];
      dimBlockMAIN_1_ParticleKernel.x=(unsigned int)ManualLocalSize_pr[0];
      dimBlockMAIN_1_ParticleKernel.y=(unsigned int)ManualLocalSize_pr[1];
      dimBlockMAIN_1_ParticleKernel.z=(unsigned int)ManualLocalSize_pr[2];
  }
  if (ManualGroupSize_pr[0] != -1)
  {
      dimGridMAIN_1_StressKernel.x  = (unsigned int)ManualGroupSize_pr[0];
      dimGridMAIN_1_StressKernel.y  = (unsigned int)ManualGroupSize_pr[1];
      dimGridMAIN_1_StressKernel.z  = (unsigned int)ManualGroupSize_pr[2];
      dimGridMAIN_1_ParticleKernel.x  = (unsigned int)ManualGroupSize_pr[0];
      dimGridMAIN_1_ParticleKernel.y  = (unsigned int)ManualGroupSize_pr[1];
      dimGridMAIN_1_ParticleKernel.z  = (unsigned int)ManualGroupSize_pr[2];
  }
  
  dim3 dimBlockSnap;
  dim3 dimGridSnap;
  mxcheckGPUErrors(cudaOccupancyMaxPotentialBlockSize( &minGridSize, &minBlockSize,
                                  SnapShot, 0, 0));
  PRINTF("N1:minGridSize and Blocksize from API for SnapShot = %i and %i\n",minGridSize,minBlockSize);
  dimBlockSnap.x=8;
  dimBlockSnap.y=(unsigned int)floor(minBlockSize/(dimBlockSnap.x));

  dimGridSnap.x  = (unsigned int)ceil((float)(INHOST(N1)+1) / dimBlockSnap.x);
  dimGridSnap.y  = (unsigned int)ceil((float)(INHOST(N2)+1) / dimBlockSnap.y);

  PRINTF(" Snapshot block size to %dx%d\n", dimBlockSnap.x, dimBlockSnap.y);
  PRINTF(" Snapshot grid size to %dx%d\n", dimGridSnap.x, dimGridSnap.y);

  dim3 dimBlockSensors;
  dim3 dimGridSensors;
  mxcheckGPUErrors(cudaOccupancyMaxPotentialBlockSize(  &minGridSize, &minBlockSize,
                                  SensorsKernel, 0, 0));
  PRINTF("minGridSize and Blocksize from API for SensorsKernel = %i and %i\n",minGridSize,minBlockSize);
  dimBlockSensors.x=minBlockSize;
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
  #if defined(USE_MINI_KERNELS_CUDA)
  #define TOTAL_streams 7
  #else
  #define TOTAL_streams 1
  #endif
  cudaStream_t streams[TOTAL_streams];
  for (unsigned n =0;n<TOTAL_streams;n++)
  mxcheckGPUErrors(cudaStreamCreate ( &streams[n])) ;

#endif

#ifdef OPENCL
  //We handle the case the user wants to specify manually the computing grid sizes
  size_t global_stress_particle[3];
  if (ManualGroupSize_pr[0] != -1)
  {
      global_stress_particle[0]=(size_t)ManualGroupSize_pr[0];
      global_stress_particle[1]=(size_t)ManualGroupSize_pr[1];
      global_stress_particle[2]=(size_t)ManualGroupSize_pr[2];
  }
  else
  {
      global_stress_particle[0]=(size_t)INHOST(N1);
      global_stress_particle[1]=(size_t)INHOST(N2);
      global_stress_particle[2]=(size_t)INHOST(N3);
  }
  size_t * local_stress = NULL;
  size_t local_stress_manual[3];
  if (ManualLocalSize_pr[0] != -1)
  {
      local_stress_manual[0]=(size_t)ManualLocalSize_pr[0];
      local_stress_manual[1]=(size_t)ManualLocalSize_pr[1];
      local_stress_manual[2]=(size_t)ManualLocalSize_pr[2];
      local_stress=local_stress_manual;
  }

  PRINTF("global_stress_particle %i %i %i\n",
        global_stress_particle[0],global_stress_particle[1],global_stress_particle[2]);
  if (local_stress!=NULL)
  {
      PRINTF("local_stress %i %i %i\n",
          local_stress[0],local_stress[1],local_stress[2]);
  }
  else{
     PRINTF("local_stress is NULL\n");
  }

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

#ifdef METAL
  unsigned int PML_1_local_stress[3];
  unsigned int PML_1_global_stress[3];
  unsigned int PML_2_local_stress[3];
  unsigned int PML_2_global_stress[3];
  unsigned int PML_3_local_stress[3];
  unsigned int PML_3_global_stress[3];
  unsigned int PML_4_local_stress[3];
  unsigned int PML_4_global_stress[3];
  unsigned int PML_5_local_stress[3];
  unsigned int PML_5_global_stress[3];
  unsigned int PML_6_local_stress[3];
  unsigned int PML_6_global_stress[3];
  unsigned int MAIN_1_local_stress[3];
  unsigned int MAIN_1_global_stress[3];

  unsigned int PML_1_local_particle[3];
  unsigned int PML_1_global_particle[3];
  unsigned int PML_2_local_particle[3];
  unsigned int PML_2_global_particle[3];
  unsigned int PML_3_local_particle[3];
  unsigned int PML_3_global_particle[3];
  unsigned int PML_4_local_particle[3];
  unsigned int PML_4_global_particle[3];
  unsigned int PML_5_local_particle[3];
  unsigned int PML_5_global_particle[3];
  unsigned int PML_6_local_particle[3];
  unsigned int PML_6_global_particle[3];
  unsigned int MAIN_1_local_particle[3];
  unsigned int MAIN_1_global_particle[3];

  if (ManualLocalSize_pr[0] != -1)
  {
      
      SET_USER_LOCAL_STRESS(MAIN_1)
      SET_USER_LOCAL_PARTICLE(MAIN_1)
      
  }
  else
  {
      
      CALC_USER_LOCAL_STRESS(MAIN_1)
      CALC_USER_LOCAL_PARTICLE(MAIN_1)
  }

  CALC_USER_LOCAL_STRESS(PML_1)  
  CALC_USER_LOCAL_STRESS(PML_2)  
  CALC_USER_LOCAL_STRESS(PML_3)   
  CALC_USER_LOCAL_STRESS(PML_4)  
  CALC_USER_LOCAL_STRESS(PML_5)  
  CALC_USER_LOCAL_STRESS(PML_6) 
  CALC_USER_LOCAL_PARTICLE(PML_1)
  CALC_USER_LOCAL_PARTICLE(PML_2)
  CALC_USER_LOCAL_PARTICLE(PML_3)
  CALC_USER_LOCAL_PARTICLE(PML_4)
  CALC_USER_LOCAL_PARTICLE(PML_5)
  CALC_USER_LOCAL_PARTICLE(PML_6)
  
  unsigned int local_sensors[3];
  local_sensors[0]=computePipelineStateSensors.GetMaxTotalThreadsPerThreadgroup();
  local_sensors[1]=1;
  local_sensors[2]=1;


  if (ManualGroupSize_pr[0] != -1)
  {
      SET_USER_GROUP_STRESS(MAIN_1)
      SET_USER_GROUP_PARTICLE(MAIN_1)
  }
  else
  {

      CALC_USER_GROUP_STRESS_MAIN(MAIN_1)
      CALC_USER_GROUP_PARTICLE_MAIN(MAIN_1)

  }

  
  CALC_USER_GROUP_PML(stress);
  CALC_USER_GROUP_PML(particle);

  unsigned int global_sensors[3];
  global_sensors[0]=(unsigned int)ceil((float)(INHOST(NumberSensors)) / (float)local_sensors[0]);
  global_sensors[1]=1;
  global_sensors[2]=1;

  // PRINTF("global_stress %i %i %i, local_stress %i %i %i\n",
  //       global_stress[0],global_stress[1],global_stress[2],
  //       local_stress[0],local_stress[1],local_stress[2]);
  // PRINTF("global_particle %i %i %i, local_particle %i %i %i\n",
  //       global_particle[0],global_particle[1],global_particle[2],
  //       local_particle[0],local_particle[1],local_particle[2]);
  // PRINTF("global_sensors %i %i %i, local_sensors %i %i %i\n",
  //       global_sensors[0],global_sensors[1],global_sensors[2],
  //       local_sensors[0],local_sensors[1],local_sensors[2]);
  
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
  unsigned int SensorEntry=0;

//%%%%%%%%%%%%% MAIN TEMPORAL LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% MAIN TEMPORAL LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%% MAIN TEMPORAL LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%

  while(INHOST(nStep)<INHOST(TimeSteps))
	{
   // PRINTF("nStep %i of %i\n",INHOST(nStep),INHOST(TimeSteps));
#define CUDA_CALL(__KERNEL__,_IDSTREAM)\
   __KERNEL__ <<< dimGrid## __KERNEL__,dimBlock## __KERNEL__,0,streams[_IDSTREAM] >>> (pGPU,INHOST(nStep),INHOST(TypeSource));

#if defined(CUDA)
#if defined(USE_MINI_KERNELS_CUDA)
            CUDA_CALL(PML_1_StressKernel,1);
            CUDA_CALL(PML_2_StressKernel,2);
            CUDA_CALL(PML_3_StressKernel,3);
            CUDA_CALL(PML_4_StressKernel,4);
            CUDA_CALL(PML_5_StressKernel,5);
            CUDA_CALL(PML_6_StressKernel,6);
#endif
            CUDA_CALL(MAIN_1_StressKernel,0);

            for(unsigned int nSyncStream=0;nSyncStream<TOTAL_streams;nSyncStream++)
                mxcheckGPUErrors(cudaStreamSynchronize(streams[nSyncStream]));

#if defined(USE_MINI_KERNELS_CUDA)
            CUDA_CALL(PML_1_ParticleKernel,1);
            CUDA_CALL(PML_2_ParticleKernel,2);
            CUDA_CALL(PML_3_ParticleKernel,3);
            CUDA_CALL(PML_4_ParticleKernel,4);
            CUDA_CALL(PML_5_ParticleKernel,5);
            CUDA_CALL(PML_6_ParticleKernel,6);
#endif
            CUDA_CALL(MAIN_1_ParticleKernel,0);

            for(unsigned int nSyncStream=0;nSyncStream<TOTAL_streams;nSyncStream++)
                mxcheckGPUErrors(cudaStreamSynchronize(streams[nSyncStream]));
        
#endif
#ifdef OPENCL
        int nextSnap=-1;
        if (NumberSnapshots>0)
            nextSnap=SnapshotsPos_pr[INHOST(CurrSnap)]-1;
        mxcheckGPUErrors(clSetKernelArg(StressKernel, 54, sizeof(unsigned int), &INHOST(nStep)));
        mxcheckGPUErrors(clSetKernelArg(StressKernel, 55, sizeof(unsigned int), &INHOST(TypeSource)));
        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 54, sizeof(unsigned int), &INHOST(nStep)));
        mxcheckGPUErrors(clSetKernelArg(ParticleKernel, 55, sizeof(unsigned int), &INHOST(TypeSource)));
        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, StressKernel, 3, NULL, global_stress_particle, local_stress, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));
        mxcheckGPUErrors(clEnqueueNDRangeKernel(commands, ParticleKernel, 3, NULL, global_stress_particle, local_stress, 0, NULL, NULL));
        mxcheckGPUErrors(clFinish(commands));

#endif
#ifdef METAL

        InitSymbol(nStep,unsigned int,G_INT);
        InitSymbol(TypeSource,unsigned int,G_INT);
        InitSymbol(SelK,unsigned int,G_INT);
        mtlpp::CommandBuffer StresscommandBuffer = commandQueue.CommandBuffer();
        mxcheckGPUErrors(((int)StresscommandBuffer));

        ENCODE_STRESS(PML_1)
        ENCODE_STRESS(PML_2)
        ENCODE_STRESS(PML_3)
        ENCODE_STRESS(PML_4)
        ENCODE_STRESS(PML_5)
        ENCODE_STRESS(PML_6)
        ENCODE_STRESS(MAIN_1)
    
        ENCODE_PARTICLE(PML_1)
        ENCODE_PARTICLE(PML_2)
        ENCODE_PARTICLE(PML_3)
        ENCODE_PARTICLE(PML_4)
        ENCODE_PARTICLE(PML_5)
        ENCODE_PARTICLE(PML_6)
        ENCODE_PARTICLE(MAIN_1)
        
        StresscommandBuffer.Commit();
        StresscommandBuffer.WaitUntilCompleted();
        
#endif

   // Snapshots
		if (INHOST(CurrSnap) <NumberSnapshots)
			if(INHOST(nStep)==SnapshotsPos_pr[INHOST(CurrSnap)]-1)
			{
  #if defined(CUDA)
				SnapShot<<<dimGridSnap,dimBlockSnap,0,streams[0]>>>(INHOST(SelK),gpu_Snapshots_pr,gpu_Sigma_xx_pr,gpu_Sigma_yy_pr,gpu_Sigma_zz_pr,INHOST(CurrSnap));
        mxcheckGPUErrors(cudaStreamSynchronize(streams[0]));
				
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
        mtlpp::CommandBuffer StresscommandBuffer = commandQueue.CommandBuffer();
        mtlpp::ComputeCommandEncoder commandEncoderSnapShot = StresscommandBuffer.ComputeCommandEncoder();
        commandEncoderSnapShot.SetBuffer(_CONSTANT_BUFFER_UINT, 0, 0);
        commandEncoderSnapShot.SetBuffer(_CONSTANT_BUFFER_MEX, 0, 1);
        commandEncoderSnapShot.SetBuffer(_INDEX_MEX, 0, 2);
        commandEncoderSnapShot.SetBuffer(_INDEX_UINT, 0, 3);
        commandEncoderSnapShot.SetBuffer(_UINT_BUFFER, 0, 4);
        for (_PT ii=0;ii<12;ii++)
            commandEncoderSnapShot.SetBuffer(_MEX_BUFFER[ii], 0, 5+ii);
        commandEncoderSnapShot.SetBuffer(gpu_Snapshots_pr, 0, 17);
        commandEncoderSnapShot.SetComputePipelineState(computePipelineStateSnapShot);
        commandEncoderSnapShot.DispatchThreadgroups(
            mtlpp::Size(
              (unsigned int)ceil((float)(INHOST(N1)+1) / 8),
              (unsigned int)ceil((float)(INHOST(N2)+1) / 8),
              1),
            mtlpp::Size(8, 8,1));
        commandEncoderSnapShot.EndEncoding();
        
  #endif

				INHOST(CurrSnap)++;
			}

		//~ //Finally, the sensors
    if (((((_PT)INHOST(nStep)) % ((_PT)INHOST(SensorSubSampling)))==0) && 
        ((((_PT)INHOST(nStep)) / ((_PT)INHOST(SensorSubSampling)))>=((_PT)INHOST(SensorStart))) &&
        (SensorEntry < MaxSensorSteps))
		{
      SensorEntry++;
#if defined(CUDA)
      SensorsKernel<<<dimGridSensors,dimBlockSensors,0,streams[0]>>>(pGPU,gpu_IndexSensorMap_pr,INHOST(nStep));
      mxcheckGPUErrors(cudaStreamSynchronize(streams[0]));
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
      commandEncoderSensors.SetBuffer(_CONSTANT_BUFFER_UINT, 0, 0);
      commandEncoderSensors.SetBuffer(_CONSTANT_BUFFER_MEX, 0, 1);
      commandEncoderSensors.SetBuffer(_INDEX_MEX, 0, 2);
      commandEncoderSensors.SetBuffer(_INDEX_UINT, 0, 3);
      commandEncoderSensors.SetBuffer(_UINT_BUFFER, 0, 4);
      for (_PT ii=0;ii<12;ii++)
            commandEncoderSensors.SetBuffer(_MEX_BUFFER[ii], 0, 5+ii);
      commandEncoderSensors.SetComputePipelineState(computePipelineStateSensors);
      commandEncoderSensors.DispatchThreadgroups(
          mtlpp::Size(
            global_sensors[0],
            global_sensors[1],
            global_sensors[2]),
          mtlpp::Size(local_sensors[0],
                      local_sensors[1],
                      local_sensors[2]));
      commandEncoderSensors.EndEncoding();
      commandBufferSensors.Commit();
      commandBufferSensors.WaitUntilCompleted();
#endif
    }
      

    INHOST(nStep)++;
	}
 
  #if defined(METAL)
   
      //#we just synchronize before transferring data back to CPU
      mtlpp::CommandBuffer commandBufferSync = commandQueue.CommandBuffer();
      mxcheckGPUErrors(((int)commandBufferSync));

      mtlpp::BlitCommandEncoder blitCommandEncoderSync = commandBufferSync.BlitCommandEncoder();
      for (_PT ii=0;ii<12;ii++)
          blitCommandEncoderSync.Synchronize(_MEX_BUFFER[ii]);
      blitCommandEncoderSync.EndEncoding();
      commandBufferSync.Commit();
      commandBufferSync.WaitUntilCompleted();
  #endif



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
    free(Platform);
#endif
PRINTF("Number of unfreed allocs (it should be 0):%i\n",NumberAlloc);
