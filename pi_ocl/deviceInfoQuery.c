// ================================================================================================
// A simple script to query some of the basic information about the OpenCL devices available in
// your system
//
// Author: Sivagnanam Namasivayamurthy
//
// ================================================================================================

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

void check_cl_error(cl_int err_num, char* msg) {
  if(err_num != CL_SUCCESS) {
    printf("[Error] OpenCL error code: %d in %s \n", err_num, msg);
    exit(EXIT_FAILURE);
  }
}

int main(void) {

  printf("\nStarting OpenCL device query: \n");
  printf("------------------------------\n");
  cl_int err_num;
  char str_buffer[1024];
  cl_uint num_platforms_available;

  // Get the number of OpenCL capable platforms available
  err_num = clGetPlatformIDs(0, NULL, &num_platforms_available);
  check_cl_error(err_num, "clGetPlatformIDs: Getting number of available platforms");

  // Exit if no OpenCL capable platform found
  if(num_platforms_available == 0){
    printf("No OpenCL capable platforms found ! \n");
    return EXIT_FAILURE;
  } else {
    printf("\n Number of OpenCL capable platforms available: %d \n", num_platforms_available);
    printf("--------------------------------------------------\n\n");
  }

  // Create a list for storing the platform id's
  cl_platform_id cl_platforms[num_platforms_available];

  err_num = clGetPlatformIDs(num_platforms_available, cl_platforms, NULL);
  check_cl_error(err_num, "clGetPlatformIDs: Getting available platform id's");

  // Get attributes of each platform available
  for(int platform_idx = 0; platform_idx < num_platforms_available; platform_idx++) {
    printf("\t Platform ID: %d \n", platform_idx);
    printf("\t ----------------\n\n");

    // Get platform name
    err_num = clGetPlatformInfo(cl_platforms[platform_idx], CL_PLATFORM_NAME, sizeof(str_buffer), &str_buffer, NULL);
    check_cl_error(err_num, "clGetPlatformInfo: Getting platform name");
    printf("\t\t [Platform %d] CL_PLATFORM_NAME: %s\n", platform_idx, str_buffer);

    // Get platform vendor
    err_num = clGetPlatformInfo(cl_platforms[platform_idx], CL_PLATFORM_VENDOR, sizeof(str_buffer), &str_buffer, NULL);
    check_cl_error(err_num, "clGetPlatformInfo: Getting platform vendor");
    printf("\t\t [Platform %d] CL_PLATFORM_VENDOR: %s\n", platform_idx, str_buffer);

    // Get platform OpenCL version
    err_num = clGetPlatformInfo(cl_platforms[platform_idx], CL_PLATFORM_VERSION, sizeof(str_buffer), &str_buffer, NULL);
    check_cl_error(err_num, "clGetPlatformInfo: Getting platform version");
    printf("\t\t [Platform %d] CL_PLATFORM_VERSION: %s\n", platform_idx, str_buffer);

    // Get platform OpenCL profile
    err_num = clGetPlatformInfo(cl_platforms[platform_idx], CL_PLATFORM_PROFILE, sizeof(str_buffer), &str_buffer, NULL);
    check_cl_error(err_num, "clGetPlatformInfo: Getting platform profile");
    printf("\t\t [Platform %d] CL_PLATFORM_PROFILE: %s\n", platform_idx, str_buffer);

    // Get platform OpenCL supported extensions
    err_num = clGetPlatformInfo(cl_platforms[platform_idx], CL_PLATFORM_EXTENSIONS, sizeof(str_buffer), &str_buffer, NULL);
    check_cl_error(err_num, "clGetPlatformInfo: Getting platform supported extensions");
    printf("\t\t [Platform %d] CL_PLATFORM_EXTENSIONS: %s\n", platform_idx, str_buffer);

    // Get the number of OpenCL supported device available in this platform
    cl_uint num_devices_available;
    err_num = clGetDeviceIDs(cl_platforms[platform_idx], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices_available);
    check_cl_error(err_num, "clGetDeviceIDs: Get number of OpenCL supported devices available");
    printf("\n\t\t [Platform %d] Number of devices available: %d \n", platform_idx, num_devices_available);
    printf("\t\t ---------------------------------------------\n\n");

    cl_device_id cl_devices[num_devices_available];
    err_num = clGetDeviceIDs(cl_platforms[platform_idx], CL_DEVICE_TYPE_ALL, num_devices_available, cl_devices, NULL);
    check_cl_error(err_num, "clGetDeviceIDs: Getting available OpenCL capable device id's");

    // Get attributes of each device
    for(int device_idx = 0; device_idx < num_devices_available; device_idx++) {

      printf("\t\t\t [Platform %d] Device ID: %d\n", platform_idx, device_idx);
      printf("\t\t\t ---------------------------\n\n");

      // Get device name
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device name");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_NAME: %s\n", platform_idx, device_idx,str_buffer);

      // Get device hardware version
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_VERSION, sizeof(str_buffer), &str_buffer, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device hardware version");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_VERSION: %s\n", platform_idx, device_idx,str_buffer);

      // Get device software version
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DRIVER_VERSION, sizeof(str_buffer), &str_buffer, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device software version");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DRIVER_VERSION: %s\n", platform_idx, device_idx,str_buffer);

      // Get device OpenCL C version
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_OPENCL_C_VERSION, sizeof(str_buffer), &str_buffer, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device OpenCL C version");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_OPENCL_C_VERSION: %s\n", platform_idx, device_idx,str_buffer);

      // Get device max clock frequency
      cl_uint max_clock_freq;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_freq), &max_clock_freq, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device max clock frequency");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_CLOCK_FREQUENCY: %d MHz\n", platform_idx, device_idx, max_clock_freq);

      // Get device max compute units available
      cl_uint max_compute_units_available;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device max compute units available");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", platform_idx, device_idx, max_compute_units_available);

      // Get device global mem size
      cl_ulong global_mem_size;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device global mem size");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_GLOBAL_MEM_SIZE: %llu MB\n", platform_idx, device_idx, (_PT)global_mem_size/(1024*1024));

      // Get device max compute units available
      cl_ulong max_mem_alloc_size;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device max mem alloc size");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_MEM_ALLOC_SIZE: %llu MB\n", platform_idx, device_idx, (_PT)max_mem_alloc_size/(1024*1024));

      // Get device local mem size
      cl_ulong local_mem_size;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device local mem size");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_LOCAL_MEM_SIZE: %llu KB\n", platform_idx, device_idx, (_PT)local_mem_size/1024);

      // Get device max work group size
      size_t max_work_group_size;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device max work group size");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_WORK_GROUP_SIZE: %ld\n", platform_idx, device_idx, (long int)max_work_group_size);

      // Get device max work item dim
      cl_uint max_work_item_dims;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device max work item dimension");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %d\n", platform_idx, device_idx, max_work_item_dims);

      // Get device max work item sizes in each dimension
      size_t work_item_sizes[max_work_item_dims];
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_sizes), &work_item_sizes, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device max work items dimension");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_WORK_ITEM_SIZES: ", platform_idx, device_idx);
      for (size_t work_item_dim = 0; work_item_dim < max_work_item_dims; work_item_dim++) {
                printf("%ld ", (long int)work_item_sizes[work_item_dim]);
      }
      printf("\n");

      // Get device image support
      cl_bool image_support;
      err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
      check_cl_error(err_num, "clGetDeviceInfo: Getting device image support");
      printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_IMAGE_SUPPORT: %u (%s)\n", platform_idx, device_idx, image_support, image_support? "Available" : "Not available");

      if(image_support) {

        size_t image_size;

        // Get device image 2d max width
        err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(image_size), &image_size, NULL);
        check_cl_error(err_num, "clGetDeviceInfo: Getting device image max 2d width");
        printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_IMAGE2D_MAX_WIDTH: %ld\n", platform_idx, device_idx, (long int)image_size);

        // Get device image 2d max height
        err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(image_size), &image_size, NULL);
        check_cl_error(err_num, "clGetDeviceInfo: Getting device image max 2d width");
        printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_IMAGE2D_MAX_HEIGHT: %ld\n", platform_idx, device_idx, (long int)image_size);

        // Get device image 3d max width
        err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(image_size), &image_size, NULL);
        check_cl_error(err_num, "clGetDeviceInfo: Getting device image max 3d width");
        printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_IMAGE3D_MAX_WIDTH: %ld\n", platform_idx, device_idx, (long int)image_size);

        // Get device image 3d max height
        err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(image_size), &image_size, NULL);
        check_cl_error(err_num, "clGetDeviceInfo: Getting device image max 3d height");
        printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_IMAGE3D_MAX_HEIGHT: %ld\n", platform_idx, device_idx, (long int)image_size);

        // Get device image 2d max depth
        err_num = clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(image_size), &image_size, NULL);
        check_cl_error(err_num, "clGetDeviceInfo: Getting device image max 3d depth");
        printf("\t\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_IMAGE3D_MAX_DEPTH: %ld\n", platform_idx, device_idx, (long int)image_size);

      }
      printf("\n\n");
    }
  }


  return 0;
}
