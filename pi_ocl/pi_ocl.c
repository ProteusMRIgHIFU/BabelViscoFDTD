/*------------------------------------------------------------------------------
 *
 * Name:       pi_ocl.c
 *
 * Purpose:    Numeric integration to estimate pi
 *
 * HISTORY:    Written by Tim Mattson, May 2010
 *             Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
 *             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
 *             Ported back to C by Tom Deakin, July 2013
 *             Updated by Tom Deakin, October 2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#include "device_picker.h"


extern double wtime();       // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------
char * getKernelSource(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) +1;
    rewind(file);

    char *source = (char *)calloc(sizeof(char), len);
    if (!source)
    {
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        exit(EXIT_FAILURE);
    }
    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
}


//------------------------------------------------------------------------------

#define INSTEPS (512*512*512)
#define ITERS (262144)

//------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int in_nsteps = INSTEPS;    // default number of steps (updated later to device preferable)
    int niters = ITERS;         // number of iterations
    int nsteps;
    float step_size;
    size_t nwork_groups;
    size_t max_size, work_group_size = 8;
    float pi_res;

    cl_int err;
    cl_device_id        device;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        kernel_stress,kernel_particle,kernel_sensor,kernel_snapshot;     // compute kernel
    size_t binary_size;
    char * binary;
    char * inputname =0;
    char * outputname =0;
    FILE * f;
    // Set up OpenCL context, queue, kernel, etc.
    cl_uint deviceIndex = 0;
    parseArguments(argc, argv, &deviceIndex,&inputname,&outputname);

    // Get list of devices
    cl_device_id devices[MAX_DEVICES];
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices)
    {
      printf("Invalid device index (try '--list')\n");
      return EXIT_FAILURE;
    }

    if (inputname==0)
    {
        printf("need to specify input file with --input\n");
      return EXIT_FAILURE;
    }

    if (outputname==0)
    {
        printf("need to specify output file with --output\n");
      return EXIT_FAILURE;
    }

    char *kernelsource = getKernelSource(inputname);             // Kernel source


    device = devices[deviceIndex];

    char name[MAX_INFO_STRING];
    getDeviceName(device, name);
    printf("\nUsing OpenCL device: %s\n", name);



    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");
    // Create a command queue
    commands = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelsource, NULL, &err);
    checkError(err, "Creating program");
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[600048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("buffer=%s\n", buffer);
        return EXIT_FAILURE;
    }

    kernel_stress = clCreateKernel(program, "StressKernel", &err);
    checkError(err, "Creating stress kernel");

    kernel_particle = clCreateKernel(program, "ParticleKernel", &err);
    checkError(err, "Creating particle kernel");


    kernel_snapshot = clCreateKernel(program, "SnapShot", &err);
    checkError(err, "Creating snapshot kernel");


    kernel_sensor = clCreateKernel(program, "SensorsKernel", &err);
    checkError(err, "Creating sensors kernel");



   clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
       printf("bion sixe= %ld\n",binary_size);
   binary = malloc(binary_size+1);
   clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, &binary, NULL);
   printf("after getting binaries\n");
   f = fopen(outputname, "w");
   fwrite(binary, binary_size, 1, f);
   fclose(f);


    clReleaseProgram(program);
    clReleaseKernel(kernel_stress);
    clReleaseKernel(kernel_particle);
    clReleaseKernel(kernel_snapshot);
    clReleaseKernel(kernel_sensor);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(kernelsource);
}
