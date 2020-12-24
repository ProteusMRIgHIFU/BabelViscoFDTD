
ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -std=c99

LIBS = -lOpenCL -fopenmp -lm

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL -lm
endif


pi_ocl: pi_ocl.c wtime.c device_info.c
	$(CC) $^ $(CCFLAGS) $(LIBS)  -o $@


clean:
	rm -f pi_ocl
