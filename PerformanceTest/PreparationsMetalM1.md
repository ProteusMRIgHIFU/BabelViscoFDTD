# Steps to run test
## Prepare environment 
1. Prepare a Native ARM64 Python environment using `miniforge`, for example (assuming having brew installed):
    * `brew install miniforge`
1. Create a test Python environment with some basic packages
    * `conda create --name fdtdtest python==3.9 scipy numpy matplotlib h5py`
1. Activate environment 
    * `conda activate fdtdtest`
1. Install BabelViscoFDTD using regular `pip`. It will install latest version based on mini-kernels approach for Metal and extra dependencies
    * `pip install BabelViscoFDTD`

# Run tests with SimpleBenchmark.py
The script is simple enough to test differences of performance between Metal and OpenCL
```
usage: python SimpleBenchmark.py [options]

Run a quick simulation for benchmark Metal vs OpenCL

positional arguments:
  GPU_ID                Substring for GPU id such as "M1"
  {Metal,OpenCL}        Backend to test, it must be one of the available options shown on the left

optional arguments:
  -h, --help            show this help message and exit
  --EnableMetalCapture, --no-EnableMetalCapture
                        GPU Capture will be done when running Metal backend, be sure of running with environment variable
                        METAL_CAPTURE_ENABLED=1 (default: None)
  --ShortRun, --no-ShortRun
                        Enable short simulation (1/10 of normal length), useful when doing Metal capture to reduce size of
                        capture file (default: None)
```
## OpenCL test
* `python SimpleBenchmark.py M1 OpenCL ` 

In a M1 Max, it should report a wall-time around 3s
```
...
time to do low level calculations 3.1081128120422363
```
## Metal test
* `python SimpleBenchmark.py M1 Metal ` 

In in a M1 Max, it should report a wall-time around 8s
```
...
Time to run low level FDTDStaggered_3D = 8.194736957550049
```
## Metal test with Metal capture
* `METAL_CAPTURE_ENABLED=1 python SimpleBenchmark.py M1 Metal --EnableMetalCapture --ShortRun`

The `--ShortRun` option indicates the number of iterations of kernel execution will be 1/10 of the original. This helps to reduce the size of the GPU capture file. If `--ShortRun` is not specified, the capture will take ~ 45 GB of space and 30s to execute.
The Metal capture is a bit simplistic, it wil always save in a directory named `frameCapture.gputrace`. If you want to repeat the capture, you need to rename/delete the previous one.