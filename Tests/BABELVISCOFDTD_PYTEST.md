# Running BabelViscoFDTD Pytests
## Description
This document covers the how to run the various types of BabelViscoFDTD tests. For details on setting up pytest for BabelViscoFDTD, see [README.md](README.md)

## Running a BabelViscoFDTD Unit Test
Simplest case, which for the most part can be run simply by calling the test name

**Example:**  
After making changes to BHTE function in RayleighAndBHTE.py file, the following pytest command 
```bash
pytest -k "test_BHTE"
```
will collect and run the following 3 tests to ensure proper operation of BHTE following changes
- test_BHTE_no_source
- test_BHTE_no_source_with_perfusion
- test_BHTE_source_with_perfusion

If multiple changes to RayleighAndBHTE.py were made instead, you could run

```bash
pytest -k "test_RayleighAndBHTE.py"
```
or
```bash
pytest Tests/Unit/BabelViscoFDTD/tools/test_RayleighAndBHTE.py
```
which would run available tests for all the functions inside RayleighAndBHTE.py (assuming the tests exist)

**VERY IMPORTANT**

Typically, pytest will test the code contained within the project folders. However for apple systems, since the `ForwardSimple` function requires compilation ahead of time. We need to first reinstall BabelViscoFDTD, following any changes, in our environment using `pip install .` while in the parent directory of BabelViscoFDTD. 

The tests will then automatically search for BabelViscoFDTD functions in the **environment** rather than local folders. Failure to follow this step will result in tests being performed on older versions of BabelViscoFDTD (pre changes). This is necessary since if we try to use local folder on apple systems, we get an missing library error.

### Parameterized Tests

**Example**  
test_ForwardSimple_low_res_failure is parameterized for gpu backend (OpenCL, CUDA, Metal, MLX) and frequency (e.g. 200000,600000,1000000) Runnning

```bash
pytest -k "test_ForwardSimple_low_res_failure"
```
will collect/run for all combinations of gpu backend and frequency so tests ran would be:
- test_ForwardSimple_low_res_failure[OpenCL-200kHz]
- test_ForwardSimple_low_res_failure[OpenCL-600kHz]
- test_ForwardSimple_low_res_failure[OpenCL-1000kHz]
- test_ForwardSimple_low_res_failure[CUDA-200kHz]
- etc.

Therefore to run a specific combination, simply specify like so
```bash
pytest -k "test_ForwardSimple_low_res_failure[CUDA-200kHz]"
```
or
```bash
pytest -k "test_ForwardSimple_low_res_failure and CUDA and 200kHz"
```

### Running GPU unit tests
Majority of GPU unit tests are parameterized either for spatial step or frequency/ppw pair
- low_res (0.919 spatial step or 200kHz/6PPW)
- medium_res (0.306 spatial step or 600kHz/6PPW)
- high_res (0.184 spatial step or 1000kHz/6PPW)
- stress_res (0.092 spatial step or 1000kHz/12PPW)

where high_res and stress_res are **much slower** to run due to higher resolution sims. Note that tests for GPU function are marked with the "gpu" marker. That is, to collect/run all GPU unit tests, run the following command:

```bash
pytest -k "unit" -m "gpu"
```
To avoid slow gpu tests, run
```bash
pytest -k "unit" -m "gpu and (low_res or med_res)"
```
or
```bash
pytest -k "unit" -m "gpu and not slow"
```
where slow is another marker.

## Running Regression Tests
### Background
Regression Tests for PropagationModel and PropagationModel2D parametrized based on:
- Frequency
- PPW
- GPU Backend
    - OpenCL
    - CUDA
    - Metal
    - MLX

**Note:** Tolerance is another parameter used when comparing BabelViscoFDTD outputs in the test.
- 0% tolerance
- 1% tolerance
- 5% tolerance

### test_PropagationModel_regression
This test compares the results from the current setup to a reference folder containing previously generated outputs. To create a reference folder with data, see the [Running a BabelViscoFDTD Generating_Outputs "Test"](#running-a-babelviscofdtd-generating_outputs-test) below. 

Before running this type of test, ensure the `ref_output_folder_1_name` in your [config.ini](config.ini) file is set to the folder name you want to use as reference. 

**Example**  
You're interested in testing your current setup of BabelViscoFDTD PropagationModel against BabelViscoFDTD v1.0.5 results for then your [config.ini](config.ini) file should be specified something like
```ini
[Paths]
...
ref_output_folder_1 = BabelViscoFDTD_1_0_5
```

You can then call the following command:
```bash
pytest -k "test_PropagationModel_regression"
```

This will also run tests for different tolerances (e.g. 0%,1%,5%). If you're only interested in one type of tolerance, you can run:

```bash
pytest -k "test_PropagationModel_regression" -m "tol_0"
```
or replace tol_0 with tol_1 or tol_5

### test_PropagationModel_two_outputs
This test is similar to above, except it doesn't run your current setup and depends solely on previously generated output folders. 

Before running this type of test, ensure both  `ref_output_folder_1_name` and `ref_output_folder_2_name` in your [config.ini](config.ini) file are set to the folders you want to use. 

**Example**  
You're interested in testing MLX outputs against Metal outputs then your [config.ini](config.ini) file should be specified something like
```ini
[Paths]
...
ref_output_folder_1 = METAL
ref_output_folder_2 = MLX
```

You can then call the following command:
```bash
pytest -k "test_PropagationModel_two_outputs"
```

This will also run tests for different tolerances (e.g. 0%,1%,5%) which you can specify (same as previous section).

## Running a BabelViscoFDTD Generating_Outputs "Test"
The test_generate_PropagationModel_outputs "test" is used not to check BabelViscoFDTD proper functionality, but to leverage pytest to automatically generate outputs for various BabelViscoFDTD parameters to be used later in regression tests.

Before running this type of "test", ensure `gen_output_folder_name` in your [config.ini](config.ini) file is set to the folder where you want to store your outputs. The folder name should be descriptive of what you're generating. Some examples include

Outputs for BabelViscoFDTD v1.0.5
```ini
[Paths]
...
gen_output_folder = BabelViscoFDTD_v1_0_5
```

Outputs for latest BabelViscoFDTD version using only Metal GPU backend
```ini
[Paths]
...
gen_output_folder = Metal
```

Outputs for latest BabelViscoFDTD version using newer version of cupy
```ini
[Paths]
...
gen_output_folder = Cupy_v13_6
```

This "test" can be run with the following command:
```bash
pytest -k "test_generate_PropagationModel_outputs"
```

### NOTE
The generate_outputs and regression tests are also available for PropagationModel2D