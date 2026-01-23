# Pytest Setup and Usage Guide

This guide provides instructions for setting up and running pytest tests for BabelViscoFDTD, along with additional instructions for using it in Visual Studio Code.

## Installation
Before you begin, ensure you have Pytest installed in your environment.

``` bash
pip install pytest
```

The following packages are also required for BabelViscoFDTD pytests:
- pytest-html (Generating html test reports)
- pytest-metadata 
- pytest-qt (Handling QT applications)

```bash
pip install pytest-html, pytest-metadata, pytest-qt, pytest-xdist
```

These optional packages may also be installed:
- pytest-xdist (Parallelization of tests to speed up test run time, avoid using for GPU tests)
- pytest-benchmark (Evaluate code performance)
- pytest-profiling (Useful for determing code bottlenecks)
- pytest-sugar (Improve readability of pytest output)

```bash
pip install pytest-benchmark, pytest-profiling, pytest-sugar
```

## Writing Tests

Pytest allows you to write tests using Python's built-in `assert` statement. Test files/tests should be named with the prefix `test_` so that pytest can automatically discover them.

Here's an example of a few simple pytest test functions:

```python
# test_example.py

def test_addition():
    x = 1
    y = 2

    assert x + y == 3, "message shown if assertion fails"

@pytest.mark.custom_marker_1 # example of a marker
def test_subtraction():
    x = 1
    y = 2

    assert y - x == 1, "message shown if assertion fails"
```

Note that the folder structure of tests should parallel to that of the main directory

<pre>
BabelViscoFDTD/
├── pytest.ini
├── BabelViscoFDTD/
│     ├── PropagationModel.py
│     └── ...
├── Tests/
│     ├── config.ini
│     ├── conftest.py
│     ├── E2E/
│     ├── Integration/
│     ├── Unit/
│     │    ├── BabelViscoFDTD/
│     │    │    ├── test_PropagationModel.py
│     │    │    └── ...
│     │    └── ...
│     └── ...
└── ...
</pre>

## Running Tests
### In terminal

In the highest level directory of BabelViscoFDTD, simply execute the `pytest` command in your terminal. This command will discover and run **ALL** tests. Note that pytest will automatically be configured based on parameters specified in the `pytest.ini` file. 

A timestamped report of all tests ran will automatically be saved to the `Pytest_Reports` folder while the individual tests ran will also be saved to `Pytest_Reports/individual_tests/`

Specific tests can be ran by searching for test files/names using the -k commandline argument and/or searching for specific test markers uing the -m option. Some example searches include:

```bash
# Retrieves all test inside test_example.py (i.e. test_addition and test_subtraction)
pytest -k "test_example.py" 

# Retrieves test containing addition OR subtraction in the test filepath. Achieves same result as above command
pytest -k "addition or subtraction"

# Retrieves test containing BOTH addition and subtraction in the test filepath. Returns no tests in this case.
pytest -k "addition and subtraction"

# Retrieves any test marked with custom_marker_1 (i.e. test_subtraction)
pytest -m "custom_marker_1"
```

Note that a full list of available markers can be found in the `pytest.ini` file.

If you want to see what tests are collected without running them, you can add the `--collect-only` option

```bash
pytest -m "custom_marker_1" --collect-only
```

More information on other types of pytest arguments can be found [here](https://docs.pytest.org/en/6.2.x/usage.html).

### In Visual Studio Code

Open your project in Visual Studio Code. Install Python and Test Explorer UI extensions for Visual Studio Code.

Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS) and select "Open Workspace Settings (JSON)". Then set up the following parameters:

```json
{
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestPath": "<path_to_your_pytest>"
}
```
Example: For a conda environment, the pytest path looks something like 

`<user>/miniconda3/envs/<unit_testing_env>/Lib/site-packages/pytest`

Visual Studio Code will now discover pytest tests and display them in the testing tab; You can run them using the UI. 

If they do not appear, open the Command Palette again and select "Python: Select Interpreter" and choose the one in the same environment as your pytest package.

Note that the commandline arguments are still specified in the `pytest.ini` file and are run automatically.

Alternatively, tests can be run from the integrated terminal similar to previous section.


## Additional Setup for BabelViscoFDTD testing
Certain tests require a directory containing test data, GPU device name, etc. These parameters are specified in the `config.ini` which is an untracked file as it is user-specific. If it's your first time running tests, you should create the `config.ini` file inside the `Tests/` folder (see folder structure example from earlier) and format it as follows:

```plaintext
[Paths]
ref_output_folder_1_name = <folder name containing reference BabelViscoFDTD output data>
ref_output_folder_2_name = <folder name containing second reference BabelViscoFDTD output data>
gen_output_folder_name = <folder name to store outputs from test_generate_outputs.py>

[GPU]
device_name = <GPU device name> 
```

Where the

`ref_output_folder_1_name` folder contains previously generated BabelViscoFDTD outputs to be use in regression testing (i.e. test current setup against this reference).

`ref_output_folder_2_name` folder contains another previously generated BabelViscoFDTD outputs. Used when you want to compare two already generated ouput folders.

`gen_output_folder_name` folder is where output data generated from Generate_Outputs "Tests" will be stored. It will be stored at the following location: `Tests/Generate_Outputs/Generated_Outputs/<gen_output_folder_name>`

<!-- <pre>
BabelBrain_Test_Data/
├── &lt;Dataset_1&gt;/
│     ├── m2m_&lt;Dataset_1&gt;/
│     │     ├── final_tissues.nii.gz
│     │     └── ...
│     ├── Trajectories/
│     │     ├── &lt;Deep_Target&gt;.txt
│     │     ├── &lt;Superficial_Target&gt;.txt
│     │     ├── &lt;Skin_Target&gt;.txt
│     │     └── &lt;Outside_Target&gt;.txt
│     ├── &lt;Truth_file_1&gt;
│     └── &lt;Truth_file_2&gt;
├── &lt;Dataset_2&gt;/
│     └── ...
├── Profiles/
│     ├── Thermal_Profile_1.yaml
│     ├── Thermal_Profile_2.yaml
│     ├── MultiFocus_Profile1.yaml
│     └── ...
└── ...
</pre> -->

<!-- The names of datasets, trajectories, and transducers can be modified in the parameters section in the `conftest.py` file. -->

## Further Reading
For more information about pytest and its features, check out the [pytest documentation](https://docs.pytest.org/en/latest/).
