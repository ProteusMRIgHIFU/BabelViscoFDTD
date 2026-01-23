import logging
import os
import sys

import numpy as np
import pytest

# Ensure that we don't use the local version of BabelViscoFDTD for apple systems as this will fail through a missing library error
if sys.platform == "darwin":
    working_dir = os.getcwd()
    if working_dir in sys.path:
        sys.path.remove(working_dir)
    
# Grab BabelViscoFDTD from environment
from BabelViscoFDTD.PropagationModel import PropagationModel
from BabelViscoFDTD.PropagationModel2D import PropagationModel2D

pytest.mark.generate_outputs
def test_generate_PropagationModel_outputs(frequency,ppw,computing_backend,get_gpu_device,setup_propagation_model,request,get_config_dirs):

    # =============================================================================
    # Test Setup
    # =============================================================================
    
    # Create output folder
    config_dirs = get_config_dirs
    output_dir = os.path.join(os.getcwd(),config_dirs['gen_output_dir'])
    os.makedirs(output_dir,exist_ok=True)
    
    # output file name
    output_file = f"{output_dir}/PropagationModel_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW"
        
    # =============================================================================
    # PROPAGATIONMODEL SETUP
    # =============================================================================
    
    # Current system GPU device
    gpu_device = get_gpu_device()
    
    # Create propagation model and get parameters necessary for the sim
    propagation_model = PropagationModel()
    pmodel_params = setup_propagation_model(us_frequency=frequency,points_per_wavelength=ppw,skip_plots=True)
    
    # =============================================================================
    # RUN PROPAGATIONMODEL USING GPU
    # =============================================================================
    
    # Additional setup
    results_type = 3 # Return RMS Data (1), Peak Data (2), or both (3)
    results_outputs = ['Vx','Vy','Vz','Pressure','Sigmaxx','Sigmayy', 'Sigmazz','Sigmaxy','Sigmaxz','Sigmayz']
    sensor_outputs = ['Pressure','Vx','Vy','Vz','Sigmaxx','Sigmayy', 'Sigmazz','Sigmaxy','Sigmaxz','Sigmayz']
    
    computing_backend_index = 0 # default to CPU
    if computing_backend['type'] == 'CUDA':
        computing_backend_index = 1
    elif computing_backend['type'] == 'OpenCL':
        computing_backend_index = 2 
    elif computing_backend['type'] == 'Metal':
        computing_backend_index = 3
    elif computing_backend['type'] == 'MLX':
        computing_backend_index = 4
    else:
        raise ValueError("Invalid computing_backend specified")
    
    if results_type == 1:
        results_type_str = "RMS"
    elif results_type == 2:
        results_type_str = "Peak"
    else:
        results_type_str = "RMS_Peak"
    output_file += f"_{len(results_outputs)}_{results_type_str}_results.npy"
    
    gpu_results = propagation_model.StaggeredFDTD_3D_with_relaxation(MaterialMap = pmodel_params['material_map'],
                                                                     MaterialProperties = pmodel_params['material_list'],
                                                                     Frequency = frequency,
                                                                     SourceMap = pmodel_params['source_map'],
                                                                     SourceFunctions = pmodel_params['pulse_source'],
                                                                     SpatialStep = pmodel_params['spatial_step'],
                                                                     DurationSimulation = pmodel_params['sim_time'],
                                                                     SensorMap = pmodel_params['sensor_map'],
                                                                     Ox = pmodel_params['Ox'],
                                                                     Oy = pmodel_params['Oy'],
                                                                     Oz = pmodel_params['Oz'],
                                                                     NDelta = pmodel_params['pml_thickness'],
                                                                     ReflectionLimit = pmodel_params['reflection_limit'],
                                                                     COMPUTING_BACKEND = computing_backend_index,
                                                                     USE_SINGLE = True,
                                                                     DT = pmodel_params['dt'],
                                                                     QfactorCorrection = True,
                                                                     SelRMSorPeak = results_type,
                                                                     SelMapsRMSPeakList = results_outputs,
                                                                     SelMapsSensorsList = sensor_outputs,
                                                                     SensorSubSampling = pmodel_params['sensor_steps'],
                                                                     DefaultGPUDeviceName = gpu_device,
                                                                     TypeSource=0)
    
    logging.info('Saving results for future use in regression tests')
    np.save(output_file,gpu_results)
    
    pytest.mark.generate_outputs
def test_generate_PropagationModel2D_outputs(frequency,ppw,computing_backend,get_gpu_device,setup_propagation_model,request,get_config_dirs):

    # =============================================================================
    # Test Setup
    # =============================================================================
    
    # Create output folder
    config_dirs = get_config_dirs
    output_dir = os.path.join(os.getcwd(),config_dirs['gen_output_dir'])
    os.makedirs(output_dir,exist_ok=True)
    
    # output file name
    output_file = f"{output_dir}/PropagationModel2D_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW"
        
    # =============================================================================
    # PROPAGATIONMODEL SETUP
    # =============================================================================
    
    # Current system GPU device
    gpu_device = get_gpu_device()
    
    # Create propagation model and get parameters necessary for the sim
    propagation_model_2D = PropagationModel2D()
    pmodel_params = setup_propagation_model(us_frequency=frequency,points_per_wavelength=ppw,axes=2,skip_plots=True)
    
    # =============================================================================
    # RUN PROPAGATIONMODEL USING GPU
    # =============================================================================
    
    # Additional setup
    results_type = 3 # Return RMS Data (1), Peak Data (2), or both (3)
    results_outputs = ['Vx','Vy','Pressure','Sigmaxx','Sigmayy','Sigmaxy']
    sensor_outputs = ['Pressure','Vx','Vy','Sigmaxx','Sigmayy','Sigmaxy']
    
    computing_backend_index = 0 # default to CPU
    if computing_backend['type'] == 'CUDA':
        computing_backend_index = 1
    elif computing_backend['type'] == 'OpenCL':
        computing_backend_index = 2 
    elif computing_backend['type'] == 'Metal':
        computing_backend_index = 3
    elif computing_backend['type'] == 'MLX':
        computing_backend_index = 4
    else:
        raise ValueError("Invalid computing_backend specified")
    
    if results_type == 1:
        results_type_str = "RMS"
    elif results_type == 2:
        results_type_str = "Peak"
    else:
        results_type_str = "RMS_Peak"
    output_file += f"_{len(results_outputs)}_{results_type_str}_results.npy"
    
    gpu_results = propagation_model_2D.StaggeredFDTD_2D_with_relaxation(MaterialMap = pmodel_params['material_map'],
                                                                        MaterialProperties = pmodel_params['material_list'],
                                                                        Frequency = frequency,
                                                                        SourceMap = pmodel_params['source_map'],
                                                                        SourceFunctions = pmodel_params['pulse_source'],
                                                                        SpatialStep = pmodel_params['spatial_step'],
                                                                        DurationSimulation = pmodel_params['sim_time'],
                                                                        SensorMap = pmodel_params['sensor_map'],
                                                                        Ox = pmodel_params['Ox'],
                                                                        Oy = pmodel_params['Oy'],
                                                                        NDelta = pmodel_params['pml_thickness'],
                                                                        ReflectionLimit = pmodel_params['reflection_limit'],
                                                                        COMPUTING_BACKEND = computing_backend_index,
                                                                        USE_SINGLE = True,
                                                                        DT = pmodel_params['dt'],
                                                                        QfactorCorrection = True,
                                                                        SelRMSorPeak = results_type,
                                                                        SelMapsRMSPeakList = results_outputs,
                                                                        SelMapsSensorsList = sensor_outputs,
                                                                        SensorSubSampling = pmodel_params['sensor_steps'],
                                                                        DefaultGPUDeviceName = gpu_device,
                                                                        TypeSource=0)
    
    logging.info('Saving results for future use in regression tests')
    np.save(output_file,gpu_results)