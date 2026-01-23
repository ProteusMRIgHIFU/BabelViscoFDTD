import glob
import logging
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Ensure that we don't use the local version of BabelViscoFDTD for apple systems as this will fail through a missing library error
if sys.platform == "darwin":
    working_dir = os.getcwd()
    if working_dir in sys.path:
        sys.path.remove(working_dir)

# Grab BabelViscoFDTD from environment
from BabelViscoFDTD.PropagationModel2D import PropagationModel2D

@pytest.mark.skip("CPU backend for 2D sims is not yet supported")
def test_PropagationModel2D_vs_CPU(frequency,ppw,computing_backend,get_gpu_device,request,setup_propagation_model,get_line_plot,get_mpl_plot,compare_data,tolerance):
    
    # Save plot screenshots to be added to html report later
    request.node.screenshots = []
    
    # CPU truth file name
    truth_file = os.path.join(os.getcwd(),f"Tests/Test_Data/PropagationModel2D_CPU_{int(frequency/1e3)}kHz_{ppw}PPW")
    
    # =============================================================================
    # PROPAGATIONMODEL SETUP
    # =============================================================================
    
    # Current system GPU device
    gpu_device = get_gpu_device()
    
    # Create propagation model and get parameters necessary for the sim
    propagation_model_2D = PropagationModel2D()
    pmodel_params = setup_propagation_model(us_frequency=frequency,points_per_wavelength=ppw,axes=2)
    
    # =============================================================================
    # RUN PROPAGATIONMODEL2D USING GPU
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
    
    if results_type == 3:
        sensor_results_gpu_dict,last_map_gpu_dict,rms_results_gpu_dict,peak_results_gpu_dict,input_params_gpu = gpu_results
    else:
        sensor_results_gpu_dict,last_map_gpu_dict,rmsorpeak_results_gpu_dict,input_params_gpu = gpu_results
    
    # =============================================================================
    # RUN PROPAGATIONMODEL USING CPU
    # =============================================================================
    if results_type == 1:
        results_type_str = "RMS"
    elif results_type == 2:
        results_type_str = "Peak"
    else:
        results_type_str = "RMS_Peak"
    truth_file += f"_{len(results_outputs)}_{results_type_str}_results.npy"
    
    try:
        logging.info('Reloading CPU truth')
        cpu_results = np.load(truth_file, allow_pickle=True)
    except:
        logging.info("File doesn't exist")
        logging.info('Generating CPU truth')
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
                                                                        COMPUTING_BACKEND = 0, # CPU Selected
                                                                        USE_SINGLE = True,
                                                                        DT = pmodel_params['dt'],
                                                                        QfactorCorrection = True,
                                                                        SelRMSorPeak = results_type,
                                                                        SelMapsRMSPeakList = results_outputs,
                                                                        SelMapsSensorsList = sensor_outputs,
                                                                        SensorSubSampling = pmodel_params['sensor_steps'],
                                                                        DefaultGPUDeviceName = gpu_device,
                                                                        TypeSource=0)
        
        logging.info('Saving results for future use')
        np.save(truth_file,cpu_results)
    
    if results_type == 3:
        sensor_results_cpu_dict,last_map_cpu_dict,rms_results_cpu_dict,peak_results_cpu_dict,input_params_cpu = cpu_results
    else:
        sensor_results_cpu_dict,last_map_cpu_dict,rmsorpeak_results_cpu_dict,input_params_cpu = cpu_results
    
    # =============================================================================
    # VISUALISATION
    # =============================================================================
    output_types = {'Sensor': [sensor_results_cpu_dict, sensor_results_gpu_dict]}
    if results_type == 1:
        output_types['RMS'] = [rmsorpeak_results_cpu_dict, rmsorpeak_results_gpu_dict]
    elif results_type == 2:
        output_types['Peak'] = [rmsorpeak_results_cpu_dict, rmsorpeak_results_gpu_dict]
    else:
        output_types['RMS']  = [rms_results_cpu_dict, rms_results_gpu_dict]
        output_types['Peak'] = [peak_results_cpu_dict, peak_results_gpu_dict]
    
    for output_type_key,output_type_data in output_types.items():
        for output_key in output_type_data[0].keys():
            outputs = []
            titles = []
            
            if output_key == 'time':
                continue
            
            outputs.append(output_type_data[0][output_key])
            outputs.append(output_type_data[1][output_key])
            outputs.append(abs(output_type_data[0][output_key]-output_type_data[1][output_key]))
            titles.append(f"{output_type_key} {output_key} - CPU")
            titles.append(f"{output_type_key} {output_key} - GPU")
            titles.append(f"{output_type_key} {output_key} - Difference")
          
            if output_type_key == 'Sensor':
                for i in range(len(outputs)):
                    outputs[i] = outputs[i][outputs[i].shape[0]//2,:] # Use halfway time point
                screenshot = get_line_plot(output_type_data[0]['time'],outputs, labels=titles, title = f"{output_key} Sensor Data",xlabel='time (s)')
                request.node.screenshots.append(screenshot)
            else:
                screenshot = get_mpl_plot(outputs, axes_num=2,titles=titles,color_map=plt.cm.jet,colorbar=True)
                request.node.screenshots.append(screenshot)
    
    # =============================================================================
    # COMPARISON
    # =============================================================================
    
    calc_dice_coeff = compare_data['dice_coefficient']
    total_dice_coeff = []
    
    for output in results_outputs:
        logging.info(f"\nComparing {output}")
        if results_type == 3:
            dice_coeff = calc_dice_coeff(rms_results_cpu_dict[output],rms_results_gpu_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
            dice_coeff = calc_dice_coeff(peak_results_cpu_dict[output],peak_results_gpu_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
        else:
            dice_coeff = calc_dice_coeff(rmsorpeak_results_cpu_dict[output],rmsorpeak_results_gpu_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)

    final_dice_coeff = np.mean(total_dice_coeff)
    
    assert final_dice_coeff == pytest.approx(1.0, rel=1e-9), f"Average DICE coefficient is not 1"

def test_PropagationModel2D_regression(frequency,ppw,computing_backend,get_gpu_device,setup_propagation_model,request,get_mpl_plot,get_line_plot,compare_data,get_config_dirs,tolerance):

    # =============================================================================
    # Test Setup
    # =============================================================================
    
    config_dirs = get_config_dirs
    ref_dir = config_dirs['ref_dir_1']
    
    # Save plot screenshots to be added to html report later
    request.node.screenshots = []
    
    # Reference file name
    ref_file = os.path.join(ref_dir,f"PropagationModel2D_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW")
    
    # =============================================================================
    # PROPAGATIONMODEL SETUP
    # =============================================================================
    
    # Current system GPU device
    gpu_device = get_gpu_device()
    
    # Create propagation model and get parameters necessary for the sim
    propagation_model = PropagationModel2D()
    pmodel_params = setup_propagation_model(us_frequency=frequency,points_per_wavelength=ppw,axes=2)
    
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
    
    test_results = propagation_model.StaggeredFDTD_2D_with_relaxation(MaterialMap = pmodel_params['material_map'],
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
    
    if results_type == 3:
        test_sensor_results_dict,test_last_map_dict,test_rms_results_dict,test_peak_results_dict,test_input_params = test_results
    else:
        test_sensor_results_dict,test_last_map_dict,test_rmsorpeak_results_dict,test_input_params = test_results
    
    # =============================================================================
    # LOAD REFERENCE RESULTS
    # =============================================================================
    
    # Load reference file
    logging.info('Reloading Reference results')
    if results_type == 1:
        results_type_str = "RMS"
    elif results_type == 2:
        results_type_str = "Peak"
    else:
        results_type_str = "RMS_Peak"
    ref_file += f"_{len(results_outputs)}_{results_type_str}_results.npy"
    try:
        ref_results = np.load(ref_file, allow_pickle=True)
    except:
        ref_file = re.sub(f"_{computing_backend['type']}","**",ref_file)
        alt_ref_file = glob.glob(ref_file,recursive=True)[0]
        ref_results = np.load(alt_ref_file, allow_pickle=True)
    
    # Unpack results
    if results_type == 3:
        ref_sensor_results_dict,ref_last_map_dict,ref_rms_results_dict,ref_peak_results_dict,ref_input_params = ref_results
    else:
        ref_sensor_results_dict,ref_last_map_dict,ref_rmsorpeak_results_dict,ref_input_params = ref_results
    
    # =============================================================================
    # VISUALISATION
    # =============================================================================
    output_types = {'Sensor': [ref_sensor_results_dict, test_sensor_results_dict]}
    if results_type == 1:
        output_types['RMS'] = [ref_rmsorpeak_results_dict, test_rmsorpeak_results_dict]
    elif results_type == 2:
        output_types['Peak'] = [ref_rmsorpeak_results_dict, test_rmsorpeak_results_dict]
    else:
        output_types['RMS']  = [ref_rms_results_dict, test_rms_results_dict]
        output_types['Peak'] = [ref_peak_results_dict, test_peak_results_dict]
    
    for output_type_key,output_type_data in output_types.items():
        for output_key in output_type_data[0].keys():
            outputs = []
            titles = []
            
            if output_key == 'time':
                continue
            
            outputs.append(output_type_data[0][output_key])
            outputs.append(output_type_data[1][output_key])
            outputs.append(abs(output_type_data[0][output_key]-output_type_data[1][output_key]))
            titles.append(f"{output_type_key} {output_key} - CPU")
            titles.append(f"{output_type_key} {output_key} - GPU")
            titles.append(f"{output_type_key} {output_key} - Difference")
          
            if output_type_key == 'Sensor':
                for i in range(len(outputs)):
                    outputs[i] = outputs[i][outputs[i].shape[0]//2,:] # Use halfway time point
                screenshot = get_line_plot(output_type_data[0]['time'],outputs, labels=titles, title = f"{output_key} Sensor Data",xlabel='time (s)')
                request.node.screenshots.append(screenshot)
            else:
                screenshot = get_mpl_plot(outputs, axes_num=3,titles=titles,color_map=plt.cm.jet,colorbar=True)
                request.node.screenshots.append(screenshot)
    
    # =============================================================================
    # COMPARISON
    # =============================================================================
    
    calc_dice_coeff = compare_data['dice_coefficient']
    total_dice_coeff = []
    
    for output in results_outputs:
        logging.info(f"\nComparing {output}")
        if results_type == 3:
            dice_coeff = calc_dice_coeff(ref_rms_results_dict[output],test_rms_results_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
            dice_coeff = calc_dice_coeff(ref_peak_results_dict[output],test_peak_results_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
        else:
            dice_coeff = calc_dice_coeff(ref_rmsorpeak_results_dict[output],test_rmsorpeak_results_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)

    final_dice_coeff = np.mean(total_dice_coeff)
    
    assert final_dice_coeff == pytest.approx(1.0, rel=1e-9), f"Average DICE coefficient is not 1"
    
def test_PropagationModel2D_two_outputs(frequency,ppw,computing_backend,get_gpu_device,setup_propagation_model,request,get_mpl_plot,get_line_plot,compare_data,get_config_dirs,tolerance):

    # =============================================================================
    # Test Setup
    # =============================================================================
    
    config_dirs = get_config_dirs
    ref_dir_1 = config_dirs['ref_dir_1']
    ref_dir_2 = config_dirs['ref_dir_2']
    
    # Save plot screenshots to be added to html report later
    request.node.screenshots = []
    
    # Reference file name
    ref_file_1 = os.path.join(ref_dir_1,f"PropagationModel2D_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW")
    ref_file_2 = os.path.join(ref_dir_2,f"PropagationModel2D_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW")
    
    # Results
    results_type = 3 # Return RMS Data (1), Peak Data (2), or both (3)
    results_outputs = ['Vx','Vy','Pressure','Sigmaxx','Sigmayy','Sigmaxy']
    sensor_outputs = ['Pressure','Vx','Vy','Sigmaxx','Sigmayy','Sigmaxy']
    
    # =============================================================================
    # LOAD RESULTS
    # =============================================================================
    
    # Finalize reference file names
    logging.info('Reloading Reference results')
    if results_type == 1:
        results_type_str = "RMS"
    elif results_type == 2:
        results_type_str = "Peak"
    else:
        results_type_str = "RMS_Peak"
    ref_file_1 += f"_{len(results_outputs)}_{results_type_str}_results.npy"
    ref_file_2 += f"_{len(results_outputs)}_{results_type_str}_results.npy"
    
    # Load reference file 1 as truth
    try:
        logging.info(f"Loading {ref_file_1}")
        ref_results = np.load(ref_file_1, allow_pickle=True)
    except:
        ref_file = re.sub(f"_{computing_backend['type']}","**",ref_file_1)
        alt_ref_file = glob.glob(ref_file,recursive=True)[0]
        ref_results = np.load(alt_ref_file, allow_pickle=True)
        logging.info(f"{ref_file_1} unavailable, loading {alt_ref_file} instead")
    
    # Load reference file 2 as test
    try:
        logging.info(f"Loading {ref_file_2}")
        test_results = np.load(ref_file_2, allow_pickle=True)
    except:
        test_file = re.sub(f"_{computing_backend['type']}","**",ref_file_2)
        alt_test_file = glob.glob(test_file,recursive=True)[0]
        test_results = np.load(alt_test_file, allow_pickle=True)
        logging.info(f"{ref_file_2} unavailable, loading {alt_test_file} instead")
        
    # Unpack results
    if results_type == 3:
        ref_sensor_results_dict,ref_last_map_dict,ref_rms_results_dict,ref_peak_results_dict,ref_input_params = ref_results
        test_sensor_results_dict,test_last_map_dict,test_rms_results_dict,test_peak_results_dict,test_input_params = test_results
    else:
        ref_sensor_results_dict,ref_last_map_dict,ref_rmsorpeak_results_dict,ref_input_params = ref_results
        test_sensor_results_dict,test_last_map_dict,test_rmsorpeak_results_dict,test_input_params = test_results
    
    # =============================================================================
    # VISUALISATION
    # =============================================================================
    output_types = {'Sensor': [ref_sensor_results_dict, test_sensor_results_dict]}
    if results_type == 1:
        output_types['RMS'] = [ref_rmsorpeak_results_dict, test_rmsorpeak_results_dict]
    elif results_type == 2:
        output_types['Peak'] = [ref_rmsorpeak_results_dict, test_rmsorpeak_results_dict]
    else:
        output_types['RMS']  = [ref_rms_results_dict, test_rms_results_dict]
        output_types['Peak'] = [ref_peak_results_dict, test_peak_results_dict]
    
    for output_type_key,output_type_data in output_types.items():
        for output_key in output_type_data[0].keys():
            outputs = []
            titles = []
            
            if output_key == 'time':
                continue
            
            outputs.append(output_type_data[0][output_key])
            outputs.append(output_type_data[1][output_key])
            outputs.append(abs(output_type_data[0][output_key]-output_type_data[1][output_key]))
            titles.append(f"{output_type_key} {output_key} - CPU")
            titles.append(f"{output_type_key} {output_key} - GPU")
            titles.append(f"{output_type_key} {output_key} - Difference")
          
            if output_type_key == 'Sensor':
                for i in range(len(outputs)):
                    outputs[i] = outputs[i][outputs[i].shape[0]//2,:] # Use halfway time point
                screenshot = get_line_plot(output_type_data[0]['time'],outputs, labels=titles, title = f"{output_key} Sensor Data",xlabel='time (s)')
                request.node.screenshots.append(screenshot)
            else:
                screenshot = get_mpl_plot(outputs, axes_num=3,titles=titles,color_map=plt.cm.jet,colorbar=True)
                request.node.screenshots.append(screenshot)
    
    # =============================================================================
    # COMPARISON
    # =============================================================================
    
    calc_dice_coeff = compare_data['dice_coefficient']
    total_dice_coeff = []
    
    for output in results_outputs:
        logging.info(f"\nComparing {output}")
        if results_type == 3:
            dice_coeff = calc_dice_coeff(ref_rms_results_dict[output],test_rms_results_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
            dice_coeff = calc_dice_coeff(ref_peak_results_dict[output],test_peak_results_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
        else:
            dice_coeff = calc_dice_coeff(ref_rmsorpeak_results_dict[output],test_rmsorpeak_results_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)

    final_dice_coeff = np.mean(total_dice_coeff)
    
    assert final_dice_coeff == pytest.approx(1.0, rel=1e-9), f"Average DICE coefficient is not 1"