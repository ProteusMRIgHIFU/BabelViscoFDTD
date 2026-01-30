import glob
import logging
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sp

# Ensure that we don't use the local version of BabelViscoFDTD for apple systems as this will fail through a missing library error
if sys.platform == "darwin":
    working_dir = os.getcwd()
    if working_dir in sys.path:
        sys.path.remove(working_dir)

# Grab BabelViscoFDTD from environment
from BabelViscoFDTD.PropagationModel import PropagationModel
 
@pytest.mark.parametrize(
    "frequency",
    [2e5,5e5,1e6],
    ids = ["200kHz","500kHz","1000kHz"]
)
@pytest.mark.parametrize(
    "material_type",
    ["bone","brain"],
    ids = ["Skull","Brain"]
    
)
def test_PropagationModel_normal_source_with_att(frequency,material_type,computing_backend,get_gpu_device,setup_propagation_model,request,get_mpl_plot,get_line_plot,compare_data,get_phase_data):
    
    # Save plot screenshots to be added to html report later
    request.node.screenshots = []
    
    ppw = 6
    w = 2 * np.pi * frequency
    tolerance = 0.1
    
    # =============================================================================
    # PROPAGATIONMODEL SETUP
    # =============================================================================
    
    # Current system GPU device
    gpu_device = get_gpu_device()
    
    # Create propagation model and get parameters necessary for the sim
    propagation_model = PropagationModel()
    pmodel_params = setup_propagation_model(us_frequency=frequency,points_per_wavelength=ppw,map_type=material_type,source_shape ="square")
    
    # Determine indices for start, end, and material change layers
    m_map = pmodel_params['material_map']
    material_diff_z = np.diff(m_map[m_map.shape[0]//2,m_map.shape[1]//2,:])
    material_change_zinds = np.where(material_diff_z != 0)[0]+1 
    start_layer = material_change_zinds[0]-1
    end_layer = material_change_zinds[-1]+int(5*ppw)
    
    # =============================================================================
    # RUN PROPAGATIONMODEL
    # =============================================================================
    
    # Additional setup
    results_type = 3 # Return RMS Data (1), Peak Data (2), or both (3)
    results_outputs = ['Pressure']
    sensor_outputs = ['Pressure']
    
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
    
    # Run sim through skull tissue
    logging.info(f"Running babelviscofdtd with {material_type} tissue")
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
                                                                     QCorrection=pmodel_params['QCorrection'],
                                                                     SelRMSorPeak = results_type,
                                                                     SelMapsRMSPeakList = results_outputs,
                                                                     SelMapsSensorsList = sensor_outputs,
                                                                     SensorSubSampling = pmodel_params['sensor_steps'],
                                                                     DefaultGPUDeviceName = gpu_device,
                                                                     TypeSource=0)
    
    # Run sim through water only
    logging.info("Running babelviscofdtd with water only")
    gpu_results_water = propagation_model.StaggeredFDTD_3D_with_relaxation(MaterialMap = np.zeros_like(pmodel_params['water_map']),
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
    
    if results_type == 3:
        sensor_results_gpu_dict,last_map_gpu_dict,rms_results_gpu_dict,peak_results_gpu_dict,input_params_gpu = gpu_results
        sensor_results_water_gpu_dict,last_map_water_gpu_dict,rms_results_water_gpu_dict,peak_results_water_gpu_dict,input_params_water_gpu = gpu_results_water
    else:
        sensor_results_gpu_dict,last_map_gpu_dict,rmsorpeak_results_gpu_dict,input_params_gpu = gpu_results
        sensor_results_water_gpu_dict,last_map_water_gpu_dict,rmsorpeak_results_water_gpu_dict,input_params_water_gpu = gpu_results_water
    
    # Calculate water acoustic impedance
    density_water = pmodel_params['material_list'][0][0]
    sos_water = pmodel_params['material_list'][0][1]
    impedance_water = density_water * sos_water
    
    # Calculate intensity at measurement plane (excluding pml layers)
    intensity = peak_results_gpu_dict['Pressure'][pmodel_params['pml_thickness']:-pmodel_params['pml_thickness'],pmodel_params['pml_thickness']:-pmodel_params['pml_thickness'],end_layer]**2 / impedance_water
    intensity_water = peak_results_water_gpu_dict['Pressure'][pmodel_params['pml_thickness']:-pmodel_params['pml_thickness'],pmodel_params['pml_thickness']:-pmodel_params['pml_thickness'],end_layer]**2 / impedance_water
    
    # Calculate acoustic power
    acoustic_power = np.sum(intensity * (pmodel_params['spatial_step'] **2))
    acoustic_power_water = np.sum(intensity_water * (pmodel_params['spatial_step']**2))
    
    # Calculate Transmission Coefficient
    transmission_coeff_babel = np.sqrt(acoustic_power/acoustic_power_water)
    
    # Need to adjust end_layer for analytical calculation
    end_layer-=1
    
    # =============================================================================
    # ANALYTICAL CALCULATION USING SYMBOLIC MATH USING FOLDS PAPER
    # =============================================================================
    # See folds paper (https://doi-org.ezproxy.lib.ucalgary.ca/10.1121/1.381643)
    # Coefficient matrix equations below were derived from this paper

    logging.info("Starting Analytical Calculation")
    a_matrix_final = sp.eye(4,4) # Final coefficient matrix
    
    # General values
    w = 2 * np.pi * frequency
    d = pmodel_params['spatial_step']
    
    # Values for layer n+1
    c_real_np1 = pmodel_params['material_list'][0][1] # Long. SOS of layer n+1 (water)
    c_imag_np1 = pmodel_params['material_list'][0][3] # Long. attenuation of layer n+1 (water)
    theta_np1 = 0 # Angle of incidence at layer n+1 (i.e. first layer of water)
    k_long_np1 = w / (c_real_np1 * (1 - 1j*(2 * c_imag_np1 * c_real_np1)/w)) # Evaluates to w/c
    
    # X-component of the wave vector (same for all layers)
    sigma = k_long_np1 * sp.sin(theta_np1)  
    
    # Calculate transmission coefficient matrix for each material
    z = 0   # current material thickness
    for n in range(end_layer,start_layer,-1):
        
        z += d
        if n not in material_change_zinds:
            continue
        
        # Current layer material index
        material_index = m_map[m_map.shape[0]//2,m_map.shape[1]//2,n]
        
        # Current layer material properties
        mat_c_real = pmodel_params['material_list'][material_index][1]      # Long. SOS
        mat_b_real = pmodel_params['material_list'][material_index][2]      # Shear SOS
        mat_c_imag = pmodel_params['material_list'][material_index][3]      # Long. attenuation
        mat_b_imag = pmodel_params['material_list'][material_index][4]      # Shear attenuation
        layer_density = pmodel_params['material_list'][material_index][0]   # Density
        if mat_b_real == 0:
            mat_b_real = 1e-32
            mat_b_imag = 1e32
            # mat_b_real = 0
            # mat_b_imag = sp.oo
        
        # Wave Number Calculation
        kl = w / (mat_c_real * (1 - 1j*(2 * mat_c_imag * mat_c_real)/w))    # longitudinal complex wave number
        ks = w / (mat_b_real * (1 - 1j*(2 * mat_b_imag * mat_b_real)/w))    # transverse complex wave number
        
        # Other calculated values
        alpha_n = sp.sqrt(kl**2 - sigma**2)  # layer longitudinal attenuation
        Beta_n = sp.sqrt(ks**2 - sigma**2)  # layer transverse attenuation
        mu_n = layer_density*((w/ks)**2)  # bulk modulus
        lambda_n = layer_density*((w/kl)**2)  # shear_modulus
        
        t1 = 2 * alpha_n**2 * mu_n + (kl**2 * lambda_n) + 2*mu_n*sigma**2
        t2 = Beta_n**2 - sigma**2
        t3 = Beta_n**2 + sigma**2
        t4 = 2*(alpha_n**2)*mu_n + (kl**2)*lambda_n
        Pn = alpha_n*z
        Qn = Beta_n*z
        
        # Coefficient matrix for current material
        # Equations derived from paper for all angles and material types
        a_mat = sp.zeros(4,4)
        a_mat[0,0] = ((2*mu_n*(sigma**2)*sp.cos(Pn)) + t4*(sp.cos(Qn)))/(t1)
        a_mat[0,1] = ((1j*sigma)*((-2*alpha_n*Beta_n*sp.sin(Qn)) + t2*(sp.sin(Pn))))/((alpha_n)*t3)
        a_mat[0,2] = ((w*sigma)*(-sp.cos(Pn)+sp.cos(Qn)))/(t1)
        a_mat[0,3] = -(1j*w*(alpha_n*Beta_n*sp.sin(Qn) + (sigma**2)*sp.sin(Pn)))/(alpha_n*mu_n*t3)
        a_mat[1,0] = (1j*sigma*(2*alpha_n*Beta_n*mu_n*sp.sin(Pn) - t4*sp.sin(Qn)))/(Beta_n*(t1))
        a_mat[1,1] = (2*(sigma**2)*sp.cos(Qn) + t2*sp.cos(Pn))/t3
        a_mat[1,2] = -(1j*w*(alpha_n*Beta_n*sp.sin(Pn) + (sigma**2)*sp.sin(Qn)))/(Beta_n*(t1))
        a_mat[1,3] = (w*sigma*(-sp.cos(Pn) + sp.cos(Qn)))/(mu_n*t3)
        a_mat[2,0] = (2*mu_n*sigma*t4*(-sp.cos(Pn) + sp.cos(Qn)))/(w*(t1))
        a_mat[2,1] = (1j*(-4*alpha_n*Beta_n*mu_n*(sigma**2)*sp.sin(Qn) - t2*t4*sp.sin(Pn)))/(alpha_n*w*t3)
        a_mat[2,2] = (2*mu_n*(sigma**2)*sp.cos(Qn) + t4*sp.cos(Pn))/(t1)
        a_mat[2,3] = (1j*sigma*(-2*alpha_n*Beta_n*mu_n*sp.sin(Qn) + t4*sp.sin(Pn)))/(alpha_n*mu_n*t3)
        a_mat[3,0] = (1j*mu_n*(-4*alpha_n*Beta_n*mu_n*(sigma**2)*sp.sin(Pn) - t2*(2*(alpha_n**2)*mu_n + ((kl**2) * lambda_n))*sp.sin(Qn)))/(Beta_n*w*(t1))
        a_mat[3,1] = (2*mu_n*sigma*t2*(-sp.cos(Pn) + sp.cos(Qn)))/(w*t3)
        a_mat[3,2] = (1j*mu_n*sigma*(2*alpha_n*Beta_n*sp.sin(Pn) + (-(Beta_n**2) + sigma**2)*sp.sin(Qn)))/(Beta_n*(t1))
        a_mat[3,3] = (2*(sigma**2)*sp.cos(Pn) + t2*sp.cos(Qn))/t3
        
        # Multiply cumulative matrix with current matrix
        a_matrix_final = a_matrix_final @ a_mat
        
        # Reset material thickness
        z = 0
    
    # Transmission coefficient calculation
    M22 = a_matrix_final[1,1] - (a_matrix_final[1,0]*a_matrix_final[3,1])/(a_matrix_final[3,0])
    M23 = a_matrix_final[1,2] - (a_matrix_final[1,0]*a_matrix_final[3,2])/(a_matrix_final[3,0])
    M32 = a_matrix_final[2,1] - (a_matrix_final[2,0]*a_matrix_final[3,1])/(a_matrix_final[3,0])
    M33 = a_matrix_final[2,2] - (a_matrix_final[2,0]*a_matrix_final[3,2])/(a_matrix_final[3,0])
    Z1 = Znp1 = impedance_water
    
    transmission_coeff_truth = (2 * Z1) / ((M22+Z1*M23)*Znp1 + M32 + (Z1*M33))
    transmission_coeff_truth = np.abs(np.array(transmission_coeff_truth,dtype=np.complex128))
    
    # =============================================================================
    # VISUALISATION
    # =============================================================================
    end_layer+=1
    diff_plot = abs(rms_results_water_gpu_dict['Pressure']-rms_results_gpu_dict['Pressure'])
    
    # Measurement Plane Plot
    measurement_plane_plot = np.zeros_like(rms_results_water_gpu_dict['Pressure'])
    measurement_plane_plot[pmodel_params['pml_thickness']:-pmodel_params['pml_thickness'],pmodel_params['pml_thickness']:-pmodel_params['pml_thickness'],end_layer] = 1
    screenshot = get_mpl_plot([measurement_plane_plot], axes_num=3,titles=['Measurement\nPlane'],measurement_plane_index=end_layer,extent=pmodel_params['extent'],colorbar=True)
    request.node.screenshots.append(screenshot)
    
    # Pressure Plot
    pmax = np.max([rms_results_water_gpu_dict['Pressure'],rms_results_gpu_dict['Pressure']])
    screenshot = get_mpl_plot([rms_results_water_gpu_dict['Pressure'],rms_results_gpu_dict['Pressure'],diff_plot], axes_num=3,titles=['Water\nRMS Pressure',f'{material_type}\nRMS Pressure','Pressure\nDifference'],measurement_plane_index=end_layer,extent=pmodel_params['extent'],colorbar=True,clim=[0.0,pmax])
    request.node.screenshots.append(screenshot)
    
    # Phase Calculation
    phase_map_material = get_phase_data(time_vector = sensor_results_gpu_dict['time'].copy(),
                                        pressure_vector = sensor_results_gpu_dict['Pressure'],
                                        sensor_steps = pmodel_params['sensor_steps'],
                                        sensor_ppp = pmodel_params['sensor_ppp'],
                                        frequency = frequency,
                                        sensor_map = pmodel_params['sensor_map'].copy(),
                                        material_map = pmodel_params['material_map'].copy())
    
    phase_map_water = get_phase_data(time_vector = sensor_results_water_gpu_dict['time'].copy(),
                                     pressure_vector = sensor_results_water_gpu_dict['Pressure'],
                                     sensor_steps = pmodel_params['sensor_steps'],
                                     sensor_ppp = pmodel_params['sensor_ppp'],
                                     frequency = frequency,
                                     sensor_map = pmodel_params['sensor_map'].copy(),
                                     material_map = pmodel_params['water_map'].copy())
    # Phase Plot
    phase_min = np.min([phase_map_water,phase_map_material])
    phase_max = np.max([phase_map_water,phase_map_material])
    screenshot = get_mpl_plot([phase_map_water,phase_map_material,np.abs(phase_map_material-phase_map_water)], axes_num=3,titles=['Water\nPressure Phase',f'{material_type}\nPressure Phase','Phase\nDifference'],measurement_plane_index=end_layer,extent=pmodel_params['extent'],colorbar=True,clim=[phase_min,phase_max])
    request.node.screenshots.append(screenshot)
    
    # =============================================================================
    # COMPARISON
    # =============================================================================
    logging.info(f"BabelViscoFDTD Transmission Coefficient: {transmission_coeff_babel}")
    logging.info(f"Truth Transmission Coefficient: {transmission_coeff_truth}")
    logging.info(f"Relative Transmission Coefficient Difference (%): {(transmission_coeff_babel/transmission_coeff_truth)*100-100}")
    
    assert transmission_coeff_babel == pytest.approx(transmission_coeff_truth, rel=tolerance), "BabelViscoFDTD Transmission coefficient does not match analytical solution"
    

def test_PropagationModel_vs_CPU(frequency,ppw,computing_backend,get_gpu_device,setup_propagation_model,request,get_mpl_plot,get_line_plot,compare_data,tolerance):

    # Save plot screenshots to be added to html report later
    request.node.screenshots = []
    
    # CPU truth file name
    truth_file = os.path.join(os.getcwd(),f"Tests/Test_Data/PropagationModel_CPU_{int(frequency/1e3)}kHz_{ppw}PPW")
        
    # =============================================================================
    # PROPAGATIONMODEL SETUP
    # =============================================================================
    
    # Current system GPU device
    gpu_device = get_gpu_device()
    
    # Create propagation model and get parameters necessary for the sim
    propagation_model = PropagationModel()
    pmodel_params = setup_propagation_model(us_frequency=frequency,points_per_wavelength=ppw)
    
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
        cpu_results = propagation_model.StaggeredFDTD_3D_with_relaxation(MaterialMap = pmodel_params['material_map'],
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
                                                                         COMPUTING_BACKEND = 0, # CPU selected
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
            dice_coeff = calc_dice_coeff(rms_results_cpu_dict[output],rms_results_gpu_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
            dice_coeff = calc_dice_coeff(peak_results_cpu_dict[output],peak_results_gpu_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)
        else:
            dice_coeff = calc_dice_coeff(rmsorpeak_results_cpu_dict[output],rmsorpeak_results_gpu_dict[output],rel_tolerance=tolerance)
            total_dice_coeff.append(dice_coeff)

    final_dice_coeff = np.mean(total_dice_coeff)
    
    assert final_dice_coeff == pytest.approx(1.0, rel=1e-9), f"Average DICE coefficient is not 1"
    
def test_PropagationModel_regression(frequency,ppw,computing_backend,get_gpu_device,setup_propagation_model,request,get_mpl_plot,get_line_plot,compare_data,get_config_dirs,load_files,tolerance):

    # =============================================================================
    # Test Setup
    # =============================================================================
    
    config_dirs = get_config_dirs
    ref_dir = config_dirs['ref_dir_1']
    
    # Save plot screenshots to be added to html report later
    request.node.screenshots = []
    
    # Reference file name
    ref_file = os.path.join(ref_dir,f"PropagationModel_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW")
    
    # =============================================================================
    # PROPAGATIONMODEL SETUP
    # =============================================================================
    
    # Current system GPU device
    gpu_device = get_gpu_device()
    
    # Create propagation model and get parameters necessary for the sim
    propagation_model = PropagationModel()
    pmodel_params = setup_propagation_model(us_frequency=frequency,points_per_wavelength=ppw)
    
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
    
    test_results = propagation_model.StaggeredFDTD_3D_with_relaxation(MaterialMap = pmodel_params['material_map'],
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

def test_PropagationModel_two_outputs(frequency,ppw,computing_backend,get_gpu_device,setup_propagation_model,request,get_mpl_plot,get_line_plot,compare_data,get_config_dirs,load_files,tolerance):

    # =============================================================================
    # Test Setup
    # =============================================================================
    
    config_dirs = get_config_dirs
    ref_dir_1 = config_dirs['ref_dir_1']
    ref_dir_2 = config_dirs['ref_dir_2']
    
    # Save plot screenshots to be added to html report later
    request.node.screenshots = []
    
    # Reference file name
    ref_file_1 = os.path.join(ref_dir_1,f"PropagationModel_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW")
    ref_file_2 = os.path.join(ref_dir_2,f"PropagationModel_{computing_backend['type']}_{int(frequency/1e3)}kHz_{ppw}PPW")
    
    # Results
    results_type = 3 # Return RMS Data (1), Peak Data (2), or both (3)
    results_outputs = ['Vx','Vy','Vz','Pressure','Sigmaxx','Sigmayy', 'Sigmazz','Sigmaxy','Sigmaxz','Sigmayz']
    sensor_outputs = ['Pressure','Vx','Vy','Vz','Sigmaxx','Sigmayy', 'Sigmazz','Sigmaxy','Sigmaxz','Sigmayz']
    
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