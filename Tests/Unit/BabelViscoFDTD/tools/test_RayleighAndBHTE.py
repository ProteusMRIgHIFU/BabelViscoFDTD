import logging
import os
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
from BabelViscoFDTD.tools.RayleighAndBHTE import BHTE, ForwardSimple, InitCuda, InitOpenCL, InitMetal, InitMLX, GenerateSurface

def test_BHTE_no_source(spatial_step,bioheat_exact,set_up_domain,compare_data,request,get_mpl_plot,computing_backend,get_gpu_device,tolerance):
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    X, Y, Z = create_grid(grid_limits = [-64, 64, -64, 64, -64, 64], grid_steps = 3*[spatial_step])
    
    # Set homogeneous brain medium
    set_medium = set_up_domain['medium_bhte']
    medium, medium_index = set_medium(medium_type='water')
    
    # Set Gaussian initial temperature distribution [degC]
    width = X.max()//6
    source = {}
    source['T0'] = (37 + 5 * np.exp( -(X / width)**2 - (Y / width)**2 - (Z / width)**2))
    
    # Time parameters
    Nt = 300 # number of time steps
    dt = 0.5 # time step
    
    # =========================================================================
    # SIMULATION USING BIOHEATEXACT
    # =========================================================================
    
    # Calculate diffusivity coefficient for medium
    D = bioheat_exact['diffusivity'](medium)
    
    # Compute Green's function solution using bioheatExact
    t0  = time.perf_counter()
    temp_exact = bioheat_exact['bioheat'](source['T0'], 0, [D, 0, 0], spatial_step, Nt * dt)
    t1 = time.perf_counter()
    logging.info(f"Truth method took {t1 - t0} s")
    
    # =========================================================================
    # SIMULATION USING BABELVISCOFDTD'S BHTE
    # =========================================================================
    
    # BHTE parameters
    nFactorMonitoring=int(2.5/dt) # Monitor every 2.5s
    pressure = np.zeros_like(source['T0'])
    MaterialMap = medium_index * np.ones_like(source['T0'],dtype=np.uint32)
    MaterialList,_ = set_up_domain['material_list_bhte']()
    
    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BHTE 
    t0  = time.perf_counter()
    temp_babelvisco,_,_,_ = BHTE(pressure,
                                MaterialMap,
                                MaterialList,
                                dx = spatial_step,
                                TotalDurationSteps=Nt,
                                nStepsOn=0,
                                LocationMonitoring=64,
                                nFactorMonitoring=nFactorMonitoring,
                                dt=dt,
                                DutyCycle=0.0,
                                Backend=computing_backend['type'],
                                stableTemp=37.0,
                                initT0=source['T0'].astype(np.float32))
    t1 = time.perf_counter()
    logging.info(f"BabelViscoFDTD BHTE method took {t1 - t0} s")

    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [source['T0'].T,temp_exact.T,temp_babelvisco.T,abs(temp_babelvisco.T-temp_exact.T)]
    plot_names = ['Initial Temp','bioheat_exact\noutput', 'babelvisco\nBHTE output','Output\nDifferences']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map=plt.cm.jet,colorbar=True)
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    calc_dice_coeff = compare_data['dice_coefficient']
    dice_coeff = calc_dice_coeff(temp_exact,temp_babelvisco,rel_tolerance=tolerance)
    
    assert dice_coeff == pytest.approx(1.0, rel=1e-9), f"DICE score is not 1 ({dice_coeff})"

def test_BHTE_no_source_with_perfusion(spatial_step,set_up_domain,bioheat_exact,compare_data,request,get_mpl_plot,computing_backend,get_gpu_device,tolerance):
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    X, Y, Z = create_grid(grid_limits = [-64, 64, -64, 64, -64, 64], grid_steps = 3*[spatial_step])
    
    # Set homogeneous brain medium
    set_medium = set_up_domain['medium_bhte']
    medium, medium_index = set_medium(medium_type='brain')
    
    # Set Gaussian initial temperature distribution [degC]
    width = X.max()//6
    source = {}
    source['T0'] = (37 + 5 * np.exp( -(X / width)**2 - (Y / width)**2 - (Z / width)**2))
    
    # Time parameters
    Nt = 300 # number of time steps
    dt = 0.5 # time step
    
    # =========================================================================
    # SIMULATION USING BIOHEATEXACT
    # =========================================================================
    
    # Calculate diffusivity coefficient for medium
    D = bioheat_exact['diffusivity'](medium)
    
    # Calculate perfusion coefficient for medium
    P = bioheat_exact['perfusion'](medium)
    
    # Compute Green's function solution using bioheatExact
    t0  = time.perf_counter()
    temp_exact = bioheat_exact['bioheat'](source['T0'], 0, [D, P, 37.0], spatial_step, Nt * dt)
    t1 = time.perf_counter()
    logging.info(f"Truth method took {t1 - t0} s")
    
    # =========================================================================
    # SIMULATION USING BABELVISCOFDTD'S BHTE
    # =========================================================================
    
    # BHTE parameters
    nFactorMonitoring=int(2.5/dt) # Monitor every 2.5s
    pressure = np.zeros_like(source['T0'])
    MaterialMap = medium_index * np.ones_like(source['T0'],dtype=np.uint32)
    MaterialList,_ = set_up_domain['material_list_bhte']()
    
    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BHTE 
    t0  = time.perf_counter()
    temp_babelvisco,_,_,_ = BHTE(pressure,
                                MaterialMap,
                                MaterialList,
                                dx = spatial_step,
                                TotalDurationSteps=Nt,
                                nStepsOn=0,
                                LocationMonitoring=64,
                                nFactorMonitoring=nFactorMonitoring,
                                dt=dt,
                                DutyCycle=0.0,
                                Backend=computing_backend['type'],
                                stableTemp=37.0,
                                initT0=source['T0'].astype(np.float32))
    t1 = time.perf_counter()
    logging.info(f"BabelViscoFDTD BHTE method took {t1 - t0} s")

    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [source['T0'].T,temp_exact.T,temp_babelvisco.T,abs(temp_babelvisco.T-temp_exact.T)]
    plot_names = ['Initial Temp','bioheat_exact\noutput', 'babelvisco\nBHTE output','Output\nDifferences']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map=plt.cm.jet,colorbar=True)
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    calc_dice_coeff = compare_data['dice_coefficient']
    dice_coeff = calc_dice_coeff(temp_exact,temp_babelvisco,rel_tolerance=tolerance)
    
    assert dice_coeff == pytest.approx(1.0, rel=1e-9), f"DICE score is not 1 ({dice_coeff})"
    
def test_BHTE_source_with_perfusion(spatial_step,set_up_domain,bioheat_exact,compare_data,request,get_mpl_plot,computing_backend,get_gpu_device,tolerance):
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    X, Y, Z = create_grid(grid_limits = [-64, 64, -64, 64, -64, 64], grid_steps = 3*[spatial_step])
    
    # Set homogeneous brain medium
    set_medium = set_up_domain['medium_bhte']
    medium, medium_index = set_medium(medium_type='brain')
    
    # Set initial temperature distribution to be constant [degC]
    source = {}
    source['T0'] = 37.0 * np.ones_like(X)
    
    # Set ultrasound parameters
    duty_cycle = 1.0
    
    # Set Gaussian volume rate of heat deposition [W/m^3]
    width = X.max()//6
    source['Q'] = 2e6 * np.exp( -(X / width)**2 - (Y / width)**2 - (Z / width)**2 )
    pressure = np.sqrt(source['Q'] * (2*medium['density']*medium['sos']*spatial_step) / duty_cycle) # pressure values needed for Babelvisco BHTE to produce same heat disposition
    
    # Time parameters
    Nt = 300 # number of time steps
    dt = 0.5 # time step
    
    # =========================================================================
    # SIMULATION USING BIOHEATEXACT
    # =========================================================================
    
    # Calculate diffusivity coefficient for medium
    D = bioheat_exact['diffusivity'](medium)
    
    # Calculate perfusion coefficient for medium
    P = bioheat_exact['perfusion'](medium)
    
    # Calculate normalized heat source
    S = bioheat_exact['heat_source'](medium,source['Q'])
    
    # Compute Green's function solution using bioheatExact
    t0  = time.perf_counter()
    temp_exact = bioheat_exact['bioheat'](source['T0'], S, [D, P, 37.0], spatial_step, Nt * dt)
    t1 = time.perf_counter()
    logging.info(f"Truth method took {t1 - t0} s")
    
    # =========================================================================
    # SIMULATION USING BABELVISCOFDTD'S BHTE
    # =========================================================================
    
    # BHTE parameters
    nFactorMonitoring=int(2.5/dt) # Monitor every 2.5s
    MaterialMap = medium_index * np.ones_like(source['T0'],dtype=np.uint32)
    MaterialList,_ = set_up_domain['material_list_bhte']()
    
    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BHTE 
    t0  = time.perf_counter()
    temp_babelvisco,_,_,_ = BHTE(pressure,
                                MaterialMap,
                                MaterialList,
                                dx = spatial_step,
                                TotalDurationSteps=Nt,
                                nStepsOn=Nt,
                                LocationMonitoring=64,
                                nFactorMonitoring=nFactorMonitoring,
                                dt=dt,
                                DutyCycle=duty_cycle,
                                Backend=computing_backend['type'],
                                stableTemp=37.0,
                                initT0=source['T0'].astype(np.float32))
    t1 = time.perf_counter()
    logging.info(f"BabelViscoFDTD BHTE method took {t1 - t0} s")

    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [source['T0'].T,temp_exact.T,temp_babelvisco.T,abs(temp_babelvisco.T-temp_exact.T)]
    plot_names = ['Initial Temp','bioheat_exact\noutput', 'babelvisco\nBHTE output','Output\nDifferences']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map=plt.cm.jet,colorbar=True)
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    calc_dice_coeff = compare_data['dice_coefficient']
    dice_coeff = calc_dice_coeff(temp_exact,temp_babelvisco,rel_tolerance=tolerance)
    
    assert dice_coeff == pytest.approx(1.0, rel=1e-9), f"DICE score is not 1 ({dice_coeff})"

def test_ForwardSimple(frequency,ppw,set_up_domain,request,get_line_plot,computing_backend,get_gpu_device,calc_axial_pressure):
    
    # =========================================================================
    # MEDIUM PROPERTIES
    # =========================================================================
    
    lossless_medium = {"density":                1.000e3,    # kg/m^3
                       "sos":                    1.500e3,    # m/s
                       }
    
    # =========================================================================
    # TRANSDUCER SETUP
    # =========================================================================
    
    # Signal Parameters
    wavelength = lossless_medium['sos'] / frequency
    spatial_step = wavelength / ppw
    amp = 60e3/lossless_medium['density']/lossless_medium['sos'] # 60 kPa
    attenuation = 0
    cwvnb_extlay=np.array(2*np.pi*frequency/lossless_medium['sos']+(-1j*attenuation)).astype(np.complex64)
    logging.info(f"\nFrequency: {frequency*1e-3} kHz\nPPW: {ppw}\nSpatial step: {spatial_step} m")
    
    # Create circular transducer
    tx_radius = 10 * wavelength
    tx_diameter = tx_radius * 2
    tx_focus = tx_radius * 4
    tx = GenerateSurface(spatial_step,tx_diameter,tx_focus)
    
    tx['center'][:,2]-=np.min(tx['center'][:,2]) #we make the back of the bowl to be aligned at 0
    tx['VertDisplay'][:,2]-=np.min(tx['VertDisplay'][:,2]) #we make the back of the bowl to be aligned at 0
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    grid_radius = 1.5 * tx_radius
    X, Y, Z = create_grid(grid_limits = [-grid_radius, grid_radius, -grid_radius, grid_radius, 0, 2*tx_focus], grid_steps = 3*[spatial_step])
    
    # =========================================================================
    # CALCULATE AXIAL PRESSURE USING FORMULA
    # =========================================================================
    h = tx_focus - np.sqrt(tx_focus**2 - tx_radius**2)
    
    axial_coords = Z[Z.shape[0]//2,Z.shape[1]//2,:]
    axial_pressure_truth = calc_axial_pressure(axial_coords,
                                               p_medium=lossless_medium['density'],
                                               omega=2*np.pi*frequency,
                                               c = lossless_medium["sos"],
                                               u0 = amp,
                                               a = tx_radius,
                                               A = tx_focus,
                                               h = h)
    
    # =========================================================================
    # RUN SIMULATION USING BABELVISCOFDTD'S FORWARDSIMPLE FUNCTION
    # =========================================================================
    
    # Additional setup
    rf=np.hstack((np.reshape(X,(np.prod(X.shape),1)),np.reshape(Y,(np.prod(Y.shape),1)), np.reshape(Z,(np.prod(Z.shape),1)))).astype(np.float32)
    u0=np.ones((tx['center'].shape[0],1),np.float32)+ 1j*np.zeros((tx['center'].shape[0],1),np.float32)
    u0*=amp

    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BabelViscoFDTD's Rayleigh Integral solver
    pressure_babelvisco = ForwardSimple(cwvnb_extlay,
                                        center=tx['center'].astype(np.float32),
                                        ds=tx['ds'].astype(np.float32),
                                        u0=u0,
                                        rf=rf,
                                        deviceMetal=gpu_device)
    
    pressure_babelvisco=np.abs(np.reshape(pressure_babelvisco,X.shape)*lossless_medium['density']*lossless_medium['sos'])
    axial_pressure_babelvisco = pressure_babelvisco[pressure_babelvisco.shape[0]//2,pressure_babelvisco.shape[1]//2,:]
    
    # =========================================================================
    # RESULTS CLEANUP
    # =========================================================================
    
    # Remove infinite values from results (Truth method will produce one at focal spot)
    mask = np.isfinite(axial_pressure_truth) & np.isfinite(axial_pressure_babelvisco)
    
    axial_pressure_truth = axial_pressure_truth[mask]
    axial_pressure_babelvisco = axial_pressure_babelvisco[mask]
    axial_coords = axial_coords[mask]
    
    removed_elements_num = abs(len(axial_pressure_truth) - len(mask))
    if removed_elements_num:
        logging.info(f"Removed {removed_elements_num} inf values from results")
    
    logging.info(f"\nTruth max: {axial_pressure_truth.max()}\nTruth min: {axial_pressure_truth.min()}\nTruth mean: {axial_pressure_truth.mean()}")
    logging.info(f"\nBabelViscoFDTD max: {axial_pressure_babelvisco.max()}\nBabelViscoFDTD min: {axial_pressure_babelvisco.min()}\nBabelViscoFDTD mean: {axial_pressure_babelvisco.mean()}")
    
    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [axial_pressure_truth,axial_pressure_babelvisco, abs(axial_pressure_babelvisco-axial_pressure_truth)]
    plot_names = ["Truth", "BabelViscoFDTD", "Difference"]
    screenshot = get_line_plot(axial_coords, data_list=plots, labels=plot_names, title = "Axial Pressure")
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    relative_L2_percent = 100.0 * np.sqrt(np.sum((axial_pressure_babelvisco - axial_pressure_truth)**2) / np.sum(axial_pressure_truth**2) )
    logging.info(f"Relative L2: {relative_L2_percent}%")
    
    assert relative_L2_percent <= 1, f"Relative L2 is >1% ({relative_L2_percent}%)"

@pytest.mark.parametrize(
    "frequency",
    [2e5,6e5,1e6],
    ids = ["200kHz","600kHz","1000kHz"]
)
def test_ForwardSimple_low_res_failure(frequency,set_up_domain,request,get_line_plot,computing_backend,get_gpu_device,calc_axial_pressure):
    
    # =========================================================================
    # MEDIUM PROPERTIES
    # =========================================================================
    
    lossless_medium = {"density":                1.000e3,    # kg/m^3
                       "sos":                    1.500e3,    # m/s
                       }
    
    # =========================================================================
    # TRANSDUCER SETUP
    # =========================================================================
    
    # Signal Parameters
    wavelength = lossless_medium['sos'] / frequency
    ppw = 6
    spatial_step = wavelength / ppw
    amp = 60e3/lossless_medium['density']/lossless_medium['sos'] # 60 kPa
    attenuation = 0
    cwvnb_extlay=np.array(2*np.pi*frequency/lossless_medium['sos']+(-1j*attenuation)).astype(np.complex64)
    logging.info(f"\nFrequency: {frequency*1e-3} kHz\nSpatial step: {spatial_step} m")
    
    # Create circular transducer
    tx_radius = 10 * wavelength
    tx_diameter = tx_radius * 2
    tx_focus = tx_radius * 4
    tx = GenerateSurface(wavelength,tx_diameter,tx_focus) # Make spatial step = wavelength
    
    tx['center'][:,2]-=np.min(tx['center'][:,2]) #we make the back of the bowl to be aligned at 0
    tx['VertDisplay'][:,2]-=np.min(tx['VertDisplay'][:,2]) #we make the back of the bowl to be aligned at 0
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    grid_radius = 1.5 * tx_radius
    X, Y, Z = create_grid(grid_limits = [-grid_radius, grid_radius, -grid_radius, grid_radius, 0, 2*tx_focus], grid_steps = 3*[spatial_step])
    
    # =========================================================================
    # CALCULATE AXIAL PRESSURE USING FORMULA
    # =========================================================================
    h = tx_focus - np.sqrt(tx_focus**2 - tx_radius**2)
    
    axial_coords = Z[Z.shape[0]//2,Z.shape[1]//2,:]
    axial_pressure_truth = calc_axial_pressure(axial_coords,
                                               p_medium=lossless_medium['density'],
                                               omega=2*np.pi*frequency,
                                               c = lossless_medium["sos"],
                                               u0 = amp,
                                               a = tx_radius,
                                               A = tx_focus,
                                               h = h)
    
    # =========================================================================
    # RUN SIMULATION USING BABELVISCOFDTD'S FORWARDSIMPLE FUNCTION
    # =========================================================================
    
    # Additional setup
    rf=np.hstack((np.reshape(X,(np.prod(X.shape),1)),np.reshape(Y,(np.prod(Y.shape),1)), np.reshape(Z,(np.prod(Z.shape),1)))).astype(np.float32)
    u0=np.ones((tx['center'].shape[0],1),np.float32)+ 1j*np.zeros((tx['center'].shape[0],1),np.float32)
    u0*=amp

    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BabelViscoFDTD's Rayleigh Integral solver
    pressure_babelvisco = ForwardSimple(cwvnb_extlay,
                                        center=tx['center'].astype(np.float32),
                                        ds=tx['ds'].astype(np.float32),
                                        u0=u0,
                                        rf=rf,
                                        deviceMetal=gpu_device)
    
    pressure_babelvisco=np.abs(np.reshape(pressure_babelvisco,X.shape)*lossless_medium['density']*lossless_medium['sos'])
    axial_pressure_babelvisco = pressure_babelvisco[pressure_babelvisco.shape[0]//2,pressure_babelvisco.shape[1]//2,:]
    
    # =========================================================================
    # RESULTS CLEANUP
    # =========================================================================
    
    # Remove infinite values from results (Truth method will produce one at focal spot)
    mask = np.isfinite(axial_pressure_truth) & np.isfinite(axial_pressure_babelvisco)
    
    axial_pressure_truth = axial_pressure_truth[mask]
    axial_pressure_babelvisco = axial_pressure_babelvisco[mask]
    axial_coords = axial_coords[mask]
    
    removed_elements_num = abs(len(axial_pressure_truth) - len(mask))
    if removed_elements_num:
        logging.info(f"Removed {removed_elements_num} inf values from results")
    
    logging.info(f"\nTruth max: {axial_pressure_truth.max()}\nTruth min: {axial_pressure_truth.min()}\nTruth mean: {axial_pressure_truth.mean()}")
    logging.info(f"\nBabelViscoFDTD max: {axial_pressure_babelvisco.max()}\nBabelViscoFDTD min: {axial_pressure_babelvisco.min()}\nBabelViscoFDTD mean: {axial_pressure_babelvisco.mean()}")
    
    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [axial_pressure_truth,axial_pressure_babelvisco, abs(axial_pressure_babelvisco-axial_pressure_truth)]
    plot_names = ["Truth", "BabelViscoFDTD", "Difference"]
    screenshot = get_line_plot(axial_coords, data_list=plots, labels=plot_names, title = "Axial Pressure")
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    relative_L2_percent = 100.0 * np.sqrt(np.sum((axial_pressure_babelvisco - axial_pressure_truth)**2) / np.sum(axial_pressure_truth**2) )
    logging.info(f"Relative L2: {relative_L2_percent}%")
    
    assert relative_L2_percent > 1, f"We expected BabelViscoFDTD's rayleigh solver to fail due to a low spatial step (~size of lambda), however it produced similar resuts to truth method"