import datetime
import os
import sys
sys.path.append('./BabelViscoFDTD/')
import platform
import shutil
import re
import configparser
import logging

import base64
import h5py
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is noninteractive
import matplotlib.pyplot as plt
import nibabel
from nibabel import processing, nifti1, affines
import numpy as np
np.random.seed(42) # RNG is same every time
from PySide6.QtCore import Qt
import pytest
import pytest_html
import pyvista as pv
import SimpleITK as sitk
from skimage.metrics import structural_similarity, mean_squared_error
import trimesh

# ================================================================================================================================
# FOLDER/FILE PATHS
# ================================================================================================================================
config = configparser.ConfigParser()
config.read('Tests' + os.sep + 'config.ini')
gpu_device = config['GPU']['device_name']               # GPU device used for test
print('Using GPU device: ',gpu_device)
ref_output_dir = f"Tests/Generate_Outputs/Generated_Outputs/{config['Paths']['ref_output_folder_1_name']}/"   # Folder containing previously generated outputs. Used in regression tests
ref_output_dir_2 = f"Tests/Generate_Outputs/Generated_Outputs/{config['Paths']['ref_output_folder_2_name']}/"   # Folder containing previously generated outputs. Used in test_two_outputs tests
gen_output_dir = f"Tests/Generate_Outputs/Generated_Outputs/{config['Paths']['gen_output_folder_name']}/"   # Folder to store newly generated outputs. Used for generate_outputs "test"
REPORTS_DIR = "PyTest_Reports"

# ================================================================================================================================
# PARAMETERS
# ================================================================================================================================

computing_backends = [
    # {'type': 'CPU','supported_os': ['Mac','Windows','Linux']},
    {'type': 'OpenCL','supported_os': ['Windows','Linux']},
    {'type': 'CUDA',  'supported_os': ['Windows','Linux']},
    {'type': 'Metal', 'supported_os': ['Mac']},
    {'type': 'MLX',   'supported_os': ['Mac']} # Linux too?
]
spatial_step = {
    'Low_Res': 0.919,  # 200 kHz,   6 PPW
    'Med_Res': 0.306,  # 600 kHz,   6 PPW
    'High_Res': 0.184,  # 1000 kHz,  6 PPW
    'Stress_Res': 0.092,  # 1000 kHz, 12 PPW
}
freq_ppw = {
    'Low_Res': [200e3, 6],
    'Med_Res': [600e3, 6],
    'High_Res': [1000e3, 6],
    'Stress_Res': [1000e3,12]
}

# ================================================================================================================================
# PYTEST FIXTURES
# ================================================================================================================================
@pytest.fixture()
def check_files_exist():

    def _check_files_exist(fnames):
        missing_files = []
        for file in fnames:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            return False, missing_files
        else:
            return True, ""

    return _check_files_exist

@pytest.fixture()
def load_files(check_files_exist):
    
    def _load_files(fnames,nifti_load_method='nibabel',skip_test=True):

        if isinstance(fnames,dict):
            datas = fnames.copy()
            fnames_list = fnames.values()
        else:
            datas = []
            fnames_list = fnames

        # Check files exist
        files_exist, missing_files = check_files_exist(fnames_list)

        if not files_exist:
            if skip_test:
                logging.warning(f"Following files are missing: {', '.join(missing_files)}")
                pytest.skip(f"Skipping test because the following files are missing: {', '.join(missing_files)}")
            else:
                raise FileNotFoundError(f"Following files are missing: {', '.join(missing_files)}")

        # Load files based on their extensions
        if isinstance(datas, dict):
            # Iterate over dictionary keys and values directly
            for key, fname in fnames.items():
                datas[key] = _load_file(fname, nifti_load_method)
        else:
            # For lists, just iterate over file names
            for fname in fnames_list:
                datas.append(_load_file(fname,nifti_load_method))

        return datas
    
    def _load_file(fname, nifti_load_method='nibabel'):
        """Helper function to load a single file based on its extension."""
        
        # Get file extension type
        base, ext = os.path.splitext(fname)
        
        # Repeat for compressed files
        if ext == '.gz':
            base, ext = os.path.splitext(base)

        # Load file using appropriate method
        if ext == '.npy':
            return np.load(fname)
        elif ext == '.stl':
            return trimesh.load(fname)
        elif ext == '.nii':
            if nifti_load_method == 'nibabel':
                return nibabel.load(fname)
            elif nifti_load_method == 'sitk':
                return sitk.ReadImage(fname)
            else:
                raise ValueError(f"Invalid nifti load method specified: {nifti_load_method}")
        elif ext == '.txt':
            with open(fname, 'r') as file:
                content = file.read()
                return content
        else:
            logging.warning(f"Unsupported file extension, {fname} not loaded")

    return _load_files

@pytest.fixture()
def check_os(computing_backend):
    sys_os = None
    sys_platform = platform.platform(aliased=True)
    
    if 'macOS' in sys_platform:
        sys_os = 'Mac'
    elif 'Windows' in sys_platform:
        sys_os = 'Windows'
    elif 'Linux' in sys_platform:
        sys_os = 'Linux'
    else:
        logging.warning("No idea what os you're using")

    if sys_os not in computing_backend['supported_os']:
        pytest.skip("Skipping test because the selected computing backend is not available on this system")

@pytest.fixture(scope="session")
def get_gpu_device():
    
    def _get_gpu_device():
        return gpu_device

    return _get_gpu_device

@pytest.fixture(scope="session")
def get_config_dirs():
    config_dirs = {}
    config_dirs["ref_dir_1"] = ref_output_dir
    config_dirs["ref_dir_2"] = ref_output_dir_2
    config_dirs["gen_output_dir"] = gen_output_dir
    return config_dirs

@pytest.fixture()
def get_rmse():
    def _get_rmse(output_points, truth_points):
        rmse = np.sqrt(np.mean((output_points - truth_points) ** 2))
        data_range = np.max(truth_points) - np.min(truth_points)
        norm_rmse = rmse / data_range

        return rmse, data_range, norm_rmse
        
    return _get_rmse

@pytest.fixture()
def get_resampled_input(load_files):
    def _get_resampled_input(input,new_zoom,output_fname):

        if input.ndim > 3:
            tmp_data = input.get_fdata()[:,:,:,0]
            tmp_affine = input.affine
            input = nifti1.Nifti1Image(tmp_data,tmp_affine)

        # Determine new output dimensions and affine
        zooms = np.asarray(input.header.get_zooms())
        new_zooms = np.full(3,new_zoom)
        logging.info(f"Original zooms: {zooms}")
        logging.info(f"New zooms: {new_zooms}")
        new_x_dim = int(input.shape[0]/(new_zooms[0]/zooms[0]))
        new_y_dim = int(input.shape[1]/(new_zooms[1]/zooms[1]))
        new_z_dim = int(input.shape[2]/(new_zooms[2]/zooms[2]))
        new_affine = affines.rescale_affine(input.affine.copy(),
                                                input.shape,
                                                new_zooms,
                                                (new_x_dim,new_y_dim,new_z_dim))

        # Create output
        output_data = np.zeros((new_x_dim,new_y_dim,new_z_dim),dtype=np.uint8)
        output_nifti = nifti1.Nifti1Image(output_data,new_affine)
        logging.info(f"New Dimensions: {output_data.shape}")
        logging.info(f"New Size: {output_data.size}")

        try:
            logging.info('Reloading resampled input')
            resampled_nifti = load_files([output_fname],skip_test=False)[0]
            resampled_data = resampled_nifti.get_fdata()
        except:
            logging.info("File doesn't exist")
            logging.info('Generating resampled input')
            resampled_nifti = processing.resample_from_to(input,output_nifti,mode='constant',order=0,cval=input.get_fdata().min()) # Truth method
            resampled_data = resampled_nifti.get_fdata()
            logging.info('Saving file for future use')
            nibabel.save(resampled_nifti,output_fname)

        # Check data is contiguous
        if not resampled_data.flags.contiguous:
            logging.info("Changing resampled input data to be a contiguous array")
            resampled_data = np.ascontiguousarray(resampled_data)

        return resampled_nifti, resampled_data
    
    return _get_resampled_input

@pytest.fixture()
def check_data():
    def isometric_check(nifti):
        logging.info('Running isometric check')
        zooms = nifti.header.get_zooms()
        logging.info(f"Zooms: {zooms}")
        diffs = np.abs(np.subtract.outer(zooms, zooms))
        isometric = np.all(diffs <= 1e-6)

        return isometric

    # Return the fixture object with the specified attribute
    return {'isometric': isometric_check}

@pytest.fixture()
def compare_data(get_rmse):

    def array_data(output_array,truth_array):
        logging.info('Calculating root mean square error')

        array_rmse = array_range = array_norm_rmse = None

        # Check array size
        if len(output_array) == len(truth_array):
            logging.info(f"Number of array points are equal: {len(output_array)}")
            array_length_same = True

            array_rmse, array_range, array_norm_rmse = get_rmse(output_array,truth_array)
            if array_norm_rmse > 0:
                logging.warning(f"Array had a root mean square error of {array_rmse}, range of {array_range}, and a normalized RMSE of {array_norm_rmse}")
        else:
            logging.error(f"# of array points in output ({len(output_array)}) vs truth ({len(truth_array)})")
            array_length_same = False
        
        return array_length_same, array_norm_rmse
    
    def bhattacharyya_coefficient(arr1,arr2,num_bins=None):

        # Check arrays are not empty
        if arr1.size == 0 or arr2.size == 0:
            pytest.fail("One or both arrays are empty")

        # Determine range of values. We extended the range slightly so bins are divided at 0.5 marks 
        # instead of 1.0 (e.g. -0.5, 0.5, 1.5,...) as array values are more likely to exist at integer 
        # values and helps prevent errors when values lie exactly at bin edge
        min_val = int(np.floor(min(arr1.min(),arr2.min()))) - 0.5
        max_val = int(np.ceil(max(arr1.max(),arr2.max()))) + 0.5
        logging.debug(f"Using {min_val} to {max_val} range for bhatt coeff calculation")
        
        
        # Determine number of bins if argument is not supplied
        if num_bins is None:
            num_bins = int(max_val - min_val)
        logging.debug(f"Using {num_bins} bins for bhatt coeff calculation")
        
        # Get and normalize histograms
        hist1,_ = np.histogram(arr1,bins=num_bins,range=(min_val,max_val))
        hist2,_ = np.histogram(arr2,bins=num_bins,range=(min_val,max_val))
        norm_hist1 = hist1 / np.sum(hist1)
        norm_hist2 = hist2 / np.sum(hist2)

        # Compute Bhattacharyya coefficient
        logging.info('Calculating Bhattacharyya Coefficient')
        bhatt_coefficent = np.sum(np.sqrt(norm_hist1 * norm_hist2))
        logging.info(f"Bhattacharyya coefficient : {bhatt_coefficent}")

        return bhatt_coefficent

    def dice_coefficient(output_array,truth_array,abs_tolerance=1e-8,rel_tolerance=1e-05):
        logging.info('Calculating dice coefficient')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")
        
        if output_array.dtype == bool:
            matches = output_array == truth_array
        else:
            matches = np.isclose(output_array,truth_array,atol=abs_tolerance,rtol=rel_tolerance)
        matches_count = len(matches[matches==True])

        dice_coeff = 2 * matches_count / (output_array.size + truth_array.size)
        logging.info(f"DICE Coefficient: {dice_coeff}")
        return dice_coeff
    
    def h5_data(h5_ref_path,h5_test_path,tolerance=0):
        mismatches = []
        
        def compare_items(name, obj1):
            if name not in f2:
                logging.warning(f"{name} missing in test file")
                mismatches.append(name)
                return
            obj2 = f2[name]
            if isinstance(obj1, h5py.Dataset):
                data1, data2 = obj1[()], obj2[()]
                if not np.allclose(data1, data2, rtol=tolerance, atol=0):
                # if not np.array_equal(data1, data2):
                    if data1.size > 1:
                        logging.warning(f"Dataset {name} differs")
                    else:
                        logging.warning(f"Dataset {name} differs: {data1} vs {data2}")
                    mismatches.append(name)
            elif isinstance(obj1, h5py.Group):
                pass  # groups are containers, children checked recursively
                
        with h5py.File(h5_ref_path, "r") as f1, h5py.File(h5_test_path, "r") as f2:
            exact_match = f1.visititems(lambda name, obj: compare_items(name, obj1=obj))
            
        return len(mismatches) == 0
    
    def mse(output_array,truth_array):
        logging.info('Calculating mean square error')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")
        
        mean_square_error = mean_squared_error(output_array, truth_array)
        return mean_square_error
    
    def ssim(output_array,truth_array,win_size=7,data_range=None):
        logging.info('Calculating structural similarity')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")

        score = structural_similarity(output_array, truth_array, win_size=win_size,data_range=data_range)
        return score
    
    def stl_area(output_stl,truth_stl):
        logging.info('Calculating percent error of stl area')
        
        # Check STL area
        percent_error_area = abs((output_stl.area - truth_stl.area)/truth_stl.area)
        if percent_error_area > 0:
            logging.warning(f"STL area had a percent error of {percent_error_area*100}%")
        else:
            logging.info(f"STL area is identical ({output_stl.area})")
        
        return percent_error_area

    # Return the fixture object with the specified attribute
    return {'array_data': array_data,'bhatt_coeff': bhattacharyya_coefficient,'dice_coefficient': dice_coefficient,'h5_data': h5_data,'mse': mse,'ssim': ssim,'stl_area': stl_area}

@pytest.fixture()
def extract_nib_info():
    def _extract_nib_info(nifti_nib):
        zooms = np.asarray(nifti_nib.header.get_zooms())[:3]
        affine = nifti_nib.affine
        data = np.squeeze(nifti_nib.get_fdata())

        return zooms, affine, data
    
    return _extract_nib_info

@pytest.fixture()
def extract_sitk_info():
    def _extract_sitk_info(nifti_sitk):
        spacing = np.asarray(nifti_sitk.GetSpacing())
        direction = np.asarray(nifti_sitk.GetDirection())
        origin = np.asarray(nifti_sitk.GetOrigin())
        data = sitk.GetArrayFromImage(nifti_sitk)

        return spacing, direction, origin, data
    
    return _extract_sitk_info

@pytest.fixture()
def image_to_base64():
    def _image_to_base64(image_path):
        # Ensure the file exists
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"File {image_path} does not exist.")
        
        # Open the image file in binary mode
        with image_path.open("rb") as image_file:
            # Read the binary data from the file
            image_data = image_file.read()
            
            # Encode the binary data to a base64 string
            base64_string = base64.b64encode(image_data).decode('utf-8')
        
        return base64_string
    
    return _image_to_base64

@pytest.fixture()
def get_mpl_plot():
    def _get_mpl_plot(datas, axes_num=1, titles=None, color_map='viridis', colorbar=False, clim=None,measurement_plane_index=None,extent=None,extent_units="mm"):
        """
        Create one or multiple Matplotlib plots for 2D or 3D data.

        Parameters
        ----------
        datas : list of np.ndarray
            List of 2D or 3D arrays to plot.
        axes_num : int
            Number of slices or axes to plot (e.g., 1, 2, or 3 for 3D data).
        titles : list of str, optional
            Titles for each data array.
        color_map : str
            Colormap for imshow.
        colorbar : bool
            Whether to show colorbars.
        clim : tuple, optional
            (vmin, vmax) for color limits.
        """
        data_num = len(datas)
        fig, axs = plt.subplots(axes_num, data_num, figsize=(data_num * 2.5, axes_num * 2.5),constrained_layout=True)

        # Normalize axs to a 2D numpy array for consistent indexing
        if isinstance(axs, np.ndarray):
            if axs.ndim == 1:
                if axes_num == 1:  # shape (data_num,)
                    axs = axs[np.newaxis, :]  # (1, data_num)
                elif data_num == 1:  # shape (axes_num,)
                    axs = axs[:, np.newaxis]  # (axes_num, 1)
            # else already 2D
        else:
            axs = np.array([[axs]])  # scalar → (1,1)

        for i_axis in range(axes_num):
            for i_data, data in enumerate(datas):
                # Handle both 2D and 3D arrays
                if data.ndim == 2:
                    img_data = data
                elif data.ndim == 3:
                    # Compute midpoint along the relevant axis
                    midpoint = data.shape[i_axis % 3] // 2
                    
                    if i_axis % 3 == 0:
                        img_data = data[midpoint, :, :].T
                        if extent is not None:
                            extent_2d = extent[2:]
                    elif i_axis % 3 == 1:
                        img_data = data[:, midpoint, :].T
                        if extent is not None:
                            extent_2d = extent[[0, 1, -2, -1]]
                    else:
                        if measurement_plane_index is not None:
                            img_data = data[:, :, measurement_plane_index].T
                        else:
                            img_data = data[:, :, midpoint].T
                        if extent is not None:
                            extent_2d = extent[:4]
                else:
                    raise ValueError(f"Unsupported data dimension: {data.ndim}")

                ax = axs[i_axis, i_data]
                if extent is not None:
                    im = ax.imshow(img_data, cmap=color_map,extent=extent_2d)
                    
                    if i_axis == 0:
                        ax.set_xlabel(f"y ({extent_units})")
                        ax.set_ylabel(f"z ({extent_units})")
                    elif i_axis == 1:
                        ax.set_xlabel(f"x ({extent_units})")
                        ax.set_ylabel(f"z ({extent_units})")
                    else:
                        ax.set_xlabel(f"x ({extent_units})")
                        ax.set_ylabel(f"y ({extent_units})")
                else:
                    im = ax.imshow(img_data, cmap=color_map)

                # Titles only on first row
                if titles is not None and i_axis == 0:
                    ax.set_title(titles[i_data], fontsize=12)

                if colorbar:
                    if clim is not None:
                        im.set_clim(clim)
                    
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="4%", pad=0.1)
                    cbar = fig.colorbar(im, cax=cax)
                    
                    cbar.formatter.set_scientific(True)
                    cbar.formatter.set_powerlimits((-2, 2))  # always scientific
                    cbar.update_ticks()
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return base64_plot

    return _get_mpl_plot

@pytest.fixture()
def get_line_plot():
    def _get_line_plot(x, data_list, labels=None, title=None, xlabel='X', ylabel='Y'):
        """
        Plot multiple data arrays against the same x-coordinates.

        Parameters
        ----------
        x : array-like
            X-coordinates for the plot.
        data_list : list of array-like
            List of Y data arrays to plot.
        labels : list of str, optional
            Labels for each data array.
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label for x-axis.
        ylabel : str, optional
            Label for y-axis.
        """
        # Cycle through line styles
        line_styles = ['-', '--', ':','-.']
    
        plt.figure(figsize=(7.5, 2.5))

        for i, y in enumerate(data_list):
            label = labels[i] if labels and i < len(labels) else f"Line {i+1}"
            line_i = i
            while i >= len(line_styles):
                line_i -= len(line_styles)
            line_style = line_styles[line_i]
            plt.plot(x, y, label=label,linestyle=line_style)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png',bbox_inches='tight')
        buffer.seek(0)
        
        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_plot
    
    return _get_line_plot

@pytest.fixture
def get_pyvista_plot():

    def intersection_plot(mesh1,mesh2,mesh3):
        # Create pyvista plot
        plotter = pv.Plotter(window_size=(400, 400),off_screen=True)
        plotter.background_color = 'white'
        plotter.add_mesh(mesh1, opacity=0.2)
        plotter.add_mesh(mesh2, opacity=0.2)
        plotter.add_mesh(mesh3, opacity=0.5, color='red')

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)
        
        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_plot
    
    def mesh_plot(meshes,title=''):

        # Create pyvista plot
        plotter = pv.Plotter(window_size=(500, 500),off_screen=True)
        plotter.background_color = 'white'
        for mesh in meshes:
            plotter.add_mesh(pv.wrap(mesh),opacity=0.5)
        plotter.add_title(title, font_size=12)
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)

        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_plot
    
    def voxel_plot(mesh,Points,title=''):
        # Create points mesh
        step = Points.shape[0]//1000000 # Plot 1000000 points
        points_mesh =  pv.PolyData(Points[::step,:])

        # Create pyvista plot
        plotter = pv.Plotter(window_size=(500, 500),off_screen=True)
        plotter.background_color = 'white'
        plotter.add_mesh(pv.wrap(mesh),opacity=0.5)
        plotter.add_mesh(points_mesh,color='blue',opacity=0.1)
        plotter.add_title(title,font_size=12)
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)

        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_plot
    
    # Return the fixture object with the specified attribute
    return {'intersection_plot': intersection_plot,'mesh_plot': mesh_plot,'voxel_plot': voxel_plot}

@pytest.fixture()
def get_example_data():
    def numpy_data(dims = (4,4)):
        return np.random.random(dims)
    
    def nifti_nib_data(dims=(256,256,128)):
        affine = np.random.rand(4,4)
        data = np.random.random(dims)
        nibabel_nifti = nibabel.nifti1.Nifti1Image(data,affine)
        nibabel_nifti.header.set_zooms(np.random.rand(3))
        
        return nibabel_nifti
    
    def nifti_sitk_data():
        data = np.random.rand(256,256,128)
        nibabel_sitk = sitk.GetImageFromArray(data)

        # Set the spacing, direction, and origin in the SimpleITK image
        nibabel_sitk.SetSpacing(np.random.rand(3))
        nibabel_sitk.SetDirection(np.random.rand(3,3))
        nibabel_sitk.SetOrigin(np.random.rand(3))
        
        return nibabel_sitk
    
    # Return the fixture object with the specified attribute
    return {'numpy': numpy_data,
            'nifti_nib':nifti_nib_data,
            'nifti_sitk':nifti_sitk_data}

@pytest.fixture()
def set_up_domain():
    def _get_material_list_bhte():
        
        material_list = {}                 #Water    #Water      #Blood      #Brain      #Skull      #Skin
        material_list['Density']         = [1000.0,  1000.0,     1050.0,     1041.0,     1041.0,     1041.0,    ]   # (kg/m3)
        material_list['SoS']             = [1500.0,  1500.0,     1570.0,     1562.0,     1562.0,     1562.0,    ]   # (m/s)
        material_list['Attenuation']     = [0.0,     0.0,        0.0,        3.45,       3.45,       3.45,      ]   # (Np/m)
        material_list['SpecificHeat']    = [4178.0,  4178.0,     3617.0,     3630.0,     3630.0,     3630.0,    ]   # (J/kg/°C)
        material_list['Conductivity']    = [0.6,     0.6,        0.52,       0.51,       0.51,       0.51,      ]   # (W/m/°C)
        material_list['Perfusion']       = [0.0,     0.0,        10000.0,    559.0,      559.0,      559.0,     ]   # (ml/min/kg)
        material_list['Absorption']      = [0.0,     0.0,        0.0,        0.85,       0.85,       0.85,      ]   # Unitless
        material_list['InitTemperature'] = [37.0,    37.0,       37.0,       37.0,       37.0,       37.0,      ]   # (°C)
        # Water material is duplicated since metal compute sims run into issues when
        # trying to run in homogenous material (like water) and its properties are stored in index 0

        material_indices = {"water": 1, "blood": 2, "brain":3, "skull": 4, "skin": 5}
        
        return material_list, material_indices
    
    def _get_material_list_vwe(freq):
        
        def FitSpeedCorticalLong(frequency):
            #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014 
            FRef=np.array([270e3,836e3])
            ClRef=np.array([2448.0,2516])
            p=np.polyfit(FRef, ClRef, 1)
            return(np.round(np.poly1d(p)(frequency)))
        
        def FitSpeedCorticalShear(frequency):
            #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
            FRef=np.array([270e3,836e3])
            Cs270=np.array([1577.0,1498.0,1313.0]).mean()
            Cs836=np.array([1758.0,1674.0,1545.0]).mean()
            CsRef=np.array([Cs270,Cs836])
            p=np.polyfit(FRef, CsRef, 1)
            return(np.round(np.poly1d(p)(frequency)))
        
        def FitAttCorticalLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
            # fitting from data obtained from
            #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
            # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
            # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
            
            return np.round(203.25090263*((frequency/1e6)**bcoeff)*reductionFactor)

        def FitAttBoneShear(frequency,reductionFactor=1.0):
            #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
            PichardoData=(57.0/.27 +373/0.836)/2
            return np.round(PichardoData*(frequency/1e6)*reductionFactor) 
        
        def FitSpeedTrabecularLong(frequency):
            #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
            FRef=np.array([270e3,836e3])
            ClRef=np.array([2140.0,2300])
            p=np.polyfit(FRef, ClRef, 1)
            return(np.round(np.poly1d(p)(frequency)))

        def FitSpeedTrabecularShear(frequency):
            #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
            FRef=np.array([270e3,836e3])
            Cs270=np.array([1227.0,1365.0,1200.0]).mean()
            Cs836=np.array([1574.0,1252.0,1327.0]).mean()
            CsRef=np.array([Cs270,Cs836])
            p=np.polyfit(FRef, CsRef, 1)
            return(np.round(np.poly1d(p)(frequency)))

        def FitAttTrabecularLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
            #reduction factor 
            # fitting from data obtained from
            #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
            # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
            # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
            return np.round(202.76362433*((frequency/1e6)**bcoeff)*reductionFactor) 
        
        material_list={}                        #Density (kg/m3)    LongSoS (m/s),                  ShearSoS (m/s),                 Long Att (Np/m),                        Shear Att (Np/m)
        material_list['water']       = np.array([1000.0,            1500.0,                         0,                              0.0,                                    0] )
        material_list['cortical']    = np.array([1896.5,            FitSpeedCorticalLong(freq),     FitSpeedCorticalShear(freq),    FitAttCorticalLong_Multiple(freq),      FitAttBoneShear(freq)])
        material_list['trabecular']  = np.array([1738.0,            FitSpeedTrabecularLong(freq),   FitSpeedTrabecularShear(freq),  FitAttTrabecularLong_Multiple(freq),    FitAttBoneShear(freq)])
        material_list['skin']        = np.array([1116.0,            1537.0,                         0.0,                            2.3*freq/500e3 ,                        0.0])
        material_list['brain']       = np.array([1041.0,            1562.0,                         0.0,                            3.45*freq/500e3 ,                       0.0])
    
        return material_list

    def _set_medium_bhte(medium_type='brain'):
        
        # Indices for materials
        MaterialList,material_index_dict = _get_material_list_bhte()
        medium_index = material_index_dict[medium_type]
        blood_index = material_index_dict['blood']
        
        # Define medium properties
        medium = {}
        # Medium properties specific to diffusion
        medium['density']               = MaterialList['Density'][medium_index]
        medium['thermal_conductivity']  = MaterialList['Conductivity'][medium_index]
        medium['specific_heat']         = MaterialList['SpecificHeat'][medium_index]
        # Blood properties specific to perfusion
        medium['blood_density']             = MaterialList['Density'][blood_index]
        medium['blood_specific_heat']       = MaterialList['SpecificHeat'][blood_index]
        medium['blood_perfusion_rate']      = MaterialList['Perfusion'][medium_index]*(1/60)*(1e-6)*medium['density'] # Need units in 1/s for truth method
        medium['blood_ambient_temperature'] = MaterialList['InitTemperature'][blood_index]
        # Medium properties specific to heat disposition
        medium['sos'] = MaterialList['SoS'][medium_index]
        medium['attenuation'] = MaterialList['Attenuation'][medium_index]
        medium['absorption'] = MaterialList['Absorption'][medium_index]
        logging.info(medium)
        
        return medium, medium_index
        
    def _create_grid(grid_limits=[], grid_steps=[], pml_thickness=0):
        """
        Create a 2D or 3D computational grid with exact grid_steps,
        including optional PML extension.
        """

        # Validate input as 2D or 3D
        dim = len(grid_steps)
        if dim not in (2, 3):
            raise ValueError("grid_steps must have 2 (2D) or 3 (3D) elements.")

        # Determine grid limits and spatial steps
        if dim == 3:
            if len(grid_limits) == 6:
                xmin, xmax, ymin, ymax, zmin, zmax = grid_limits
            elif len(grid_limits) == 3:
                xmin = ymin = zmin = 0.0
                xmax, ymax, zmax = grid_limits
            else:
                raise ValueError("Invalid grid_limits for 3D.")
            dx, dy, dz = grid_steps
        else:
            if len(grid_limits) == 4:
                xmin, xmax, ymin, ymax = grid_limits
            elif len(grid_limits) == 2:
                xmin = ymin = 0.0
                xmax, ymax = grid_limits
            else:
                raise ValueError("Invalid grid_limits for 2D.")
            dx, dy = grid_steps

        # Determine intervals
        x = np.arange(xmin - pml_thickness*dx, xmax + 0.5*dx + pml_thickness*dx, dx)
        y = np.arange(ymin - pml_thickness*dy, ymax + 0.5*dy + pml_thickness*dy, dy)
        if dim == 3:
            z = np.arange(zmin - pml_thickness*dz, zmax + 0.5*dz + pml_thickness*dy, dz)
        
        # Create and return meshgrid
        if dim == 3:
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            return X, Y, Z
        else:
            X, Y = np.meshgrid(x, y, indexing="ij")
            return X, Y
        
    return {'grid': _create_grid,
            'medium_bhte': _set_medium_bhte,
            'material_list_bhte': _get_material_list_bhte,
            'material_list_vwe': _get_material_list_vwe}

@pytest.fixture()
def setup_propagation_model(set_up_domain,get_mpl_plot,get_line_plot,request):
    def _setup_propagation_model(us_frequency, points_per_wavelength, axes = 3, map_type="mixed",source_shape ="circular",skip_plots=False):

        # =============================================================================
        # SIMULATION PARAMETERS
        # =============================================================================

        dt = 4e-8                       # time step
        medium_SOS = 1500               # m/s - water
        medium_density = 1000           # kg/m3
        pml_thickness = 12              # grid points for perfect matching layer
        reflection_limit = 1.0000e-05   # reflection parameter for PML
        us_amplitude = 100e3            # Pa
        sensor_ppp = 8                  # sensor points per period

        # Properties
        shortest_wavelength = medium_SOS/us_frequency
        spatial_step = shortest_wavelength/ points_per_wavelength
        
        # =============================================================================
        # SIMULATION DOMAIN SETUP
        # =============================================================================
        
        # Domain Dimensions
        if axes == 3:
            if map_type in ("mixed","water"):
                x_dim = y_dim = 0.05
                z_dim = 0.10
            elif map_type == "bone":
                bone_thickness = 0.002  # 2mm
                x_dim = y_dim = shortest_wavelength*20
                z_dim = 3*spatial_step + (3*bone_thickness) + (8*shortest_wavelength)
            elif map_type == "brain":
                brain_thickness = 0.008 # 8mm
                x_dim = y_dim = shortest_wavelength*20
                z_dim = 3*spatial_step + brain_thickness + (8*shortest_wavelength)
            else:
                raise ValueError("Invalid map_type provided to test")        
        else:
            x_dim = 0.20    # m
            y_dim = 0.40    # m

        logging.info(f"Domain Dimensions: {x_dim*1e3} mm x {y_dim*1e3} mm * {z_dim*1e3} mm")
        
        # Transducer Dimensions
        if map_type == "mixed":
            tx_radius = 0.025   # m
            tx_plane_loc = 0.01 # m
            tx_loc = int(np.round(tx_plane_loc/spatial_step)) + pml_thickness
        else:
            tx_radius = x_dim/2
            tx_loc = 1 + pml_thickness
            
        # Create meshgrid
        if axes == 3:
            X, Y, Z = set_up_domain['grid'](grid_limits=[-x_dim/2, x_dim/2, -y_dim/2, y_dim/2, 0, z_dim],
                                            grid_steps=3*[spatial_step],
                                            pml_thickness=pml_thickness)
        else:
            X, Y = set_up_domain['grid'](grid_limits=[-x_dim/2, x_dim/2, 0, y_dim],
                                         grid_steps=2*[spatial_step],
                                         pml_thickness=pml_thickness)
        logging.info(f'Domain size: {X.shape}')

        # Calculate time to cross from one corner to opposite corner
        if axes == 3:
            sim_time = np.sqrt(x_dim**2+y_dim**2+z_dim**2)/medium_SOS 
        else:
            sim_time = np.sqrt(x_dim**2+y_dim**2)/medium_SOS

        # Number of sensor steps i.e. how many timepoints we record babelvisco results
        sensor_steps = int((1/us_frequency/sensor_ppp)/dt) # for the sensors, we do not need really high temporal resolution, so we are keeping 8 time points per period

        # =============================================================================
        # MATERIAL MAP SETUP
        # =============================================================================

        # Retrieve list of different materials (e.g. brain, skull, etc.)
        material_list = set_up_domain["material_list_vwe"](us_frequency)
        ml_keys, ml_values = zip(*material_list.items())
        ml_keys = np.array(ml_keys)
        material_list = np.array(ml_values)
        
        # Map value for each tissue type
        index_water = np.where(np.array(ml_keys) == "water")[0][0]
        index_cortical = np.where(np.array(ml_keys) == "cortical")[0][0]
        index_trabecular = np.where(np.array(ml_keys) == "trabecular")[0][0]
        index_skin = np.where(np.array(ml_keys) == "skin")[0][0]
        index_brain = np.where(np.array(ml_keys) == "brain")[0][0]

        # Initialize material map as all water
        water_map = index_water * np.ones_like(X,np.uint32)
        material_map = water_map.copy()

        # Modify material map based on map type
        if map_type == "mixed":
            # Add spheres of different materials at different locations
            def add_material_sphere(material_index,material_radius,material_center,center_offsets):
                material_center[0] += center_offsets[0]
                material_center[1] += center_offsets[1]
                if axes == 3:
                    material_center[2] += center_offsets[2]

                if axes == 3:
                    r = np.sqrt((X - material_center[0])**2 + (Y - material_center[1])**2 + (Z - material_center[2])**2)
                else:
                    r = np.sqrt((X - material_center[0])**2 + (Y - material_center[1])**2)
                material_mask = r <= material_radius
                material_map[material_mask] = material_index

            if axes == 3:
                mat_radius = tx_radius/2
                add_material_sphere(index_cortical,mat_radius,[0, 0, z_dim/2],center_offsets=[0,-1*mat_radius,0])
                add_material_sphere(index_brain, mat_radius,[0, 0, z_dim/2],center_offsets=[0,mat_radius,2*mat_radius])
                add_material_sphere(index_skin,mat_radius,[0, 0, z_dim/2],center_offsets=[0,mat_radius,-2*mat_radius])
            else:
                mat_radius = tx_radius*2
                add_material_sphere(index_cortical,mat_radius,[0, y_dim/2],center_offsets=[-1*mat_radius,0])
                add_material_sphere(index_brain, mat_radius,[0, y_dim/2],center_offsets=[mat_radius,2*mat_radius])
                add_material_sphere(index_skin,mat_radius,[0, y_dim/2],center_offsets=[mat_radius,-2*mat_radius])
        elif map_type == "bone":
            # Add few layers of cortical, trabecular, and more cortical bone
            bone_thickness = 0.002 # 2mm
            bone_layers = int(bone_thickness/spatial_step)
            bone_start = 3 + pml_thickness

            material_map[pml_thickness:-pml_thickness,pml_thickness:-pml_thickness,bone_start:bone_start+bone_layers] = index_cortical
            material_map[pml_thickness:-pml_thickness,pml_thickness:-pml_thickness,bone_start+bone_layers:bone_start+2*bone_layers] = index_trabecular
            material_map[pml_thickness:-pml_thickness,pml_thickness:-pml_thickness,bone_start+2*bone_layers:bone_start+3*bone_layers] = index_cortical
        elif map_type == 'brain':
            # Add layer of brain tissue
            brain_thickness = 0.008 # 8mm
            brain_layers = int(brain_thickness/spatial_step)
            brain_start = 3 + pml_thickness

            material_map[pml_thickness:-pml_thickness,pml_thickness:-pml_thickness,brain_start:brain_start+brain_layers] = index_brain
        elif map_type == "water":
            pass

        # =============================================================================
        # GENERATE SOURCE MAP + SIGNAL
        # =============================================================================

        # Create source mask
        if axes == 3:
            if source_shape == "circular":
                source_mask = (X[:,:,X.shape[2]//2]**2+Y[:,:,Y.shape[2]//2]**2) <= (tx_radius)**2
            elif source_shape == "square":
                source_mask = (np.abs(X[:,:,X.shape[2]//2]) <= tx_radius) & \
                            (np.abs(Y[:,:,Y.shape[2]//2]) <= tx_radius)
        else:
            source_mask = (X[:,X.shape[1]//2]**2) <= (tx_radius)**2
        source_mask = (source_mask*1.0).astype(np.uint32)

        # Create source map
        source_map = np.zeros_like(X,np.uint32)
        if axes == 3:
            source_map[:,:,tx_loc] = source_mask
        else:
            source_map[:,tx_loc] = source_mask

        # Create particle displacement maps
        amp_displacement = us_amplitude/medium_density/medium_SOS
        Ox = np.zeros_like(X)
        Oy = np.zeros_like(X)
        if axes == 3:
            Oz = np.zeros_like(X)
            Oz[source_map > 0] = 1 #only Z has a value of 1
        else:
            Oy[source_map > 0] = 1 #only Z has a value of 1

        Ox *= amp_displacement
        Oy *= amp_displacement
        if axes == 3:
            Oz *= amp_displacement

        # Generate source time signal
        source_length = 4.0/us_frequency # we will use 4 pulses
        source_time_vector = np.arange(0,source_length+dt,dt)

        # Plot source time signal
        pulse_source_tmp = np.sin(2*np.pi*us_frequency*source_time_vector)

        # Note we need expressively to arrange the data in a 2D array
        pulse_source = np.reshape(pulse_source_tmp,(1,len(source_time_vector))) 
        logging.info(f"Number of time points in source signal: {len(source_time_vector)}")

        # =============================================================================
        # GENERATE SENSOR MAP
        # =============================================================================

        # Create sensor map
        sensor_map=np.zeros_like(X,np.uint32)
        if axes == 3:
            sensor_map[pml_thickness:-pml_thickness,X.shape[1]//2,pml_thickness:-pml_thickness] = 1
        else:
            sensor_map[pml_thickness:-pml_thickness,pml_thickness:-pml_thickness] = 1

        # =============================================================================
        # SAVE PLOTS
        # =============================================================================
        
        # dimensions in mm
        xmin = X[0,0,0]*1e3
        xmax = X[-1,0,0]*1e3
        ymin = Y[0,0,0]*1e3
        ymax = Y[0,-1,0]*1e3
        extent = np.array([xmin,xmax,ymin,ymax])
        
        if axes == 3:
            zmin = Z[0,0,0]*1e3
            zmax = Z[0,0,-1]*1e3
            extent = np.append(extent,[zmin,zmax])
    
        if not skip_plots:
            # Source Map Plot
            screenshot = get_mpl_plot([source_map], axes_num=3,titles=['Source Map'],measurement_plane_index=tx_loc,extent=extent)
            request.node.screenshots.append(screenshot)
            
            # Material Map Plot
            screenshot = get_mpl_plot([material_map], axes_num=3,titles=['Material Map'],extent=extent)
            request.node.screenshots.append(screenshot)
            
            # Sensor Map Plot
            screenshot = get_mpl_plot([sensor_map], axes_num=3,titles=['Sensor Map'],extent=extent)
            request.node.screenshots.append(screenshot)
            
            # Source Signal Plot
            screenshot = get_line_plot(source_time_vector*1e6,[pulse_source_tmp],title="Source Signal",xlabel="time (us)")
            request.node.screenshots.append(screenshot)

        # =============================================================================
        # SAVE PARAMETERS TO DICTIONARY
        # =============================================================================

        propagation_model_params = {}
        propagation_model_params['material_map'] = material_map
        propagation_model_params['water_map'] = water_map
        propagation_model_params['material_list'] = material_list
        propagation_model_params['source_map'] = source_map
        propagation_model_params['pulse_source'] = pulse_source
        propagation_model_params['spatial_step'] = spatial_step
        propagation_model_params['sim_time'] = sim_time
        propagation_model_params['sensor_map'] = sensor_map
        propagation_model_params['Ox'] = Ox
        propagation_model_params['Oy'] = Oy
        if axes == 3:
            propagation_model_params['Oz'] = Oz
        propagation_model_params['pml_thickness'] = pml_thickness
        propagation_model_params['reflection_limit'] = reflection_limit
        propagation_model_params['dt'] = dt
        propagation_model_params['sensor_steps'] = sensor_steps
        propagation_model_params['sensor_ppp'] = sensor_ppp
        propagation_model_params['extent'] = extent

        return propagation_model_params

    return _setup_propagation_model

@pytest.fixture()
def get_phase_data():
    def _get_phase_data(time_vector,pressure_vector,sensor_steps,sensor_ppp,sensor_map,material_map,frequency,degrees=False):
        time_step = np.diff(time_vector).mean()
        
        if int(time_vector.shape[0]%(sensor_ppp/sensor_steps)) !=0:
            logging.info('Rounding of time vector was not exact multiple of PPP, truncating time vector a little')
            nDiff = int(time_vector.shape[0]%(sensor_ppp/sensor_steps))
            logging.info(' Cutting %i entries from sensor from length %i to %i' %(nDiff,time_vector.shape[0],time_vector.shape[0]-nDiff))
            time_vector = time_vector[:-nDiff]
            pressure_vector = pressure_vector[:-nDiff]
            
        assert(int(time_vector.shape[0]%(sensor_ppp/sensor_steps))==0)
        
        freqs = np.fft.fftfreq(time_vector.size, time_step)
        IndSpectrum = np.argmin(np.abs(freqs-frequency)) # frequency entry closest to fundamental frequency
        
        pressure_vector = np.ascontiguousarray(pressure_vector)
                
        index = np.nonzero(np.transpose(sensor_map).flatten()>0)[0]
        nStep = 100000
        
        phase_map = np.zeros_like(material_map,np.float32)
        
        for n in range(0,pressure_vector.shape[0],nStep):
            top=np.min([n+nStep,pressure_vector.shape[0]])
            FSignal=np.fft.fft(pressure_vector[n:top,:],axis=1)
            
            k=index[n:top]//(material_map.shape[0]*material_map.shape[1])
            j=index[n:top]%(material_map.shape[0]*material_map.shape[1])
            i=j%material_map.shape[0]
            j=j//material_map.shape[0]
            FSignal=FSignal[:,IndSpectrum]
            pa= np.angle(FSignal)
            
            phase_map[i,j,k] = pa
        
        if degrees:
            return np.degrees(phase_map)
        else:
            return phase_map
    
    return _get_phase_data
        
    
# ================================================================================================================================
# PYTEST HOOKS
# ================================================================================================================================
def pytest_generate_tests(metafunc):
    # Parametrize + mark tests based on fixtures used
    
    if 'computing_backend' in metafunc.fixturenames:
        params = [pytest.param(cb, id=cb['type'], marks=pytest.mark.gpu) for cb in computing_backends]
        metafunc.parametrize("computing_backend", params)

    if 'spatial_step' in metafunc.fixturenames:
        params = []
        for ss_key,ss_value in spatial_step.items():
            if "low" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=pytest.mark.low_res))
            elif "med" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=pytest.mark.medium_res))
            elif "high" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=[pytest.mark.slow,pytest.mark.high_res]))
            elif "stress" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=[pytest.mark.slow,pytest.mark.stress_res]))
            else:
                params.append(pytest.param(ss_value, id=ss_key))
        metafunc.parametrize('spatial_step',params)
    elif 'frequency' in metafunc.fixturenames and 'ppw' in metafunc.fixturenames:
        params = []
        for key,value in freq_ppw.items():
            if "low" in key.lower():
                params.append(pytest.param(value[0],value[1], id=key, marks=pytest.mark.low_res))
            elif "med" in key.lower():
                params.append(pytest.param(value[0],value[1], id=key, marks=pytest.mark.medium_res))
            elif "high" in key.lower():
                params.append(pytest.param(value[0],value[1], id=key, marks=[pytest.mark.slow,pytest.mark.high_res]))
            elif "stress" in key.lower():
                params.append(pytest.param(value[0],value[1], id=key, marks=[pytest.mark.slow,pytest.mark.stress_res]))
            else:
                params.append(pytest.param(value[0],value[1], id=key))
        metafunc.parametrize('frequency,ppw',params)
        
    if 'tolerance' in metafunc.fixturenames:
        metafunc.parametrize('tolerance',
                             [pytest.param(0, marks=pytest.mark.tol_0, id="0%_tolerance"),
                              pytest.param(0.01, marks=pytest.mark.tol_1, id="1%_tolerance"),
                              pytest.param(0.05, marks=pytest.mark.tol_5, id="5%_tolerance")])

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome = yield
    report = outcome.get_result()
    
    if report.when == 'call':
        extras = getattr(report, 'extras', [])

        # Add saved screenshots to html report
        if hasattr(item, 'screenshots'):
            img_tags = ''
            for screenshot in item.screenshots:
                img_tags += "<td><img src='data:image/png;base64,{}' >></td>".format(screenshot)
            extras.append(pytest_html.extras.html(f"<tr>{img_tags}</tr>"))
            
        report.extras = extras
        
    if (report.when == 'call') or (report.when in ['setup', 'teardown'] and report.failed):
        extras = getattr(report, 'extras', [])
        _create_individual_test_report(item, report, extras)

def _create_individual_test_report(item, report, extras):
    '''
    Create an individual HTML report for a single test.

    Parameters
    ----------
    item : pytest.Item
        The test item that was executed.
    report : pytest.TestReport
        The test report containing results and metadata.
    extras : list
        List of extra content (screenshots, logs, etc.) to include in the report.
    '''
    # Create test-specific report directory
    test_reports_dir = os.path.join(REPORTS_DIR, "individual_tests")
    os.makedirs(test_reports_dir, exist_ok=True)
    
    # Generate safe filename from test name
    safe_test_name = re.sub(r'[^\w\-_\.]', '_', item.nodeid)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"{safe_test_name}_{timestamp}.html"
    report_path = os.path.join(test_reports_dir, report_filename)
    
    # Determine test status and styling
    if report.passed:
        status = "PASSED"
        status_color = "#28a745"
    elif report.failed:
        status = "FAILED"
        status_color = "#dc3545"
    elif report.skipped:
        status = "SKIPPED"
        status_color = "#ffc107"
    else:
        status = "UNKNOWN"
        status_color = "#6c757d"
    
    # Build HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Report: {item.name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .status {{ font-weight: bold; color: {status_color}; }}
            .section {{ margin: 20px 0; }}
            .section h3 {{ color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }}
            .code {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; }}
            .error {{ background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            td {{ padding: 5px; border: 1px solid #dee2e6; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Test Report: {item.name}</h1>
            <p><strong>Status:</strong> <span class="status">{status}</span></p>
            <p><strong>Test ID:</strong> {item.nodeid}</p>
            <p><strong>Duration:</strong> {report.duration:.4f} seconds</p>
            <p><strong>Timestamp:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    # Add test parameters if available
    if hasattr(item, 'callspec') and item.callspec.params:
        html_content += """
        <div class="section">
            <h3>Test Parameters</h3>
            <table>
        """
        for param_name, param_value in item.callspec.params.items():
            html_content += f"<tr><td><strong>{param_name}</strong></td><td>{param_value}</td></tr>"
        html_content += "</table></div>"
    
    # Add failure information if test failed
    if report.failed and report.longrepr:
        html_content += f"""
        <div class="section">
            <h3>Failure Details</h3>
            <div class="error">
                <pre>{report.longrepr}</pre>
            </div>
        </div>
        """
    
    # Add skip reason if test was skipped
    if report.skipped and report.longrepr:
        html_content += f"""
        <div class="section">
            <h3>Skip Reason</h3>
            <div class="code">
                {report.longrepr}
            </div>
        </div>
        """
    
    # Add captured output if available
    if hasattr(report, 'capstdout') and report.capstdout:
        html_content += f"""
        <div class="section">
            <h3>Captured Output</h3>
            <div class="code">
                <pre>{report.capstdout}</pre>
            </div>
        </div>
        """
    
    # Add captured logs if available
    if hasattr(report, 'caplog') and report.caplog:
        html_content += f"""
        <div class="section">
            <h3>Captured Logs</h3>
            <div class="code">
                <pre>{report.caplog}</pre>
            </div>
        </div>
        """
    
    # Add extras (screenshots, plots, etc.)
    if extras:
        html_content += """
        <div class="section">
            <h3>Additional Content</h3>
            <table>
        """
        for extra in extras:
            if hasattr(extra, 'content'):
                html_content += f"<tr><td>{extra.content}</td></tr>"
        html_content += "</table></div>"

    if hasattr(item, 'screenshots'):
        html_content += f"""
        <div class="section">
            <h3>Screenshots</h3>
        """
        for screenshot in item.screenshots:
            img_tags = "<td><img src='data:image/png;base64,{}'></td>".format(screenshot)
            html_content += f"""
            <table>
                <tr>{img_tags}</tr>
            </table>
            """
        html_content += f"""
        </div>
        """
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML report to file
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        print(f"Failed to create individual test report: {e}")


@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Hook to modify inline final report"""
    terminalreporter.write_line(f"Total tests run: {terminalreporter._numcollected}")
    terminalreporter.write_line(f"Total failures: {len(terminalreporter.stats.get('failed', []))}")
    terminalreporter.write_line(f"Total passes: {len(terminalreporter.stats.get('passed', []))}")

    if os.path.isfile(os.path.join('PyTest_Reports','report.html')):
    # Change report name to include time of completion
        report_name = f"report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"
        os.rename(os.path.join('PyTest_Reports','report.html'), os.path.join('PyTest_Reports',report_name))
        print(f"Report saved as {report_name}")