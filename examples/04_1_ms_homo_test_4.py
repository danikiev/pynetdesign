r"""
04.1 Magnitude sensitivity for homogeneous model: test case 4
=============================================================
This is an example of computing magnitude sensitivity for a homogeneous model. 

Test case 4: 

* Geometry: combination of station geometry and two DAS geometries (2 vertical boreholes)
* Velocity model: homogeneous, Vp = 4300 m/s, Vp/Vs = 1.73, Qp = 100, Qs = 50, rho = 2300 kg/m^3
* Processing: magnitude sensitivity computation for individual geometries and for combined geometry (for grid slices only).
"""

###############################################################################
# Load all necessary packages
# ---------------------------

import pynetdesign.visualization as pndvis
import pynetdesign.modelling as pndmod
import numpy as np
from time import time

# sphinx_gallery_thumbnail_number = -2

#%%

###############################################################################
# Input
# -----

###############################################################################
# Read geometry
# ^^^^^^^^^^^^^

station_geometry_file_path = '../data/homo/test_04/TMPL_net_geometryA.txt'
das_geometry_file_path_1 = '../data/homo/test_04/TMPL_net_geometryB.txt'
das_geometry_file_path_2 = '../data/homo/test_04/TMPL_net_geometryC.txt'

station_geometry_df = pndmod.io.read_geometry(station_geometry_file_path)
das_geometry_df_1 = pndmod.io.read_geometry(das_geometry_file_path_1)
das_geometry_df_2 = pndmod.io.read_geometry(das_geometry_file_path_2)

### Show
print("\nStation geometry:")
print(station_geometry_df.head())
print(station_geometry_df.attrs)
print("\nDAS 1 geometry:")
print(das_geometry_df_1.head())
print(das_geometry_df_1.attrs) 
print("\nDAS 2 geometry:")
print(das_geometry_df_2.head())
print(das_geometry_df_2.attrs) 


###############################################################################
# Combine geometry
# ^^^^^^^^^^^^^^^^
combined_geometry_df = pndmod.io.combine_geometry(station_geometry_df, 
                                                  das_geometry_df_1, 
                                                  das_geometry_df_2, 
                                                  names=['Stations', 
                                                         'DAS_1', 
                                                         'DAS_2'])

### Show combined
print("\nCombined geometry:")
print(combined_geometry_df)
print(combined_geometry_df.attrs)
print("\nGeometry names:")
print(combined_geometry_df.get_names())
print("\nCombined status:",
      pndmod.io.is_combined_geometry(combined_geometry_df))

### Show recovered
print("\nStation geometry (recovered):")
print(combined_geometry_df.recover('Stations').head())
print("\nCombined status:",
      pndmod.io.is_combined_geometry(combined_geometry_df.recover('Stations')))

print("\nDAS 1 geometry (recovered):")
print(combined_geometry_df.recover('DAS_1').head())
print("\nCombined status:",
      pndmod.io.is_combined_geometry(combined_geometry_df.recover('DAS_1')))
print("\nAttributes of DAS 1 (recovered):")
print(combined_geometry_df.recover('DAS_1').attrs) 

print("\nDAS 2 geometry (recovered):")
print(combined_geometry_df.recover('DAS_2').head())
print("\nCombined status:",
      pndmod.io.is_combined_geometry(combined_geometry_df.recover('DAS_2')))
print("\nAttributes of DAS 2 (recovered):")
print(combined_geometry_df.recover('DAS_2').attrs) 
  

###############################################################################
# Read velocity model
# ^^^^^^^^^^^^^^^^^^^

velocity_file_path = '../data/homo/test_04/TMPL_velocity_model.txt'
velocity_df = pndmod.io.read_velocity(velocity_file_path)

print(velocity_df)
print(velocity_df.attrs)


###############################################################################
# Set imaging grid parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Imaging grid parameters
gx = np.arange(-3000, 8000 + 250, 250)
gy = np.arange(-3000, 8000 + 250, 250)
gz = np.arange(250, 5000 + 250, 250)

# Generate grid points
grid_points = pndmod.utils.generate_grid(x=gx,y=gy,z=gz)
print("Number of grid points:",grid_points.shape[0])
print("grid_points.shape:",grid_points.shape)

###############################################################################
# Plot geometry and imaging grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.geometry.plot_stations_view(df=combined_geometry_df, 
                                       grid_points=grid_points, 
                                       plot_names=False,
                                       grid_marker_size = 1,
                                       station_marker_size=100)
_ = pndvis.geometry.plot_stations_view(df=combined_geometry_df, 
                                       grid_points=grid_points, 
                                       plot_names=False,
                                       grid_marker_size = 1, 
                                       station_marker_size=1,
                                       view = 'xz')
_ = pndvis.geometry.plot_stations_view(df=combined_geometry_df, 
                                       grid_points=grid_points, 
                                       plot_names=False,
                                       grid_marker_size = 1, 
                                       station_marker_size=1,
                                       view = 'yz')

#%%

###############################################################################
# Prepare to compute magnitude sensitivity
# ----------------------------------------

# Get medium parameters
density = velocity_df.at[0,'Rho']
v_p = velocity_df.at[0,'Vp']
Q_p = velocity_df.at[0,'Qp']
v_s = velocity_df.at[0,'Vp']/velocity_df.at[0,'VpVsRatio']
Q_s = velocity_df.at[0,'Qs']
print("density, v_p, Q_p, v_s, Q_s:",density,v_p,Q_p,v_s,Q_s)

# Specify source paramerers
frequency = 10 # Hz

# Minimum S/N for detection on individual channel
min_SNR_p = 2
min_SNR_s = 2

# Minimum stations on which an event must be detected
min_stations_p = 3
min_stations_s = 3

# Save parameters
params = pndmod.classes.DetectionParameters(
    f_p=frequency,
    f_s=frequency,
    min_SNR_p=min_SNR_p,
    min_SNR_s=min_SNR_s,
    min_stations_p=min_stations_p,
    min_stations_s=min_stations_s,  
)

# Define slices to use
slice_X = 250
slice_Y = 250
slice_Z = 2000

# Get slice planes
X_plane_points, X_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=0,slice_coordinate=slice_X)
Y_plane_points, Y_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=1,slice_coordinate=slice_Y)
Z_plane_points, Z_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=2,slice_coordinate=slice_Z)

#%%

###############################################################################
# Compute and plot magnitude sensitivity for independent geometries
# -----------------------------------------------------------------
print("Computing and plotting magnitude sensitivity for independent geometries...")

# Loop over geometries
for geometry in combined_geometry_df.get_names():   
    print(f"Computing and plotting magnitude sensitivity for {geometry} geometry...")
    start_time = time()
    for wave_mode in ['P','S','PS']:
        X_slice_mag_sens = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                                geometry_df=combined_geometry_df.recover(geometry),
                                                                velocity_df=velocity_df,
                                                                params=params,
                                                                wave_mode=wave_mode) 
        Y_slice_mag_sens = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                                geometry_df=combined_geometry_df.recover(geometry),
                                                                velocity_df=velocity_df,
                                                                params=params,
                                                                wave_mode=wave_mode) 
        Z_slice_mag_sens = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
                                                                geometry_df=combined_geometry_df.recover(geometry),
                                                                velocity_df=velocity_df,
                                                                params=params,
                                                                wave_mode=wave_mode)     
        
        _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=X_slice_mag_sens, yi=gy, zi=gz, 
                                                      plot_title=f"Magnitude sensitivity for {geometry} geometry at X={slice_X} m: YZ view, {wave_mode}",
                                                      clb_title="$M_w$",                                                             
                                                      geometry_df=combined_geometry_df.recover(geometry),
                                                      plot_names = True if 'Name' in combined_geometry_df.recover(geometry).columns else False,
                                                      station_marker_size = 100 if 'gauge_length' not in combined_geometry_df.recover(geometry).attrs else 10)
            
        _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Y_slice_mag_sens, xi=gx, zi=gz, 
                                                      plot_title=f"Magnitude sensitivity for {geometry} geometry at Y= {slice_Y} m: XZ view, {wave_mode}",
                                                      clb_title="$M_w$",                                                             
                                                      geometry_df=combined_geometry_df.recover(geometry),
                                                      plot_names = True if 'Name' in combined_geometry_df.recover(geometry).columns else False,
                                                      station_marker_size = 100 if 'gauge_length' not in combined_geometry_df.recover(geometry).attrs else 10)
            
        _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Z_slice_mag_sens, xi=gx, yi=gy,
                                                      plot_title=f"Magnitude sensitivity for {geometry} geometry at depth {slice_Z} m: Map view, {wave_mode}",
                                                      clb_title="$M_w$",                                                             
                                                      geometry_df=combined_geometry_df.recover(geometry),
                                                      plot_names = True if 'Name' in combined_geometry_df.recover(geometry).columns else False,
                                                      station_marker_size = 100 if 'gauge_length' not in combined_geometry_df.recover(geometry).attrs else 10)
    end_time = time()
    print(f"Computation time: {end_time - start_time} seconds")
    

#%%

###############################################################################
# Compute and plot magnitude sensitivity for combined geometry
# ------------------------------------------------------------

print("Computing and plotting magnitude sensitivity for combined geometry...")

start_time = time()
for wave_mode in ['P','S','PS']:
    X_slice_mag_sens_comb = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                                 geometry_df=combined_geometry_df,
                                                                 velocity_df=velocity_df,
                                                                 params=params,
                                                                 wave_mode=wave_mode) 
    Y_slice_mag_sens_comb = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                                 geometry_df=combined_geometry_df,
                                                                 velocity_df=velocity_df,
                                                                 params=params,
                                                                 wave_mode=wave_mode) 
    Z_slice_mag_sens_comb = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
                                                                 geometry_df=combined_geometry_df,
                                                                 velocity_df=velocity_df,
                                                                 params=params,
                                                                 wave_mode=wave_mode) 
    
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=X_slice_mag_sens_comb, yi=gy, zi=gz, 
                                                  plot_title=f"Magnitude sensitivity for combined geometry at X={slice_X} m: YZ view, {wave_mode}",
                                                  clb_title="$M_w$",                                                             
                                                  geometry_df=combined_geometry_df.recover(geometry),
                                                  plot_names = True if 'Name' in combined_geometry_df.recover(geometry).columns else False,
                                                  station_marker_size = 100 if 'gauge_length' not in combined_geometry_df.recover(geometry).attrs else 10)
        
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Y_slice_mag_sens_comb, xi=gx, zi=gz, 
                                                  plot_title=f"Magnitude sensitivity for combined geometry at Y= {slice_Y} m: XZ view, {wave_mode}",
                                                  clb_title="$M_w$",                                                             
                                                  geometry_df=combined_geometry_df.recover(geometry),
                                                  plot_names = True if 'Name' in combined_geometry_df.recover(geometry).columns else False,
                                                  station_marker_size = 100 if 'gauge_length' not in combined_geometry_df.recover(geometry).attrs else 10)
        
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Z_slice_mag_sens_comb, xi=gx, yi=gy,
                                                  plot_title=f"Magnitude sensitivity for combined geometry at depth {slice_Z} m: Map view, {wave_mode}",
                                                  clb_title="$M_w$",                                                             
                                                  geometry_df=combined_geometry_df.recover(geometry),
                                                  plot_names = True if 'Name' in combined_geometry_df.recover(geometry).columns else False,
                                                  station_marker_size = 100 if 'gauge_length' not in combined_geometry_df.recover(geometry).attrs else 10)
    
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")    