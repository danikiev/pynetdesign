r"""
02.1 Magnitude sensitivity for homogeneous model: test case 2
=============================================================
This is an example of computing magnitude sensitivity for a homogeneous model.
 
Test case 2: 

* Geometry: simple station geometry with 4 surface stations in the form of a triangle with 1 station in the center
* Velocity model: homogeneous, Vp = 3100 m/s, Vp/Vs = 1.73, Qp = 70, Qs = 50, rho = 2400 kg/m^3
* Processing: magnitude sensitivity computation for all stations (for grid slices only).
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

geometry_file_path = '../data/homo/test_02/TMPL_net_geometryA.txt'
geometry_df = pndmod.io.read_geometry(geometry_file_path)

print(geometry_df)
print(geometry_df.attrs)

###############################################################################
# Read velocity model
# ^^^^^^^^^^^^^^^^^^^

velocity_file_path = '../data/homo/test_02/TMPL_velocity_model.txt'
velocity_df = pndmod.io.read_velocity(velocity_file_path)

print(velocity_df)
print(velocity_df.attrs)

###############################################################################
# Set imaging grid parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Imaging grid parameters
gx = np.arange(-4500, 4500 + 250, 250)
gy = np.arange(-3750, 4750 + 250, 250)
gz = np.arange(1500, 5000 + 250, 250)

# Generate grid points
grid_points = pndmod.utils.generate_grid(x=gx,y=gy,z=gz)
print("Number of grid points:",grid_points.shape[0])
print("grid_points.shape:",grid_points.shape)

###############################################################################
# Plot geometry and imaging grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.geometry.plot_stations_view(df=geometry_df, grid_points=grid_points, grid_marker_size = 1)
_ = pndvis.geometry.plot_stations_view(df=geometry_df, grid_points=grid_points, grid_marker_size = 1, view = 'xz')
_ = pndvis.geometry.plot_stations_view(df=geometry_df, grid_points=grid_points, grid_marker_size = 1, view = 'yz')

#%%

###############################################################################
# Prepare to compute magnitude sensitivity
# ----------------------------------------

# Specify source parameters
frequency = 10 # Hz

# Minimum S/N for detection on individual channel
min_SNR_p = 2
min_SNR_s = 2

# Minimum stations on which an event must be detected
min_stations_p = 3
min_stations_s = 3

# Use free surface correction
#free_surface = False
free_surface = True

# Save parameters
params = pndmod.classes.DetectionParameters(
    f_p=frequency,
    f_s=frequency,
    min_SNR_p=min_SNR_p,
    min_SNR_s=min_SNR_s,
    min_stations_p=min_stations_p,
    min_stations_s=min_stations_s,  
    free_surface=free_surface 
)

# Define slices to use
slice_X = 0
slice_Y = 0
slice_Z = 2000

# Get slice planes
X_plane_points, X_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=0,slice_coordinate=slice_X)
Y_plane_points, Y_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=1,slice_coordinate=slice_Y)
Z_plane_points, Z_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=2,slice_coordinate=slice_Z)

#%%

###############################################################################
# Compute magnitude sensitivity
# -----------------------------

# Compute magnitude sensitivity for P wave
print("Computing magnitude sensitivity for P wave...")
start_time = time()
X_slice_mag_sens_stations_p = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                              geometry_df=geometry_df,
                                                              velocity_df=velocity_df,
                                                              params=params,
                                                                   wave_mode='P')
Y_slice_mag_sens_stations_p = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                              geometry_df=geometry_df,
                                                              velocity_df=velocity_df,
                                                              params=params,
                                                              wave_mode='P')
Z_slice_mag_sens_stations_p = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
                                                              geometry_df=geometry_df,
                                                              velocity_df=velocity_df,
                                                              params=params,
                                                              wave_mode='P')
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

# Compute magnitude sensitivity for S wave
print("Computing magnitude sensitivity for S wave...")
start_time = time()
X_slice_mag_sens_stations_s = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                              geometry_df=geometry_df,
                                                              velocity_df=velocity_df,
                                                              params=params,
                                                              wave_mode='S')
Y_slice_mag_sens_stations_s = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                              geometry_df=geometry_df,
                                                              velocity_df=velocity_df,
                                                              params=params,
                                                              wave_mode='S')
Z_slice_mag_sens_stations_s = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
                                                              geometry_df=geometry_df,
                                                              velocity_df=velocity_df,
                                                              params=params,
                                                              wave_mode='S')
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")


# Compute magnitude sensitivity for the whole grid for P and S waves
print("Computing magnitude sensitivity for both P and S waves...")
start_time = time()
X_slice_mag_sens_stations_ps = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                               geometry_df=geometry_df,
                                                               velocity_df=velocity_df,
                                                               params=params,
                                                               wave_mode='PS')
Y_slice_mag_sens_stations_ps = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                               geometry_df=geometry_df,
                                                               velocity_df=velocity_df,
                                                               params=params,
                                                               wave_mode='PS')
Z_slice_mag_sens_stations_ps = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
                                                               geometry_df=geometry_df,
                                                               velocity_df=velocity_df,
                                                               params=params,
                                                               wave_mode='PS')

end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

#%%

###############################################################################
# Plot magnitude sensitivity
# --------------------------

###############################################################################
# Plot magnitude sensitivity for P wave
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

clb_range = None
plt_levels = None

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=X_slice_mag_sens_stations_p, yi=gy, zi=gz, 
                                                     plot_title="Magnitude sensitivity at X=" + str(slice_X) + " m: YZ view, P-wave",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Y_slice_mag_sens_stations_p, xi=gx, zi=gz, 
                                                     plot_title="Magnitude sensitivity at Y=" + str(slice_Y) + " m: XZ view, P-wave",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)
    
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Z_slice_mag_sens_stations_p, xi=gx, yi=gy, geometry_df=geometry_df,
                                                     plot_title="Magnitude sensitivity at depth " + str(slice_Z) + " m: Map view, P-wave",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)

###############################################################################
# Plot magnitude sensitivity for S wave
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=X_slice_mag_sens_stations_s, yi=gy, zi=gz, 
                                                     plot_title="Magnitude sensitivity at X=" + str(slice_X) + " m: YZ view, S-wave",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)
    
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Y_slice_mag_sens_stations_s, xi=gx, zi=gz, 
                                                     plot_title="Magnitude sensitivity at Y=" + str(slice_Y) + " m: XZ view, S-wave",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)
    
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Z_slice_mag_sens_stations_s, xi=gx, yi=gy, geometry_df=geometry_df,
                                                     plot_title="Magnitude sensitivity at depth " + str(slice_Z) + " m: Map view, S-wave",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)

###############################################################################
# Plot magnitude sensitivity for P and S waves
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=X_slice_mag_sens_stations_ps, yi=gy, zi=gz,
                                                     plot_title="Magnitude sensitivity at X=" + str(slice_X) + " m: YZ view, P- and S-waves",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)
    
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Y_slice_mag_sens_stations_ps, xi=gx, zi=gz,
                                                     plot_title="Magnitude sensitivity at Y=" + str(slice_Y) + " m: XZ view, P- and S-waves",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)
    
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Z_slice_mag_sens_stations_ps, xi=gx, yi=gy, geometry_df=geometry_df,
                                                     plot_title="Magnitude sensitivity at depth " + str(slice_Z) + " m: Map view, P- and S-waves",
                                                     clb_title="$M_w$",
                                                     clb_range=clb_range,
                                                     plt_levels=plt_levels)