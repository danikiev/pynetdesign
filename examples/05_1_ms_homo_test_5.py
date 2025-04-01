r"""
05.1 Magnitude sensitivity for homogeneous model: test case 5
=============================================================
This is an example of computing magnitude sensitivity for a homogeneous model.

Test case 5:

* Geometry: DAS cable in an inclined borehole
* Velocity model: homogeneous, Vp = 4300 m/s, Vp/Vs = 1.73, Qp = 100, Qs = 50, rho = 2300 kg/m^3
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

geometry_file_path = '../data/homo/test_05/TMPL_net_geometry_Inclined.txt'
geometry_df = pndmod.io.read_geometry(geometry_file_path)

print(geometry_df)
print(geometry_df.attrs) 

###############################################################################
# Read velocity model
# ^^^^^^^^^^^^^^^^^^^

velocity_file_path = '../data/homo/test_05/TMPL_velocity_model.txt'
velocity_df = pndmod.io.read_velocity(velocity_file_path)

print(velocity_df)
print(velocity_df.attrs)

###############################################################################
# Set imaging grid parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Imaging grid parameters
gx = np.arange(-7000, 7000 + 250, 250)
gy = np.arange(-7000, 7000 + 250, 250)
gz = np.arange(250, 5000 + 250, 250)

# Generate grid points
grid_points = pndmod.utils.generate_grid(x=gx,y=gy,z=gz)
print("Number of grid points:",grid_points.size)
print("Number of grid points in x:",gx.size)
print("Number of grid points in y:",gy.size)
print("Number of grid points in z:",gz.size)
print("grid_points.shape:",grid_points.shape)

###############################################################################
# Plot geometry and imaging grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.geometry.plot_stations_view(geometry_df,
                                       view='xy',                                                  
                                       grid_points=grid_points, 
                                       grid_marker_size=1,
                                       plot_names=False,
                                       station_marker_size=10)
_ = pndvis.geometry.plot_stations_view(geometry_df,
                                       view='xz',
                                       grid_points=grid_points, 
                                       grid_marker_size=1,
                                       plot_names=False,
                                       station_marker_size=2)
_ = pndvis.geometry.plot_stations_view(geometry_df,
                                       view='yz',
                                       grid_points=grid_points, 
                                       grid_marker_size=1,
                                       plot_names=False,
                                       station_marker_size=2)

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
slice_X = 0
slice_Y = 0
slice_Z = 2500

# Get slice planes
X_plane_points, X_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=0,slice_coordinate=slice_X)
Y_plane_points, Y_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=1,slice_coordinate=slice_Y)
Z_plane_points, Z_plane_indices = pndmod.utils.get_slice_coordinates(grid_points=grid_points,slice_dimension=2,slice_coordinate=slice_Z)

#%%

###############################################################################
# Compute magnitude sensitivity
# -----------------------------

print("Computing magnitude sensitivity...")

   
start_time = time()
X_slice_mag_sens_p = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                          geometry_df=geometry_df,
                                                          velocity_df=velocity_df,
                                                          params=params,
                                                          wave_mode='P') 
X_slice_mag_sens_s = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                          geometry_df=geometry_df,
                                                          velocity_df=velocity_df,
                                                          params=params,
                                                          wave_mode='S') 
X_slice_mag_sens_ps = pndmod.core.get_mag_sensitivity(grid_coords=X_plane_points, 
                                                           geometry_df=geometry_df,
                                                           velocity_df=velocity_df,
                                                           params=params,
                                                           wave_mode='PS') 
Y_slice_mag_sens_p = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                          geometry_df=geometry_df,
                                                          velocity_df=velocity_df,
                                                          params=params,
                                                          wave_mode='P') 
Y_slice_mag_sens_s = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                          geometry_df=geometry_df,
                                                          velocity_df=velocity_df,
                                                          params=params,
                                                          wave_mode='S') 
Y_slice_mag_sens_ps = pndmod.core.get_mag_sensitivity(grid_coords=Y_plane_points, 
                                                           geometry_df=geometry_df,
                                                           velocity_df=velocity_df,
                                                           params=params,
                                                           wave_mode='PS') 
Z_slice_mag_sens_p = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
                                                          geometry_df=geometry_df,
                                                          velocity_df=velocity_df,
                                                          params=params,
                                                          wave_mode='P') 
Z_slice_mag_sens_s = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
                                                          geometry_df=geometry_df,
                                                          velocity_df=velocity_df,
                                                          params=params,
                                                          wave_mode='S') 
Z_slice_mag_sens_ps = pndmod.core.get_mag_sensitivity(grid_coords=Z_plane_points, 
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

station_marker_size = 25
clb_range = None
plt_levels = None

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= X_slice_mag_sens_p, yi=gy, zi=gz, 
                                              plot_title=f"Magnitude sensitivity at X={slice_X} m: YZ view, P-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)
   
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= Y_slice_mag_sens_p, xi=gx, zi=gz, 
                                              plot_title=f"Magnitude sensitivity at Y={slice_Y} m: XZ view, P-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= Z_slice_mag_sens_p, xi=gx, yi=gy, 
                                              plot_title=f"Magnitude sensitivity at Z={slice_Z} m: XY view, P-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)



###############################################################################
# Plot magnitude sensitivity for S wave
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= X_slice_mag_sens_s, yi=gy, zi=gz, 
                                              plot_title=f"Magnitude sensitivity at X={slice_X} m: YZ view, S-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)
   
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= Y_slice_mag_sens_s, xi=gx, zi=gz, 
                                              plot_title=f"Magnitude sensitivity at Y={slice_Y} m: XZ view, S-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= Z_slice_mag_sens_s, xi=gx, yi=gy, 
                                              plot_title=f"Magnitude sensitivity at Z={slice_Z} m: XY view, S-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)

###############################################################################
# Plot magnitude sensitivity for P and S waves
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= X_slice_mag_sens_ps, yi=gy, zi=gz, 
                                              plot_title=f"Magnitude sensitivity at X={slice_X} m: YZ view, P- and S-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)
   
_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= Y_slice_mag_sens_ps, xi=gx, zi=gz, 
                                              plot_title=f"Magnitude sensitivity at Y={slice_Y} m: XZ view, P- and S-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)

_ = pndvis.sensitivity.plot_sensitivity_slice(sens_data= Z_slice_mag_sens_ps, xi=gx, yi=gy, 
                                              plot_title=f"Magnitude sensitivity at Z={slice_Z} m: XY view, P- and S-waves",
                                              clb_title="$M_w$",                                                             
                                              geometry_df=geometry_df,
                                              plot_names = False,
                                              station_marker_size = station_marker_size,
                                              clb_range=clb_range,
                                              plt_levels=plt_levels)
