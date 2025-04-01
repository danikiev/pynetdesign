r"""
01.1 Magnitude sensitivity for homogeneous model: test case 1
=============================================================
This is an example of computing magnitude sensitivity for a homogeneous model.
 
Test case 1 (from :cite:t:`Hallo2012`):

* Geometry: simple station geometry with 3 surface stations in the form of a triangle
* Velocity model: homogeneous, Vp = 4300 m/s, Vp/Vs = 1.73, Qp = 100, Qs = 50, rho = 2300 kg/m^3
* Processing: magnitude sensitivity computation for individual stations (for grid slices only) and for all stations (for all grid points, cutting slices after).
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

geometry_file_path = '../data/homo/test_01/TMPL_net_geometryA.txt'
geometry_df = pndmod.io.read_geometry(geometry_file_path)

print(geometry_df)
print(geometry_df.attrs)

###############################################################################
# Read velocity model
# ^^^^^^^^^^^^^^^^^^^

velocity_file_path = '../data/homo/test_01/TMPL_velocity_model.txt'
velocity_df = pndmod.io.read_velocity(velocity_file_path)

print(velocity_df)
print(velocity_df.attrs)

###############################################################################
# Set imaging grid parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Imaging grid parameters
gx = np.arange(-3000, 8000 + 250, 250)
gy = np.arange(-3000, 8000 + 250, 250)
gz = np.arange(250, 10000 + 250, 250)
#gz = np.arange(250, 5000 + 250, 250)

# Generate grid points
grid_points = pndmod.utils.generate_grid(x=gx,y=gy,z=gz)
print("Number of grid points:",grid_points.shape[0])
print("grid_points.shape:",grid_points.shape)

###############################################################################
# Plot geometry and imaging grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_ = pndvis.geometry.plot_stations_view(df=geometry_df, 
                                       grid_points=grid_points,
                                       grid_marker_size = 1)
_ = pndvis.geometry.plot_stations_view(df=geometry_df, 
                                       grid_points=grid_points, 
                                       grid_marker_size = 1, 
                                       view = 'xz')
_ = pndvis.geometry.plot_stations_view(df=geometry_df, 
                                       grid_points=grid_points, 
                                       grid_marker_size = 1, 
                                       view = 'yz')

#%%

###############################################################################
# Prepare to compute magnitude sensitivity
# ----------------------------------------

# Specify source parameters
#frequency = 10 # Hz
frequency = None # Automatic peak frequency computation

# Minimum S/N for detection on individual channel
min_SNR_p = 2
min_SNR_s = 2

# Minimum stations on which an event must be detected
min_stations_p = 3
min_stations_s = 3

# Save parameters
params = pndmod.classes.DetectionParameters(
    min_SNR_p=min_SNR_p,
    min_SNR_s=min_SNR_s,
    min_stations_p=min_stations_p,
    min_stations_s=min_stations_s,  
    f_p=frequency,
    f_s=frequency,
)
params.validate()

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
# Compute and plot magnitude sensitivity for single stations
# ----------------------------------------------------------

# Get minumum detectable amplitudes
min_amps_p, min_amps_s = pndmod.utils.retrieve_min_amps(geometry_df=geometry_df,                                                        
                                                        SNR_p=min_SNR_p,
                                                        SNR_s=min_SNR_s,
                                                        f_p=frequency,
                                                        f_s=frequency)

amps_type = geometry_df.attrs.get('noise_type')

# Loop over stations
for ist in range(len(geometry_df)):    
    station_coords = geometry_df[['X', 'Y', 'Z']].to_numpy()[ist]
    print("station_coords:",station_coords)        
    print("min_amps_p:",min_amps_p[ist])
    print("min_amps_s:",min_amps_s[ist])
        
    
    X_slice_mag_sens_p = pndmod.homo.mag_sensitivity_grid(grid_coords=X_plane_points, 
                                                          velocity_df=velocity_df,
                                                          station_coords=station_coords,                                                           
                                                          min_amps_p=min_amps_p[ist], 
                                                          min_stations_p=1,
                                                          f_p=frequency,                                                          
                                                          amps_type=amps_type,
                                                          wave_mode='P')
    
    Y_slice_mag_sens_p = pndmod.homo.mag_sensitivity_grid(grid_coords=Y_plane_points, 
                                                          velocity_df=velocity_df,
                                                          station_coords=station_coords,                                                           
                                                          min_amps_p=min_amps_p[ist], 
                                                          min_stations_p=1,
                                                          f_p=frequency,
                                                          amps_type=amps_type,
                                                          wave_mode='P')
    
    Z_slice_mag_sens_p = pndmod.homo.mag_sensitivity_grid(grid_coords=Z_plane_points, 
                                                          velocity_df=velocity_df,
                                                          station_coords=station_coords,                                                           
                                                          min_amps_p=min_amps_p[ist], 
                                                          min_stations_p=1,
                                                          f_p=frequency,
                                                          amps_type=amps_type,
                                                          wave_mode='P')
    
    X_slice_mag_sens_s = pndmod.homo.mag_sensitivity_grid(grid_coords=X_plane_points, 
                                                          velocity_df=velocity_df,
                                                          station_coords=station_coords,                                                           
                                                          min_amps_s=min_amps_s[ist], 
                                                          min_stations_s=1,
                                                          f_s=frequency,
                                                          amps_type=amps_type,
                                                          wave_mode='S')
    
    Y_slice_mag_sens_s = pndmod.homo.mag_sensitivity_grid(grid_coords=Y_plane_points, 
                                                          velocity_df=velocity_df,
                                                          station_coords=station_coords,                                                           
                                                          min_amps_s=min_amps_s[ist], 
                                                          min_stations_s=1,
                                                          f_s=frequency,
                                                          amps_type=amps_type,
                                                          wave_mode='S')
    
    Z_slice_mag_sens_s = pndmod.homo.mag_sensitivity_grid(grid_coords=Z_plane_points, 
                                                          velocity_df=velocity_df,
                                                          station_coords=station_coords,                                                           
                                                          min_amps_s=min_amps_s[ist], 
                                                          min_stations_s=1,
                                                          f_s=frequency,
                                                          amps_type=amps_type,
                                                          wave_mode='S')
        
    print("Plot magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] +" and P wave")
    
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=X_slice_mag_sens_p, yi=gy, zi=gz, 
                                                  plot_title="Magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] + " at X=" + str(slice_X) + " m: YZ view, P-wave",
                                                  clb_title="$M_w$")
    
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Y_slice_mag_sens_p, xi=gx, zi=gz, 
                                                  plot_title="Magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] + " at Y=" + str(slice_Y) + " m: XZ view, P-wave",
                                                  clb_title="$M_w$")
    
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Z_slice_mag_sens_p, xi=gx, yi=gy, geometry_df=geometry_df,
                                                  plot_title="Magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] + " at depth " + str(slice_Z) + " m: Map view, P-wave",
                                                  clb_title="$M_w$")
    
    print("Plot magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] +" and S wave")
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=X_slice_mag_sens_s, yi=gy, zi=gz, 
                                                  plot_title="Magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] + " at X=" + str(slice_X) + " m: YZ view, S-wave",
                                                  clb_title="$M_w$")
    
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Y_slice_mag_sens_s, xi=gx, zi=gz, 
                                                  plot_title="Magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] + " at Y=" + str(slice_Y) + " m: XZ view, S-wave",
                                                  clb_title="$M_w$")
    
    _ = pndvis.sensitivity.plot_sensitivity_slice(sens_data=Z_slice_mag_sens_s, xi=gx, yi=gy, geometry_df=geometry_df,
                                                  plot_title="Magnitude sensitivity for station " + geometry_df['Name'].tolist()[ist] + " at depth " + str(slice_Z) + " m: Map view, S-wave",
                                                  clb_title="$M_w$")
    
#%%

###############################################################################
# Compute magnitude sensitivity for several stations using the whole grid
# -----------------------------------------------------------------------

# Compute magnitude sensitivity for the whole grid for P 
print("Computing magnitude sensitivity for the whole grid for P wave...")
start_time = time()
mag_sens_grid_p = pndmod.core.get_mag_sensitivity(grid_coords=grid_points, 
                                                  geometry_df=geometry_df,
                                                  velocity_df=velocity_df,
                                                  params=params,
                                                  wave_mode='P') 
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

# Compute magnitude sensitivity for the whole grid for S 
print("Computing magnitude sensitivity for the whole grid for S wave...")
start_time = time()
mag_sens_grid_s = pndmod.core.get_mag_sensitivity(grid_coords=grid_points, 
                                                  geometry_df=geometry_df,
                                                  velocity_df=velocity_df,
                                                  params=params,
                                                  wave_mode='S') 
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")


# Compute magnitude sensitivity for the whole grid for P and S waves
print("Computing magnitude sensitivity for the whole grid for both P and S waves...")
start_time = time()
mag_sens_grid_ps = pndmod.core.get_mag_sensitivity(grid_coords=grid_points, 
                                                   geometry_df=geometry_df,
                                                   velocity_df=velocity_df,
                                                   params=params,
                                                   wave_mode='PS') 
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Get sensitivity slices by cutting from the grid
# -----------------------------------------------

X_slice_mag_sens_stations_p = mag_sens_grid_p[X_plane_indices]
Y_slice_mag_sens_stations_p = mag_sens_grid_p[Y_plane_indices]
Z_slice_mag_sens_stations_p = mag_sens_grid_p[Z_plane_indices]

X_slice_mag_sens_stations_s = mag_sens_grid_s[X_plane_indices]
Y_slice_mag_sens_stations_s = mag_sens_grid_s[Y_plane_indices]
Z_slice_mag_sens_stations_s = mag_sens_grid_s[Z_plane_indices]

X_slice_mag_sens_stations_ps = mag_sens_grid_ps[X_plane_indices]
Y_slice_mag_sens_stations_ps = mag_sens_grid_ps[Y_plane_indices]
Z_slice_mag_sens_stations_ps = mag_sens_grid_ps[Z_plane_indices]

#%%

###############################################################################
# Plot magnitude sensitivity for several stations
# -----------------------------------------------

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