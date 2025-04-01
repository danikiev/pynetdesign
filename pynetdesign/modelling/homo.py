from typing import Union, Optional
from pynetdesign.modelling.classes import *
from pynetdesign.modelling.utils import *
from pynetdesign.modelling.io import *

import numpy as np
import pandas as pd
import inspect

def mag_sensitivity_grid(grid_coords: np.ndarray,
                         velocity_df: pd.DataFrame,
                         geometry_df: pd.DataFrame = None,
                         station_coords: np.ndarray = None,
                         min_amps_p: np.ndarray = None,
                         min_amps_s: np.ndarray = None,
                         min_stations_p: int = None,
                         min_stations_s: int = None,
                         rad_pattern_p: float = 0.52,
                         rad_pattern_s: float = 0.63,
                         f_p: float = None,
                         f_s: float = None,
                         f_p_corner: float = None,
                         f_s_corner: float = None,
                         wave_mode: str = 'PS',
                         use_free_surface: bool = False,
                         use_station_directionality: bool = False,
                         return_stations: bool = False,
                         strict_nan_check: bool = False,
                         amps_type: str = None):

    r"""
    Computes magnitude sensitivity - the minimum moment magnitude (Mw) of a microseismic event
    which will theoretically produce particle velocity amplitudes (of P and/or S waves)
    on stations necessary for event detection assuming 3D elastic homogeneous medium with attenuation.
    It takes into account minimum number of stations on which the event should be detected.
    By request it is possible to output only minimum moment magnitude for each station.

    Parameters
    ----------
    grid_coords : :obj:`numpy.ndarray`
        Array of grid coordinates [[xs1, ys1, zs1], [xs2, ys2, zs2], ...]
    velocity_df : :obj:`pandas.DataFrame`
        Velocity model DataFrame
    geometry_df : :obj:`pandas.DataFrame`, optional
        Geometry DataFrame, use if station_coords is None
    station_coords : :obj:`numpy.ndarray`, optional
        Array of station coordinates [[xr1, yr1, zr1], [xr2, yr2, zr2], ...], use if geometry_df is None
    min_amps_p : :obj:`numpy.ndarray`, optional
        Minimal measurable displacement amplitudes of on stations [a1, a2, ...] for P waves
    min_amps_s : :obj:`numpy.ndarray`, optional
        Minimal measurable displacement amplitudes on stations [a1, a2, ...] for S waves
    min_stations_p : :obj:`int`, optional, default: None (3)
        Minimum number of stations on which event must be detected with P waves
    min_stations_s : :obj:`int`, optional, default: None (3)
        Minimum number of stations on which event must be detected with S waves
    rad_pattern_p : :obj:`float`, optional, default: 0.52
        Radiation pattern factor for P waves
    rad_pattern_s : :obj:`float`, optional, default: 0.63
        Radiation pattern factor for S waves
    f_p : :obj:`float`, optional
        Frequency of the P wave (Hz)
    f_s : :obj:`float`, optional
        Frequency of the S wave (Hz)
    f_p_corner : :obj:`float`, optional
        Corner frequency of the P wave (Hz)
    f_s_corner : :obj:`float`, optional
        Corner frequency of the S wave (Hz)
    wave_mode: :obj:`str`, optional, default: 'PS'
        Wave mode to use, can be 'P','S' or 'PS'
    use_free_surface : :obj:`bool`, optional, default: False
        Use free surface correction.
        If True, this will consider amplification of the amplitudes when recording on the daily surface
        due to the free surface boundary condition.
    use_station_directionality : :obj:`bool`, optional, default: False
        Use station directionality correction.
        This will consider sensitivity of the measurement only in direction parallel to the adjacent stations
    return_stations : :obj:`bool`, optional, default: False
        Return magnitude sensitivity for each station, don't take into account detectability
    strict_nan_check: :obj:`bool`, optional, default: False
        flag to check if NaN values appear in the result
    amps_type : :obj:`str`, optional, default: None
        Type of amplitudes, can be 'displacement', 'velocity', 'acceleration', 'strain rate', 'strain'.

    Returns
    -------
    Mw_min_grid : :obj:`numpy.ndarray`
        Array of computed detectable minimum moment magnitudes on the grid, if return_stations = False
    Mw_min_stations : :obj:`numpy.ndarray`
        Array of computed minimum moment magnitudes for individual stations, if return_stations = True
        Mw_min_stations is of shape (2, npairs) where npairs is a number of of source-grid pairs
        determined from station_coords and grid_coords. First dimension corresponds to P and S waves.

    Raises
    ------
    ValueError :
        if both station_coords and geometry_df are not None or None
    ValueError :
        if there is inconsistency in P/S wave parameters
    ValueError :
        if number of minimal measurable amplitudes do not match number of stations
    ValueError :
        if density, wave velocity or attenuation factor is not positive
    ValueError :
        if radiation pattern factor for P or S wave is less or equal to 0 or higher than 1
    ValueError :
        if number of minimal measurable amplitudes does not match number of stations
    RuntimeError :
        if strict_nan_check is True and NaN appears in the result

    Notes
    -----
    This is a refactored vectorized version of the code
    """

    if (station_coords is not None and geometry_df is not None) or (station_coords is None and geometry_df is None):
        raise ValueError("You must provide either `station_coords` or `geometry_df`, but not both.")

    # Get station coordinates from geometry
    if station_coords is None:
        station_coords = geometry_df[['X', 'Y', 'Z']].to_numpy()

    # Handle the case if station_coords has only one point (3 elements)
    if station_coords.ndim == 1 and station_coords.size == 3:
        station_coords = station_coords[np.newaxis, :]  # reshape to (1, 3)

    # Handle the case if grid_coords has only one point (3 elements)
    if grid_coords.ndim == 1 and grid_coords.size == 3:
        grid_coords = grid_coords[np.newaxis, :]  # reshape to (1, 3)

    # Check if amplitudes type is provided
    if geometry_df is None:
        if amps_type is None:
            raise ValueError("You must provide amplitude type if geometry is not provided")
    else:
        amps_type = geometry_df.attrs.get('noise_type')

    # Check velocity model
    if len(velocity_df) != 1:
        raise ValueError("Velocity model is not homogeneous!")

    # Retrieve medium parameters
    density = velocity_df.at[0,'Rho']
    v_p = velocity_df.at[0,'Vp']
    Q_p = velocity_df.at[0,'Qp']
    v_s = velocity_df.at[0,'Vp']/velocity_df.at[0,'VpVsRatio']
    Q_s = velocity_df.at[0,'Qs']

    # Validate parameters
    validate_parameters(density=density,
                        v_p=v_p,                        
                        v_s=v_s,
                        Q_s=Q_s,
                        Q_p=Q_p,
                        min_amps_p=min_amps_p,
                        min_amps_s=min_amps_s,
                        rad_pattern_p=rad_pattern_p,
                        rad_pattern_s=rad_pattern_s,
                        wave_mode=wave_mode,
                        station_coords=station_coords)

    # Precompute distances between all grid points and all stations
    # Shape: (n_grid_points, n_stations)
    distances = np.linalg.norm(grid_coords[:, np.newaxis] - station_coords, axis=2)
    if strict_nan_check:
        if np.any(distances == 0):
            raise ValueError("Zero distance: source and receiver must have different coordinates")

    # Calculate free surface correction coefficient if needed
    if use_free_surface:
        if v_p is None or v_s is None:
            raise ValueError("Both P- and S-wave velocities must be declared for free surface correction!")
        # Simple approximation
        fs_coef_p = fs_coef_s = 2*np.ones_like(distances)
    else:
        # If not using free surface correction, use ones (no effect on calculations)
        fs_coef_p = fs_coef_s = np.ones_like(distances)

    # Calculate station directionality if needed
    if use_station_directionality:
        stations_dir_coef_p, stations_dir_coef_s = get_ray_station_directionality(station_coords=station_coords,
                                                                                  grid_coords=grid_coords,
                                                                                  strict_nan_check=strict_nan_check)
    else:
        # If not using directionality, use ones (no effect on calculations)
        stations_dir_coef_p = stations_dir_coef_s = np.ones_like(distances)

    # Helper function to calculate Mw_min for a given wave type
    def calculate_Mw_min(f, fcorner, density, v, Q, rad_pattern, min_amps, fs_coef, stations_dir_coef, amps_type):
        M0 = calculate_M0(density=density,
                          v=v,
                          Q=Q,
                          rad_pattern=rad_pattern,
                          r=distances,
                          amps = min_amps * stations_dir_coef / fs_coef,
                          f=f,
                          fcorner=fcorner,
                          amps_type=amps_type)
        return calculate_Mw(M0)

    # Preallocate array for Mw_min calculations
    # Shape: (2, n_grid_points, n_stations) where index 0 is for P waves, 1 for S waves
    Mw_min_stations = np.full((2, *distances.shape), np.nan)

    # Calculate Mw_min for P waves if applicable
    if wave_mode in ('P', 'PS'):
        Mw_min_stations[0] = calculate_Mw_min(f=f_p,
                                              fcorner=f_p_corner,
                                              density=density,
                                              v=v_p,
                                              Q=Q_p,
                                              rad_pattern=rad_pattern_p,
                                              min_amps=min_amps_p,
                                              fs_coef=fs_coef_p,
                                              stations_dir_coef=stations_dir_coef_p,
                                              amps_type=amps_type)

    # Calculate Mw_min for S waves if applicable
    if wave_mode in ('S', 'PS'):
        Mw_min_stations[1] = calculate_Mw_min(f=f_s,
                                              fcorner=f_s_corner,
                                              density=density,
                                              v=v_s,
                                              Q=Q_s,
                                              rad_pattern=rad_pattern_s,
                                              min_amps=min_amps_s,
                                              fs_coef=fs_coef_s,
                                              stations_dir_coef=stations_dir_coef_s,
                                              amps_type=amps_type)

    # Return depending on the request
    if return_stations:
        return Mw_min_stations
    else:
        Mw_min_grid = mag_detectable(Mw_min_stations=Mw_min_stations,
                                     min_stations_p=min_stations_p,
                                     min_stations_s=min_stations_s,
                                     wave_mode=wave_mode)
        # Check for NaNs
        if strict_nan_check:
            if np.any(np.isnan(Mw_min_grid)):
                current_function = inspect.currentframe().f_code.co_name
                raise RuntimeError(f"Error in function {current_function}: NaN values detected while computing the magnitude sensitivity")

        return Mw_min_grid

def loc_uncertainty_grid(grid_coords: np.ndarray,
                         velocity_df: pd.DataFrame,
                         loc_gx: np.ndarray,
                         loc_gy: np.ndarray,
                         loc_gz: np.ndarray,
                         source_mw: float,
                         geometry_df: pd.DataFrame = None,
                         station_coords: np.ndarray = None,
                         min_amps_p: np.ndarray = None,
                         min_amps_s: np.ndarray = None,
                         rad_pattern_p: float = 0.52,
                         rad_pattern_s: float = 0.63,
                         sigma_p: float = None,
                         sigma_s: float = None,
                         min_sigma: float = None,
                         wave_mode: str = 'PS',
                         use_free_surface: bool = False,
                         use_station_directionality: bool = False,
                         output_pdf: bool = False,
                         strict_nan_check: bool = False):

    r"""
    Computes location uncertainty of a microseismic event using P and/or S waves
    assuming 3D elastic homogeneous medium with attenuation.
    It takes into account picking uncertainties for P and/or S waves.

    Parameters
    ----------
    grid_coords : :obj:`numpy.ndarray`
        Array of imaging grid coordinates [[xs1, ys1, zs1], [xs2, ys2, zs2], ...]
    velocity_df : :obj:`pandas.DataFrame`
        Velocity model DataFrame    
    loc_gx : :obj:`numpy.ndarray`
        Vector of x-coordinates of the location grid
    loc_gy : :obj:`numpy.ndarray`
        Vector of y-coordinates of the location grid
    loc_gz : :obj:`numpy.ndarray`
        Vector of z-coordinates of the location grid
    source_mw : :obj:`float`
        Moment magnitude of the synthetic source event
    geometry_df : :obj:`pandas.DataFrame`, optional
        Geometry DataFrame, use if station_coords is None
    station_coords : :obj:`numpy.ndarray`, optional
        Array of station coordinates [[xr1, yr1, zr1], [xr2, yr2, zr2], ...], use if geometry_df is None
    min_amps_p : :obj:`numpy.ndarray`, optional
        Minimal measurable displacement amplitudes of on stations [a1, a2, ...] for P waves
    min_amps_s : :obj:`numpy.ndarray`, optional
        Minimal measurable displacement amplitudes on stations [a1, a2, ...] for S waves
    rad_pattern_p : :obj:`float`, optional, default: 0.52
        Radiation pattern factor for P waves
    rad_pattern_s : :obj:`float`, optional, default: 0.63
        Radiation pattern factor for S waves
    sigma_p : :obj:`float`, optional
        Picking uncertainty for P waves, in seconds.
        If ``None``, estimated from the peak frequency of the P wave.
    sigma_s : :obj:`float`, optional
        Picking uncertainty for S waves, in seconds.
        If ``None``, estimated from the peak frequency of the S wave.
    min_sigma : :obj:`float`, optional, default: 1e-3 (when None)
        Minimum picking uncertainty, in seconds.
    wave_mode: :obj:`str`, optional, default: 'PS'
        Wave mode to use, can be 'P','S' or 'PS'
    use_free_surface : :obj:`bool`, optional, default: False
        Use free surface correction.
        If True, this will consider amplification of the amplitudes when recording on the daily surface
        due to the free surface boundary condition.
    use_station_directionality : :obj:`bool`, optional, default: False
        Use station directionality correction.
        This will consider sensitivity of the measurement only in direction parallel to the adjacent stations
    output_pdf : :obj:`bool`, optional, default: False
        Return probability density function
    strict_nan_check: :obj:`bool`, optional, default: False
        flag to check if NaN values appear in the result

    Returns
    -------
    LU_grid : :obj:`numpy.ndarray`
        Array of computed location uncertainties (azimuth, sigma_a, sigma_b, sigma_z) on the grid
    pdf : :obj:`numpy.ndarray`
        3D probability density function of the event location, if ``output_pdf = True``

    Raises
    ------
    ValueError :
        if both station_coords and geometry_df are not None or None
    ValueError :
        if there is inconsistency in P/S wave parameters
    ValueError :
        if number of minimal measurable amplitudes do not match number of stations
    ValueError :
        if density, wave velocity or attenuation factor is not positive
    ValueError :
        if radiation pattern factor for P or S wave is less or equal to 0 or higher than 1
    ValueError :
        if number of minimal measurable amplitudes does not match number of stations
    RuntimeError :
        if strict_nan_check is True and NaN appears in the result

    Notes
    -----
    If ``station_coords`` is not provided, ``geometry_df`` must be provided and vice versa.
    If ``output_pdf`` is True, the function will return a tuple of two elements: LU_grid and pdf.
    """

    if (station_coords is not None and geometry_df is not None) or (station_coords is None and geometry_df is None):
        raise ValueError("You must provide either `station_coords` or `geometry_df`, but not both.")

    # Get station coordinates from geometry
    if station_coords is None:
        station_coords = geometry_df[['X', 'Y', 'Z']].to_numpy()

    # Handle the case if station_coords has only one point (3 elements)
    if station_coords.ndim == 1 and station_coords.size == 3:
        station_coords = station_coords[np.newaxis, :]  # reshape to (1, 3)

    # Handle the case if grid_coords has only one point (3 elements)
    if grid_coords.ndim == 1 and grid_coords.size == 3:
        grid_coords = grid_coords[np.newaxis, :]  # reshape to (1, 3)

    # Check velocity model
    if len(velocity_df) != 1:
        raise ValueError("Velocity model is not homogeneous!")

    # Retrieve medium parameters
    density = velocity_df.at[0,'Rho']
    v_p = velocity_df.at[0,'Vp']
    Q_p = velocity_df.at[0,'Qp']
    v_s = velocity_df.at[0,'Vp']/velocity_df.at[0,'VpVsRatio']
    Q_s = velocity_df.at[0,'Qs']

    # Validate parameters
    validate_parameters(density=density,
                        v_p=v_p,                        
                        v_s=v_s,
                        Q_s=Q_s,
                        Q_p=Q_p,
                        min_amps_p=min_amps_p,
                        min_amps_s=min_amps_s,
                        rad_pattern_p=rad_pattern_p,
                        rad_pattern_s=rad_pattern_s,
                        wave_mode=wave_mode,
                        station_coords=station_coords)

    # Get minimal seismic moment
    if source_mw is None:
        raise ValueError("Source seismic moment can't be None.")
    else:
        sourceSeismicMoment = calculate_seismic_moment(Mw=source_mw)

    # Check minimal picking uncertainty
    if min_sigma is None:
        min_sigma = 1e-3

    # Generate location grid
    loc_grid_coords = generate_grid(x=loc_gx, y=loc_gy, z=loc_gz)

    # Precompute distances between all imaging grid points and all stations
    # Shape: (n_grid_points, n_stations)
    img_distances = np.linalg.norm(grid_coords[:, np.newaxis] - station_coords, axis=2)

    # Precompute distances between all location grid points and all stations
    # Shape: (n_loc_grid_points, n_stations)
    loc_distances = np.linalg.norm(loc_grid_coords[:, np.newaxis] - station_coords, axis=2)

    # Check for NaNs
    if strict_nan_check:
        if np.any(img_distances == 0):
            raise ValueError("Zero imaging distance: source and receiver must have different coordinates")
        if np.any(loc_distances == 0):
            raise ValueError("Zero location distance: source and receiver must have different coordinates")

    # Calculate free surface correction coefficient if needed
    if use_free_surface:
        # Simple approximation
        fs_coef_p = fs_coef_s = 2*np.ones_like(img_distances)
    else:
        # If not using free surface correction, use ones (no effect on calculations)
        fs_coef_p = fs_coef_s = np.ones_like(img_distances)

    # Calculate station directionality if needed
    if use_station_directionality:
        stations_dir_coef_p, stations_dir_coef_s = get_ray_station_directionality(station_coords=station_coords,
                                                                                  grid_coords=grid_coords,
                                                                                  strict_nan_check=strict_nan_check)
    else:
        # If not using directionality, use ones (no effect on calculations)
        stations_dir_coef_p = stations_dir_coef_s = np.ones_like(img_distances)

    # Compute traveltimes and pick uncertainties
    if wave_mode in ('P', 'PS'):
        tpl = loc_distances / v_p # travel times for P waves from location grid to receivers
        tpi = img_distances / v_p # travel times for P waves from imaging grid to receivers

        # Get picking uncertainty for P waves from the peak frequency
        if sigma_p is None:
            sigma_p = 1.0/calculate_fpeak(v=v_p, Q=Q_p, r=img_distances)
        sigma_p = np.maximum(sigma_p, min_sigma)

        # Estimate minimal detectable seismic moment for P wave
        minSeismicMomentP = calculate_M0(density=density,
                                         v=v_p,
                                         Q=Q_p,
                                         rad_pattern=rad_pattern_p,
                                         r=img_distances,
                                         amps = min_amps_p * stations_dir_coef_p / fs_coef_p)        
        
        # Remove P wave arrivals if they are under detection threshold
        tpi[minSeismicMomentP > sourceSeismicMoment] = np.nan
    else:
        tpl = None
        tpi = None

    if wave_mode in ('S', 'PS'):
        tsl = loc_distances / v_s # travel times for S waves from location grid to receivers
        tsi = img_distances / v_s # travel times for S waves from imaging grid to receivers       
        # Get picking uncertainty for S waves from the peak frequency
        if sigma_s is None:
            sigma_s = 1.0/calculate_fpeak(v=v_s, Q=Q_s, r=img_distances)
        sigma_s = np.maximum(sigma_s, min_sigma)
        # Estimate minimal detectable seismic moment for P wave
        minSeismicMomentS = calculate_M0(density=density,
                                         v=v_s,
                                         Q=Q_s,
                                         rad_pattern=rad_pattern_s,
                                         r=img_distances,
                                         amps = min_amps_s * stations_dir_coef_s / fs_coef_s)        
        # Remove S wave arrivals if they are under detection threshold
        tsi[minSeismicMomentS > sourceSeismicMoment] = np.nan
    else:
        tsl = None
        tsi = None

    # Calculate the 3D probability density function
    pdf = calculate_pdf(sigma_p=sigma_p,
                        sigma_s=sigma_s,
                        tpl=tpl,
                        tsl=tsl,
                        tps=tpi,
                        tss=tsi)

    # Get location uncertainty
    azimuth_a, sigma_a, sigma_b, sigma_z = get_uncertainty_from_pdf(
        pdf,
        loc_gx=loc_gx,
        loc_gy=loc_gy,
        loc_gz=loc_gz)

    # Combine uncertainties to a single array
    LU_grid = np.array([azimuth_a, sigma_a, sigma_b, sigma_z])

    # Output depending on the request
    if output_pdf:
        return LU_grid, pdf
    else:
        return LU_grid

def calculate_fpeak(v: float,
                    Q: float,
                    r: float,
                    fcorner: float=None):
    r"""
    Calculates the peak frequency :math:`f_{peak}` for particle velocity
    in a homogeneous medium
    given the wave velocity :math:`v`,
    the attenuation factor :math:`Q`,
    and the distance from a source :math:`r` using:

    .. math::
        f_{peak} = \frac{vQ}{\pi r}.

    The peak frequency estimate is only valid for frequencies below the corner frequency :cite:p:`EisnerGeiEtAl2013`.

    Parameters
    ----------
    v : :obj:`float`
        wave velocity
    Q : :obj:`float`
        attenuation factor
    r : :obj:`float`
        distance from the source
    fcorner : :obj:`float`, optional, default: ``None``
        Corner frequency that limits the peak frequency.
        If ``None``, defaults to 100 Hz.

    References
    ----------
    Eisner, L., Gei, D., Hallo, M., Opršal, I., & Ali, M. Y. (2013).
    The peak frequency of direct waves for microseismic events.
    Geophysics, 78(6), A45–A49.
    https://doi.org/10.1190/geo2013-0197.1

    Returns
    -------
    fpeak : :obj:`float`
        Peak frequency for particle velocity
    """
    # Compute the peak frequency for particle velocity
    fpeak = v*Q/(np.pi*r)

    # Default corner frequency
    if fcorner is None:
        fcorner = 100

    # Limit to corner frequency
    fpeak = np.minimum(fpeak,fcorner)

    return fpeak

def calculate_M0(density: float,
                 v: float,
                 Q: float,
                 rad_pattern: float,
                 r: Optional[Union[float, np.ndarray]],
                 amps: Optional[Union[float, np.ndarray]],
                 f: float = None,
                 fcorner: float = None,
                 amps_type: str='displacement'):
    r"""
    Calculates the seismic moment :math:`M_0` given the density :math:`\rho`, wave velocity :math:`v` and distance from the source :math:`r` using:

    .. math::
        M_0 = \frac{4 \pi \rho v^3 r \Omega_0}{R},

    where :math:`R` is the provided radiation pattern factor and the source spectra :math:`\Omega_0` given the frequency :math:`f` and displacement amplitude :math:`A` is calculated as

    .. math::
        \Omega_0 = \frac{A}{2 \pi f e^{-\pi f t^*}}

    coming from a zero-frequency limit approximation.

    The term :math:`t^*` is the integral along the ray path of the inverse value of attenuation factor :math:`Q` multiplied by the inverse of the wave velocity :math:`v`:

    .. math::
        t^* = \int_R \frac{dr}{v(r) Q(r)},

    which in a homogenous medium reads as

    .. math::
        t^* = \frac{r}{vQ}.

    Frequency :math:`f` is the representative frequency of the wave.
    If it is not provided, it defaults to the peak frequency computed from the model,
    and limited by the corner frequency.

    Parameters
    ----------
    density : :obj:`float`
        Fensity
    v : :obj:`float`
        Wave velocity
    Q : :obj:`float`
        Attenuation factor
    rad_pattern : :obj:`float`
        Radiation pattern factor
    r : :obj:`float` or :obj:`numpy.ndarray`
        Distance(s) from the source
    amps : :obj:`float` or :obj:`numpy.ndarray`
        Displacement amplitudes, single value or array of amplitudes, one for each ray.
        If ``r`` is a single float, ``amps`` must be a single float
        If ``r`` is an array, ``amps`` must be an array of the same shape.
    f : :obj:`float`, optional, default: ``None``
        Frequency. If ``None``, peak frequency is computed from the model
    fcorner : :obj:`float`, optional, default: ``None``
        Corner frequency, limit for peak frequency.
    amps_type : :obj:`str`, optional, default: ``displacement``
        Type of amplitudes in ``amps``, can be 'displacement', 'velocity', 'acceleration', 'strain rate', 'strain'.
        Used for scaling of amplitudes in case ``f`` is ``None``.

    Returns
    -------
    M0 : :obj:`float` or array_like
        Seismic moment
    """
    # Check validity of 'r' and 'amps'
    if amps is None:
        raise ValueError("Missing amplitude values: 'amps' cannot be None.")

    if r is None:
        raise ValueError("Missing distances: 'r' cannot be None.")

    # Check shape and type consistency between 'r' and 'amps'
    r_is_array = isinstance(r, np.ndarray)
    amps_is_array = isinstance(amps, np.ndarray)

    if r_is_array and not amps_is_array:
        raise ValueError("If 'r' is an array, 'amps' must also be an array.")
    if not r_is_array and amps_is_array:
        raise ValueError("If 'r' is a float, 'amps' must also be a float.")
    if r_is_array and amps_is_array:
        if r.shape != amps.shape:
            raise ValueError("If 'r' is an array, 'amps' must have the same shape as 'r'.")

    # Default corner frequency
    if fcorner is None:
        fcorner = 100

    # Compute t*
    t_star = r / (v * Q)

    # Get peak frequency and amplitude scaling
    if f is None:
        # Get fpeak
        fpeak = 1/(np.pi*t_star)
        # Limit to corner frequency
        fpeak = np.minimum(fpeak,fcorner)
        # Final scaling coefficient(s)
        scaling = get_scaling_displacement(amps_type=amps_type,f=fpeak)
    else:
        fpeak = f
        scaling = 1.0

    # Precompute constants
    pif = np.pi * fpeak
    constant = 4 * np.pi * density * v**3 / rad_pattern

    # Compute exp(pi * f * t*)
    exp_term = np.exp(pif * t_star)

    # Compute the source spectra (Omega_0) and seismic moment (M0)
    Omega_0 = scaling * amps * exp_term / (2 * pif)
    M0 = constant * r * Omega_0

    return M0

def validate_parameters(density, v_p, v_s, Q_p, Q_s, min_amps_p, min_amps_s, rad_pattern_p, rad_pattern_s, wave_mode, station_coords):
    r"""
    Validates the input parameters.

    Parameters
    ----------
    density : :obj:`float`
        Density of the medium
    v_p : :obj:`float`
        P wave velocity
    v_s : :obj:`float`  
        S wave velocity
    Q_p : :obj:`float`
        P wave attenuation factor
    Q_s : :obj:`float`
        S wave attenuation factor
    min_amps_p : :obj:`numpy.ndarray`
        Minimal measurable displacement amplitudes of on stations [a1, a2, ...] for P waves
    min_amps_s : :obj:`numpy.ndarray`
        Minimal measurable displacement amplitudes on stations [a1, a2, ...] for S waves
    rad_pattern_p : :obj:`float`
        Radiation pattern factor for P waves
    rad_pattern_s : :obj:`float`
        Radiation pattern factor for S waves
    wave_mode: :obj:`str`
        Wave mode to use, can be 'P','S' or 'PS'    
    station_coords : :obj:`numpy.ndarray`
        Array of station coordinates [[xr1, yr1, zr1], [xr2, yr2, zr2], ...]    

    Raises
    ------
    ValueError :
        if density, wave velocity or attenuation factor is not positive
    ValueError :
        if radiation pattern factor for P or S wave is less or equal to 0 or higher than 1
    ValueError :
        if number of minimal measurable amplitudes does not match number of stations
    ValueError :
        if there is inconsistency in P/S wave parameters
    
    """
    # Check density
    if density <= 0:
        raise ValueError("Density must be positive.")
    if v_p is None and v_s is None:
        raise ValueError("Both P wave and S wave velocities can not be None.")
    if Q_p is None and Q_s is None:
        raise ValueError("Both P wave and S wave attenuation factors can not be None.")

    # Validate P wave parameters if applicable
    if wave_mode in ('P', 'PS'):
        if any(x <= 0 for x in (v_p, Q_p)):
            raise ValueError("P wave parameters must be positive.")
        if min_amps_p is None:
            raise ValueError("Invalid minimal measurable amplitudes for P waves.")
        if np.isscalar(min_amps_p):
            if len(station_coords)!=1:
                raise ValueError("Invalid minimal measurable amplitudes for P waves.")
        elif len(min_amps_p) != len(station_coords):
            raise ValueError("Invalid minimal measurable amplitudes for P waves.")
        if not 0 < rad_pattern_p <= 1:
            raise ValueError("Invalid radiation pattern factor for P waves.")

    # Validate S wave parameters if applicable
    if wave_mode in ('S', 'PS'):
        if any(x <= 0 for x in (v_s, Q_s)):
            raise ValueError("S wave parameters must be positive.")
        if min_amps_s is None:
            raise ValueError("Invalid minimal measurable amplitudes for S waves.")
        if np.isscalar(min_amps_s):
            if len(station_coords)!=1:
                raise ValueError("Invalid minimal measurable amplitudes for S waves.")
        elif len(min_amps_s) != len(station_coords):
            raise ValueError("Invalid minimal measurable amplitudes for S waves.")
        if not 0 < rad_pattern_s <= 1:
            raise ValueError("Invalid radiation pattern factor for S waves.")