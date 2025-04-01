import numpy as np
import pandas as pd
import inspect
from fractions import Fraction
from math import gcd
from typing import Union, Optional
import warnings

def calculate_Mw(M0):
    r"""
    Calculates the moment magnitude :math:`M_w` according to :ref:`Kanamori (1977) <Kanamori1977_calculate_Mw>` using formula

    .. math::

        M_w = \frac{2}{3}\left(\log_{10}(M_0)-9.1\right)

    where :math:`M_0` is the seismic moment.

    Parameters
    ----------
    M0 : array_like
        seismic moment

    References
    ----------
    .. _Kanamori1977_calculate_Mw:

    Kanamori, H. (1977). The energy release in great earthquakes.
    Journal of Geophysical Research, 82(20), 2981–2987.
    https://doi.org/10.1029/jb082i020p02981

    Returns
    -------
    Mw : array_like
        moment magnitude

    Notes
    -----
    Replaces all non-positive values of M0 by nan

    """

    # Ensure input is a NumPy array
    M0 = np.asarray(M0)

    # Replace all non-positive values with NaN
    M0 = np.where(M0 <= 0, np.nan, M0)

    return 2/3 * (np.log10(M0) - 9.1)

def calculate_seismic_moment(Mw):
    r"""
    Calculates the seismic moment :math:`M_0` from moment magnitude :math:`M_w` according to :ref:`Kanamori (1977) <Kanamori1977_calculate_M0>`

    .. math::

        M_0 = 10^{\frac{3}{2} M_w + 9.1}

    where :math:`M_w` is the moment magnitude.

    Parameters
    ----------
    Mw : array_like
        moment magnitude

    References
    ----------
    .. _Kanamori1977_calculate_M0:

    Kanamori, H. (1977). The energy release in great earthquakes.
    Journal of Geophysical Research, 82(20), 2981–2987.
    https://doi.org/10.1029/jb082i020p02981

    Returns
    -------
    M0 : array_like
        seismic moment
    """

    # Ensure input is a NumPy array
    Mw = np.asarray(Mw)

    # Compute moment
    M0 = 10**((3/2)*Mw + 9.1)

    return M0

def reshape_data(data: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray):
    r"""
    Reshape data to 2D array using axes vectors.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        1D data array to be reshaped
    x : :obj:`numpy.ndarray`
        x axis vector
    y : :obj:`numpy.ndarray`
        y axis vector

    Returns
    -------
    reshaped_data : :obj:`numpy.ndarray`
        2D reshaped data array

    Raises
    ------
    ValueError
        when data array is not 1D
    """
    if data.ndim!=1:
        raise ValueError("Input data array must be 1D!")
    else:
        return np.reshape(data, (len(x), len(y)))

def generate_grid(x: np.ndarray,
                  y: np.ndarray,
                  z: np.ndarray):
    r"""
    From the grid vectors x,y,z generates an array of grid point coordinates of size (nx*ny*nz,3).

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        x axis grid vector
    y : :obj:`numpy.ndarray`
        y axis grid vector
    z : :obj:`numpy.ndarray`
        z axis grid vector

    Returns
    -------
    grid_points : :obj:`numpy.ndarray`
        2D grid coordinates array
    """
    # Generate the grid points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten the grid points into an array of coordinates
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return grid_points

def reshape_grid(grid_array: np.ndarray,
                 nx: int,
                 ny: int,
                 nz: int):
    r"""
    Reshape a flat grid array into a multidimensional array based on the provided grid vectors.

    Parameters
    ----------
    grid_array : :obj:`numpy.ndarray`
        The flat grid array to reshape. The first dimension should match the total number of grid points.
    nx : :obj:`numpy.int`
        Number of grid points along the x axis.
    ny : :obj:`numpy.int`
        Number of grid points along the y axis.
    nz : :obj:`numpy.int`
        Number of grid points along the z axis.

    Raises
    ------
    ValueError :
        If the number of grid points does not match the product of nx, ny, and nz.

    Returns
    -------
    reshaped_array : :obj:`numpy.ndarray`
        The reshaped array with shape (nx, ny, nz, ...), where "..." corresponds to the remaining dimensions of the input array.
    """
    # Calculate the target shape
    target_shape = (nx, ny, nz) + grid_array.shape[1:]

    # Reshape the array
    try:
        reshaped_array = grid_array.reshape(target_shape)
    except ValueError:
        raise ValueError(
            f"Cannot reshape grid_array with shape {grid_array.shape} to target shape {target_shape}. "
            "Ensure the number of grid points matches the product of nx, ny, and nz."
        )

    return reshaped_array

def are_points_inside(grid_coords1, grid_coords2):
    r"""Check if all points in ``grid_coords1`` are inside ``grid_coords2``.

    This function checks whether all points in the array ``grid_coords1`` are also present in the array ``grid_coords2``.
    Both arrays should have a shape of ``(n, 3)``, where each row represents a 3D coordinate.

    Parameters
    ----------
    grid_coords1 : :obj:`numpy.ndarray`
        A 2D array of shape (n_grid1, 3) representing the points to check.
    grid_coords2 : :obj:`numpy.ndarray`
        A 2D array of shape (n_grid2, 3) representing the reference points.

    Returns
    -------
    :obj:`bool`
        ``True`` if all points in ``grid_coords1`` are found in ``grid_coords2``, ``False`` otherwise.

    Raises
    ------
    ValueError
        If ``grid_coords1`` or ``grid_coords2`` does not have the required shape of ``(n, 3)``.

    Examples
    --------
    >>> grid_coords1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> grid_coords2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> are_points_inside(grid_coords1, grid_coords2)
    True

    >>> grid_coords1 = np.array([[1, 2, 3], [10, 11, 12]])
    >>> grid_coords2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> are_points_inside(grid_coords1, grid_coords2)
    False
    """
    # Validate input shapes
    if grid_coords1.shape[1] != 3 or grid_coords2.shape[1] != 3:
        raise ValueError("Both input arrays must have a shape of (n, 3).")

    # Convert grid_coords2 to a set of tuples for efficient lookup
    grid_coords2_set = set(map(tuple, grid_coords2))

    # Check if all points in grid_coords1 are in grid_coords2
    return all(tuple(point) in grid_coords2_set for point in grid_coords1)

def find_indices_of_points(grid_coords1, grid_coords2):
    r"""Find indices of points in ``grid_coords1`` that are inside ``grid_coords2``.

    This function finds the indices of points in ``grid_coords1`` that are also present in ``grid_coords2``.

    Parameters
    ----------
    grid_coords1 : :obj:`numpy.ndarray`
        A 2D array of shape ``(n_grid1, 3)`` representing the points to check.
    grid_coords2 : :obj:`numpy.ndarray`
        A 2D array of shape ``(n_grid2, 3)`` representing the reference points.

    Returns
    -------
    indices : :obj:`numpy.ndarray`
        An array of indices corresponding to the points in ``grid_coords1`` that are found in ``grid_coords2``.

    Raises
    ------
    ValueError
        If ``grid_coords1`` or ``grid_coords2`` does not have the required shape of ``(n, 3)``.

    Examples
    --------
    >>> grid_coords1 = np.array([[1, 2, 3], [4, 5, 6], [10, 11, 12]])
    >>> grid_coords2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> find_indices_of_points(grid_coords1, grid_coords2)
    array([0, 1])
    """
    # Validate input shapes
    if grid_coords1.shape[1] != 3 or grid_coords2.shape[1] != 3:
        raise ValueError("Both input arrays must have a shape of (n, 3).")

    # Convert grid_coords2 to a set of tuples for efficient lookup
    grid_coords2_set = set(map(tuple, grid_coords2))

    # Find indices of points in grid_coords1 that are in grid_coords2
    indices = [i for i, point in enumerate(grid_coords1) if tuple(point) in grid_coords2_set]

    return np.array(indices, dtype=int)


def get_slice_coordinates(grid_points: np.ndarray,
                          slice_dimension: int,
                          slice_coordinate: float):
    r"""
    From the array of grid point coordinates of size (nx*ny*nz,3) retrieve coordinates
    of the slice plane at a given slice coordinate in a given slice dimension.

    Parameters
    ----------
    grid_points : :obj:`numpy.ndarray`
        2D grid coordinates array
    dimension : :obj:`int`
        integer defining the desired slice dimension (0,1 or 2)
    slice_coordinate : :obj:`numpy.ndarray`
        coordinate of the slice to extract in the desired dimension

    Returns
    -------
    plane_points : :obj:`numpy.ndarray`
        2D slice plane coordinates array
    indices : :obj:`numpy.ndarray`
        Indices of plane_points in grid_points

    Raises
    ------
    ValueError
        when slice_dimension is not 0, 1 or 2
    RuntimeError
        when slice can't be extracted (slice coordinate does not coincide with the grid)
    """
    if slice_dimension not in range(3):
        raise ValueError("Slice dimension must be 0,1 or 2. Wrong dimension: " + str(slice_dimension))

    # Find indices where the slice condition is true
    indices = np.where(grid_points[:, slice_dimension] == slice_coordinate)[0]

    # Get coordinates of the slice plane using the indices
    plane_points = grid_points[indices]

    # Check output
    if plane_points.size == 0:
        raise RuntimeError("Slice could not be extracted, check slice coordinate: " + str(slice_coordinate))

    return plane_points, indices

def nth_minimum(arr: np.ndarray, n: int) -> np.ndarray:
    r"""
    Find the n-th minimum value in each row of a given 2D numpy array.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray`
        A 2D numpy array of shape (i, j) where i is the number of rows (features)
        and j is the number of columns (samples).
    n : :obj:`int`
        The order of the minimum value to find (1-based index). For example, if n=2,
        the function finds the 2nd minimum value in each row.

    Returns
    -------
    nth_min_values : :obj:`numpy.ndarray`
        A 1D numpy array of shape (i,) containing the n-th minimum value from each row.

    Raises
    ------
    ValueError
        If n is less than 1 or greater than the number of columns in the array.
    """
    if n < 1 or n > arr.shape[1]:
        raise ValueError("n must be between 1 and the number of columns in the array.")

    # Partially sort each row to get the n-th minimum value
    partitioned_arr = np.partition(arr, n-1, axis=1)

    # Extract the n-th minimum value (zero-based index, so n-1)
    nth_min_values = partitioned_arr[:, n-1]

    return nth_min_values

def check_zero_values(a: float, ErrorMessage: str=None):
    r"""
    Check if the input value is close to zero and raise an error if true.

    Parameters
    ----------
    a : :obj:`float` or array_like
        The input value or values to be checked.
    ErrorMessage : :obj:`str` , optional
        Custom error message to be raised if the value is close to zero.
        Default is "Array values are close to zero".

    Returns
    -------
    bool
        Returns True if the value is not close to zero.

    Raises
    ------
    RuntimeError
        If the value is close to zero, raises a RuntimeError with the provided or default error message.

    Examples
    --------
    >>> check_zero_values(0.0000001)
    RuntimeError: Array values are close to zero

    >>> check_zero_values(1.0)
    True
    """
    if ErrorMessage is None:
        ErrorMessage = "Array values are close to zero"
    if np.any(np.isclose(a,0)):
        raise RuntimeError(ErrorMessage)
    else:
        return True

def contains_only_numeric(seq):
    r"""
    Check if a sequence (tuple, list, or numpy array) contains only numeric values.

    Parameters
    ----------
    seq : :obj:`tuple`, :obj:`list`, or :obj:`numpy.ndarray`
        The sequence to be checked.

    Returns
    -------
    bool
        True if all elements in the sequence are numeric (int, float, or complex), False otherwise.

    Examples
    --------
    >>> t1 = (1, 2.5, 3+4j)
    >>> contains_only_numeric(t1)
    True

    >>> l1 = [1, 2.5, 3]
    >>> contains_only_numeric(l1)
    True

    >>> a1 = np.array([1, 2.5, 3+4j])
    >>> contains_only_numeric(a1)
    True

    >>> t2 = (1, 'a', 2.5)
    >>> contains_only_numeric(t2)
    False
    """
    if isinstance(seq, (tuple, list, np.ndarray)):
        return all(isinstance(i, (int, float, complex)) for i in seq)
    else:
        raise TypeError("Input must be a tuple, list, or numpy.ndarray")

def compute_gcd_of_floats(rx, ry, rz):
    r"""
    Computes the greatest common divisor (GCD) of three positive float values.

    Parameters
    ----------
    rx : :obj:`float`
        First float value.
    ry : :obj:`float`
        Second float value.
    rz : :obj:`float`
        Third float value.

    Returns
    -------
    :obj:`float`
        The GCD of the three float values.

    """
    # Convert floats to Fractions for exact arithmetic
    frac_rx = Fraction(rx).limit_denominator()
    frac_ry = Fraction(ry).limit_denominator()
    frac_rz = Fraction(rz).limit_denominator()

    # Extract numerators and denominators
    num_rx, den_rx = frac_rx.numerator, frac_rx.denominator
    num_ry, den_ry = frac_ry.numerator, frac_ry.denominator
    num_rz, den_rz = frac_rz.numerator, frac_rz.denominator

    # Compute GCD of numerators
    gcd_num = gcd(gcd(num_rx, num_ry), num_rz)

    # Compute LCM of denominators
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    lcm_den = lcm(den_rx, lcm(den_ry, den_rz))

    # Compute GCD as a Fraction
    gcd_fraction = Fraction(gcd_num, lcm_den)

    # Return the GCD as a float
    return float(gcd_fraction)

def get_scaling_displacement(amps_type: str,
                             f: Optional[Union[float, np.ndarray]]):
    r"""
    Get scaling coefficient(s) to convert to the amplitudes to displacement
    provided the type of original amplitudes and the frequency (or array of frequencies).
    If f is an array and is used for computation of scaling, output is also an array of the same size.

    Parameters
    ----------
    amps_type : :obj:`str`
        Type of amplitudes, can be 'displacement', 'velocity', 'acceleration', 'strain rate', 'strain'.
    f : :obj:`float` or :obj:`np.ndarray`
        Frequency or array of frequencies.

    Raises
    ------
    ValueError
        If ``amps_type`` is not one of the supported types.

    Returns
    -------
    scaling : :obj:`float` or :obj:`np.ndarray`
        Resulting scaling coefficient.
    """
    fscaling = 2 * np.pi * f
    if amps_type == 'displacement':
        scaling = 1
    elif amps_type == 'velocity':
        scaling = 1 / fscaling
    elif amps_type == 'acceleration':
        scaling = 1 / fscaling**2
    elif amps_type == 'strain rate':
        scaling = 1 / fscaling
    elif amps_type == 'strain':
        scaling = 1
    else:
        raise ValueError(f"Unsupported amplitude type: {amps_type}")

    return scaling

def retrieve_min_amps(geometry_df: pd.DataFrame,
                      SNR_p: float,
                      SNR_s: float,
                      f_p: float=None,
                      f_s: float=None,
                      sample_interval: float=None):
    r"""
    Retrieve minimum amplitudes for P and S waves from geometry data,
    scaled based on noise type.

    This function takes geometry data frame with noise levels, then scales these
    noise levels based on the noise type to displacement
    using the provided frequencies for P and S waves (optional),
    and applies SNR scaling for P and S waves.

    Parameters
    ----------
    geometry_df : :obj:`pandas.DataFrame`
        The path to the input CSV (ASCII) file containing the geometry data.
    SNR_p : :obj:`float`
        Signal-to-Noise Ratio for P waves
    SNR_s : :obj:`float`
        Signal-to-Noise Ratio for S waves
    f_p: :obj:`float`, optional, default: ``None``
        Frequency of the P-wave.
        If ``None``, scaling is postponed to be later done with peak frequency
    f_s: :obj:`float`, optional, default: ``None``
        Frequency of the S-wave.
        If ``None``, scaling is postponed to be later done with peak frequency
    sample_interval : :obj:`float`
        Interval along the cable at which strain measurements are made

    Returns
    -------
    min_amps_p : :obj:`numpy.ndarray`
        Minimum displacement amplitudes for P waves
    min_amps_s : :obj:`numpy.ndarray`
        Minimum displacement amplitudes for S waves

    Raises
    ------
    ValueError
        If an unsupported noise type is encountered.
    ValueError
        If the 'NoiseLevel' column is missing in geometry_df.
    ValueError
        If the sample_interval is None for strain and strain rate noise types
    """
    # Check if 'NoiseLevel' column exists
    if 'NoiseLevel' not in geometry_df.columns:
        raise ValueError("The 'NoiseLevel' column is missing from the geometry data.")

    # Get the noise levels and type
    noise_levels = geometry_df['NoiseLevel'].to_numpy()
    noise_type = geometry_df.attrs.get('noise_type')

    # Check sample_interval
    if (noise_type == 'strain rate' or noise_type == 'strain') and sample_interval is None:
        raise ValueError(f"Sample interval must be provided for {noise_type} data type." )

    # Scale noise levels to displacement based on noise type
    if f_p is None:
        scaling_p = 1 # scaling will be done later using peak frequency
    else:
        scaling_p = get_scaling_displacement(amps_type=noise_type,f=f_p)

    if f_s is None:
        scaling_s = 1 # scaling will be done later using peak frequency
    else:
        scaling_s = get_scaling_displacement(amps_type=noise_type,f=f_s)

    # Apply sample interval for strain and strain rate
    if noise_type == 'strain rate' or noise_type == 'strain':
        scaling_p *= sample_interval
        scaling_s *= sample_interval

    # Calculate final minimum amplitudes for P and S waves by applying scaling and SNR
    min_amps_p = noise_levels * scaling_p * SNR_p
    min_amps_s = noise_levels * scaling_s * SNR_s

    return min_amps_p, min_amps_s

def mag_detectable(Mw_min_stations: np.ndarray,
                   min_stations_p: int = None,
                   min_stations_s: int = None,
                   wave_mode: str = 'PS'):
    r"""
    Computes detectable minimum magnitudes from magnitude sensitivities for individual stations
    with respect to number of stations necessary for event detection using P and/or S waves.

    Parameters
    ----------
    Mw_min_stations : :obj:`numpy.ndarray`
        Array of computed minimum magnitudes for individual stations
    min_stations_p : :obj:`int`, optional, default: 3
        Minimum number of stations on which event must be detected with P waves
    min_stations_s : :obj:`int`, optional, default: 3
        Minimum number of stations on which event must be detected with S waves
    wave_mode: :obj:`str`, optional, default: 'PS'
        Wave mode to use, can be 'P','S' or 'PS'

    Returns
    -------
    Mw_min_grid : :obj:`numpy.ndarray`
        Array of detectable minimum magnitudes on the grid

    Raises
    ------
    ValueError :
        if minimum number of stations on which event must be detected with P or S waves is less than 1 or exceeds number of stations
    """
    if min_stations_p is None:
        min_stations_p = 3
    if min_stations_s is None:
        min_stations_s = 3

    if not 0 < min_stations_p <= len(Mw_min_stations[0]):
        raise ValueError("Invalid minimum number of stations for P waves.")
    if not 0 < min_stations_s <= len(Mw_min_stations[1]):
        raise ValueError("Invalid minimum number of stations for S waves.")

    # Compute final Mw_min_grid based on wave mode
    if wave_mode == 'P':
        # For P waves, find the nth smallest value (where n = min_stations_p)
        Mw_min_grid = np.partition(Mw_min_stations[0], min_stations_p - 1, axis=1)[:, min_stations_p - 1]
    elif wave_mode == 'S':
        # For S waves, find the nth smallest value (where n = min_stations_s)
        Mw_min_grid = np.partition(Mw_min_stations[1], min_stations_s - 1, axis=1)[:, min_stations_s - 1]
    else:  # wave_mode == 'PS'
        # For both P and S waves, find the maximum of the nth smallest values for each
        Mw_min_grid_p = np.partition(Mw_min_stations[0], min_stations_p - 1, axis=1)[:, min_stations_p - 1]
        Mw_min_grid_s = np.partition(Mw_min_stations[1], min_stations_s - 1, axis=1)[:, min_stations_s - 1]
        Mw_min_grid = np.maximum(Mw_min_grid_p, Mw_min_grid_s)

    return Mw_min_grid

def get_ray_station_directionality(station_coords: np.ndarray,
                                   grid_coords: np.ndarray,
                                   rays_p: list=None,
                                   rays_s: list=None,
                                   strict_nan_check: bool=False):
    r"""
    Derive directionality coefficients for P and S waves
    for each station using rays to this station.
    Takes into account the sequence of stations.

    Parameters
    ----------
    station_coords : :obj:`numpy.ndarray`
        Array of station coordinates [[xr1, yr1, zr1], [xr2, yr2, zr2], ...]
    grid_coords : :obj:`numpy.ndarray`, optional
        Array of grid coordinates [[xs1, ys1, zs1], [xs2, ys2, zs2], ...]
    rays_p: :obj:`list` of :obj:`numpy.ndarray`
        If provided, must contain exactly one ray path from each (station, grid) pair,
        sorted from that station (first row) to that grid point (last row).
        The path's shape is ``(m,3)``.
        Used for P-wave directionality.
    rays_s: :obj:`list` of :obj:`numpy.ndarray`
        Same as rays_p but for S-wave directionality.
    strict_nan_check: :obj:`bool`, optional, default: False
        If True, raise a RuntimeError if NaNs appear in the resulting
        directionality coefficients.

    Returns
    -------
    dir_coef_p : :obj:`numpy.ndarray`
        Array of correction coefficients for P wave, for each grid point and station
    dir_coef_s : :obj:`numpy.ndarray`
        Array of correction coefficients for S wave, for each grid point and station

    Raises
    ------
    ValueError :
        if len(station_coords) is less than 2
    RuntimeError :
        if strict_nan_check is True and NaN appears in the result

    Notes
    -----
    For each station it computes the correction coefficient for P and S wave.
    If no `rays_p` and `rays_s` are provided, we treat the medium as homogeneous.
    For P wave the correction coefficient is equal to the inverse of the absolute value of cosine of the angle between
    the line connecting the current station with each grid point (CG line)
    and the line connecting the current station with the next station (CN line).
    If there is no next station (the current station is the last one),
    the line connecting the current station with the previous station (CP line) is used instead of the CN line.
    If the angle between the CN and CP lines is larger than 30 degrees
    or if the distance between the current station and the next station (CN distance)
    is 10 times larger than the distance between the current station and the previous station (CP distance),
    the CP line is used for calculation of the angle with CG line, whereas for the next station the CN line is always used.
    For S wave the correction coefficient is equal to the inverse of the absolute value of the sine of the same angle as described for the P wave.
    This is a vectorized version of the function. It computes the directionality
    coefficients for all grid points and stations simultaneously, which should be much
    more efficient for large datasets.

    If `rays_p` is provided, we do NOT assume a simple straight ray (line)
    from grid point to station; instead, we use the ray path for that pair,
    extract only the first segment near the station, and treat that as CG.
    Ray is considered to start from station.
    The same logic applies if `rays_s` is provided for the S-wave.
    """
    # Check number of stations
    if len(station_coords) < 2:
        raise ValueError("There must be at least two stations.")

    # Determine which wave(s) we are actually computing
    # per user specification:
    #  - if both rays_p and rays_s are None => compute both P & S, assume straight ray
    #  - if rays_p is not None and rays_s is None => only P
    #  - if rays_p is None and rays_s is not None => only S
    #  - if both are provided => both P & S
    straight_rays = False
    if rays_p is not None and rays_s is not None:
        compute_p, compute_s = True, True
    elif rays_p is not None:  # only rays_p is provided
        compute_p, compute_s = True, False
    elif rays_s is not None:  # only rays_s is provided
        compute_p, compute_s = False, True
    else:
        straight_rays = True
        compute_p, compute_s = True, True  # neither is provided => both

    # Get lengths
    n_stations = station_coords.shape[0]
    n_grid_points = grid_coords.shape[0]

    # Calculate vectors between stations
    station_vectors = np.diff(station_coords, axis=0)

    # Pad the first and last rows to handle edge cases
    cn_vectors = np.vstack([station_vectors, station_vectors[-1]])
    cp_vectors = np.vstack([station_vectors[0], station_vectors])

    # Calculate distances for distance checks
    cn_distances = np.linalg.norm(cn_vectors, axis=1)
    cp_distances = np.linalg.norm(cp_vectors, axis=1)

    # Calculate angles between CN and CP vectors
    angles_cn_cp = np.arccos(np.sum(cn_vectors * cp_vectors, axis=1) /
                             (cn_distances * cp_distances))

    # Determine whether to use CP vector based on angle and distance conditions
    use_cp = (angles_cn_cp > np.radians(30)) | (cn_distances > 10 * cp_distances)

    # Prepare vectors for angle calculation
    vectors_for_angle = np.where(use_cp[:, np.newaxis], cp_vectors, cn_vectors)

    # Function to build cg vector using P and S rays
    def build_cg_vectors(rays):
        r"""
        Given ray paths in ``rays``, build an array of CG vectors
        of shape ``(n_grid_points, n_stations, 3)``.
        Use the first segment of the provided ray path near the station.
        """

        cg = np.zeros((n_grid_points, n_stations, 3), dtype=float)

        for i_st in range(n_stations):
            for j_gr in range(n_grid_points):
                iray = i_st*n_grid_points + j_gr
                path = rays[iray]
                if path is None:
                    raise ValueError(f"No ray found for station #{i_st} and grid #{j_gr}")
                if path.shape[0] < 2:
                    raise ValueError(
                        f"Ray path for station #{i_st}, grid #{j_gr} "
                        f"has too few points ({path.shape[0]})."
                    )
                stc = station_coords[i_st]
                if not np.all(np.isclose(path[0], stc, atol=1e-03)):
                    #raise ValueError(f"Ray {iray} ({path[0]}) does not start at the station {i_st} ({stc})")
                    warnings.warn(f"Ray {iray} start point ({path[0]}) differ from station {i_st} coordinates ({stc})")
                cg[j_gr, i_st, :] = path[1] - path[0]
        return cg

    if straight_rays:
        cg_vectors = grid_coords[:, np.newaxis, :] - station_coords[np.newaxis, :, :]
    else:
        cg_vectors_p = build_cg_vectors(rays_p) if compute_p else None
        cg_vectors_s = build_cg_vectors(rays_s) if compute_s else None

    # Function to compute directionality from a set of CG vectors
    # cg_vectors shape: (n_grid_points, n_stations, 3)
    def compute_dir_coeffs_for_cg_vectors(cg_vectors):
        r"""
        Given ``cg_vectors`` of shape ``(n_grid_points, n_stations, 3)``, compute
        ``dir_coef_p`` and ``dir_coef_s`` using the angle with ``vectors_for_angle``.
        Returns ``(dir_coef_p, dir_coef_s)`` each shaped as ``(n_grid_points, n_stations)``.
        """
        # Normalize vectors_for_angle per station
        # shape: (n_stations, 3)
        station_ref_norm = vectors_for_angle / np.linalg.norm(vectors_for_angle, axis=1, keepdims=True)

        # Now we want the dot-product with each cg_vector
        # cg_vectors shape: (n_grid_points, n_stations, 3)
        # We'll normalize cg_vectors along axis=2
        cg_norm = cg_vectors / np.linalg.norm(cg_vectors, axis=2, keepdims=True)

        # cos(angle) = abs( dot( cg_norm, station_ref_norm ) )
        # We'll expand station_ref_norm so we can broadcast: shape -> (1, n_stations, 3)
        dot_vals = np.sum(cg_norm * station_ref_norm[np.newaxis, :, :], axis=2)
        cos_vals = np.abs(dot_vals)
        # sin(angle) = sqrt(1 - cos^2(angle))
        sin_vals = np.sqrt(1.0 - cos_vals**2)

        # Zero or near-zero cos/sin => we'd get inf => we can turn them into NaN
        cos_near_zero = np.isclose(cos_vals, 0.0)
        sin_near_zero = np.isclose(sin_vals, 0.0)

        cos_vals[cos_near_zero] = np.nan
        sin_vals[sin_near_zero] = np.nan

        dir_coef_p = 1.0 / cos_vals   # shape (n_grid_points, n_stations)
        dir_coef_s = 1.0 / sin_vals   # shape (n_grid_points, n_stations)

        if strict_nan_check:
            if np.any(np.isnan(dir_coef_p)) or np.any(np.isnan(dir_coef_s)):
                current_function = inspect.currentframe().f_code.co_name
                raise RuntimeError(
                    f"NaN values detected while computing directionality "
                    f"coefficients in {current_function}."
                )

        return dir_coef_p, dir_coef_s

    # Compute directionality coefficients for whichever waves we need
    if straight_rays:
        dir_coef_p, dir_coef_s = compute_dir_coeffs_for_cg_vectors(cg_vectors)
    else:
        dir_coef_p = None
        dir_coef_s = None
        if compute_p:
            dir_coef_p, _ = compute_dir_coeffs_for_cg_vectors(cg_vectors_p)
        if compute_s:
            _, dir_coef_s = compute_dir_coeffs_for_cg_vectors(cg_vectors_s)

    return dir_coef_p, dir_coef_s
