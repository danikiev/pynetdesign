import numpy as np
import pandas as pd
import re
import copy
from typing import List, Dict, Union

def combine_geometry(*dataframes: pd.DataFrame, names: Union[List[str], Dict[int, str]] = None) -> pd.DataFrame:
    r"""
    Combine an arbitrary number of pandas DataFrames with geometries into one while retaining
    the ability to distinguish and recover the original DataFrames.

    This function adds a 'source' column to each DataFrame to identify its
    origin, concatenates them, and sets a multi-level index using the 'source'
    column and the original index. 
    
    It also stores and recovers attributes of the original dataframes.

    Parameters
    ----------
    *dataframes : :obj:`pandas.DataFrame`
        An arbitrary number of DataFrames to be combined.
    names : :obj:`typing.Union[List[str], Dict[int, str]]`, optional
        Names to use as identifiers for the DataFrames. Can be a list of strings
        or a dictionary mapping DataFrame positions to names. If not provided,
        default names 'df0', 'df1', etc. will be used.

    Returns
    -------
    combined: :obj:`pandas.DataFrame`
        A combined DataFrame with a multi-level index and additional methods:
        
        - ``get_names()``: Returns a list of internal DataFrame names;
        - ``recover(name)``: Recovers a specific DataFrame by name.

    Raises
    ------
    ValueError
        If no DataFrames are provided, or if the number of names doesn't match
        the number of DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    >>> df3 = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})
    >>> combined = combine_df(df1, df2, df3, names=['First', 'Second', 'Third'])
    >>> print(combined.get_names())
    ['First', 'Second', 'Third']
    >>> recovered_df2 = combined.recover('Second')
    >>> print(recovered_df2)
       A  B
    0  5  7
    1  6  8
    """
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided.")

    if names is None:
        names = [f'df{i}' for i in range(len(dataframes))]
    elif isinstance(names, dict):
        names = [names.get(i, f'df{i}') for i in range(len(dataframes))]
    elif len(names) != len(dataframes):
        raise ValueError("The number of names must match the number of DataFrames.")

    # Store attributes of each DataFrame
    attributes = {}
    labeled_dfs = []
    # noise_units = set()
    # noise_types = set()
    for df, name in zip(dataframes, names):
        df_copy = df.copy()
        df_copy['source'] = name
        labeled_dfs.append(df_copy)
        # Save all attributes, including private ones, using a deep copy
        attributes[name] = copy.deepcopy(df.attrs)
        # Add noise unit to the list
        # noise_units.add(df.attrs.get('noise_unit', 'Unknown'))
        # noise_types.add(df.attrs.get('noise_type', 'Unknown'))

    # Combine the DataFrames
    combined_df = pd.concat(labeled_dfs)
    combined_df.set_index(['source', combined_df.index], inplace=True)

    # Store the attributes in the combined DataFrame
    combined_df.attrs['original_attributes'] = attributes

    # Add method to return the names of internal DataFrames
    combined_df.get_names = lambda: combined_df.index.get_level_values('source').unique().tolist()

    # Add method to recover a specific DataFrame by name
    def recover(name):
        if name not in combined_df.index.get_level_values('source'):
            raise ValueError(f"No DataFrame named '{name}' exists in the combined DataFrame.")
        
        recovered = combined_df.loc[name].reset_index(drop=True)
        
        # Check if 'Name' column exists and all its values are NaN
        if 'Name' in recovered.columns and recovered['Name'].isna().all():
            recovered = recovered.drop(columns=['Name'])
        
        # Restore original attributes exactly as they were
        if name in combined_df.attrs['original_attributes']:
            recovered.attrs = copy.deepcopy(combined_df.attrs['original_attributes'][name])
        
        return recovered

    combined_df.recover = recover

    # Add a special attribute to identify this as a combined DataFrame
    combined_df.attrs['_is_combined'] = True
    
    # Add noise unit if it is consistent across the all input dfs
    # if len(noise_units) == 1:
    #     combined_df.attrs['noise_unit'] = noise_units.pop()
    #     combined_df.attrs['noise_type'] = noise_types.pop()

    # Add a method to check if this is a combined DataFrame
    combined_df.is_combined = lambda: True

    return combined_df

def is_combined_geometry(df: pd.DataFrame) -> bool:
    r"""
    Check if the given DataFrame is a combined geometry DataFrame created by combine_geometry.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        The DataFrame to check.

    Returns
    -------
    bool
        True if the DataFrame is a combined one, False otherwise.
    """
    # Check for the special attribute
    if '_is_combined' in df.attrs and df.attrs['_is_combined']:
        return True
    
    # Check for the is_combined method
    if hasattr(df, 'is_combined') and callable(df.is_combined) and df.is_combined():
        return True
    
    return False

def global2local(latY: np.ndarray,
                 lonX: np.ndarray, 
                 refLat: float=None, 
                 refLon: float=None):
    """
    Converts global latitude and longitude coordinates to Cartesian coordinates 
    under a simple spherical Earth approximation.

    Parameters
    ----------
    latY : :obj:`numpy.ndarray`
        An array of latitude coordinates in degrees.
    lonX : :obj:`numpy.ndarray`
        An array of longitude coordinates in degrees.
    refLat : :obj:`float`, optional
        The reference latitude for the local coordinate system. If not provided, 
        the average of the input latitude coordinates will be used.
    refLon : :obj:`float`, optional
        The reference longitude for the local coordinate system. If not provided, 
        the average of the input longitude coordinates will be used.

    Returns
    -------
    x_local_coords : :obj:`list`
        A list of local x coordinates in meters.
    y_local_coords : :obj:`list`
        A list of local y coordinates in meters.
    refLat : :obj:`float`   
        The reference latitude used for the local coordinate system.
    refLon : :obj:`float`
        The reference longitude used for the local coordinate system.

    Examples
    --------    
    >>> latY = np.array([37.7749, 34.0522, 40.7128])
    >>> lonX = np.array([-122.4194, -118.2437, -74.0060])
    >>> x_local_coords, y_local_coords, refLat, refLon = global2local(latY, lonX)
    """

    x_local_coords = []
    y_local_coords = []
    EarthD = 12756320  # Earth diameter in meters
    
    if refLat is None or refLon is None:
        refLon = sum(lonX) / len(lonX)
        refLat = sum(latY) / len(latY)
    
    for x in lonX:
        x_local_coord = (x - refLon) * EarthD * np.pi * np.cos(refLat / 180 * np.pi) / 360
        x_local_coords.append(x_local_coord)
    
    for y in latY:
        y_local_coord = (y - refLat) * EarthD * np.pi / 360
        y_local_coords.append(y_local_coord)
    
    return x_local_coords, y_local_coords, refLat, refLon

def read_geometry(file_path: str) -> pd.DataFrame:
    r"""
    Read geometry data from a whitespace-delimited CSV (ASCII) file and return a DataFrame.

    This function reads a CSV (ASCII) file containing geometry information, ensuring
    that coordinate columns (Latitude/Northing, Longitude/Easting, Elevation) and
    NoiseLevel column are present. The 'Name' column is optional.     
    The resulting DataFrame has columns renamed and units converted as necessary.
    If header has 'GaugeLength=', the appropriate gauge length is read and stored in the attributes.

    Parameters
    ----------
    file_path : :obj:`str`
        The path to the input CSV (ASCII) file containing the geometry data.    

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the renamed columns:

        - 'Name' for Name (if present in the input file)
        - 'Y' for Latitude or Northing (converted to meters)
        - 'X' for Longitude or Easting (converted to meters)
        - 'Z' for Depth (negative of Elevation, converted to meters)
        - 'NoiseLevel' for Noise Level

        The DataFrame also includes metadata:
        
        - ``df.attrs['original_coord_units']``: Original units of coordinate columns
        - ``df.attrs['noise_type']``: Type of noise measurement
        - ``df.attrs['noise_unit']``: Original unit of noise measurement
        - ``df.attrs['gauge_length']``: Gauge length (if specified in the header)
        - ``df.attrs['_is_combined']``: Combined status (False)

    Raises
    ------
    ValueError :
        If required columns are missing, unknown columns are present, inconsistent units 
        are found for coordinate columns, or if an unrecognized unit is encountered for 
        the NoiseLevel column or GaugeLength.

    Notes
    -----
    - The function handles whitespace and tabs as delimiters.
    - It recognizes and converts units for coordinate columns (m, km, ft).
    - For the NoiseLevel column, it recognizes units: m (displacement), m/s (velocity),
      m/s^2 (acceleration), and 1/s (strain).
    - The 'Z' column is converted to represent depth (negative of elevation).
    - The 'Name' column, if present, is interpreted as strings.
    - Only known columns ('Name', 'Latitude', 'Northing', 'Longitude', 'Easting', 
      'Elevation', 'NoiseLevel') or combinations like 'Latitude/Northing' or 
      'Latitude(WGS84)/Northing(m)' are read from the file.
    - For Latitude and Longitude conversion to m is not implemented, inputs are treated as meters
    - The function will raise an error if any unknown columns are present in the input file.


    Examples
    --------
    >>> df = read_geometry("data/geometry.csv")
    >>> print(df.head())
    >>> print(df.attrs)  # To view metadata
    """
    # Read the first line of the file to get the header
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()

    # Check for GaugeLength in the header
    gauge_length = None
    gauge_length_pattern = r'GaugeLength=(\d+(?:\.\d+)?)?(?:\((\w+)\))?'
    gauge_length_match = re.search(gauge_length_pattern, header_line)
    
    if gauge_length_match:
        gauge_value_str = gauge_length_match.group(1)
        gauge_unit = gauge_length_match.group(2)

        if gauge_value_str is None:
            raise ValueError("GaugeLength= found in header but no value provided")
        
        gauge_value = float(gauge_value_str)
        
        # If no unit is specified, assume meters
        if gauge_unit is None:
            gauge_length = gauge_value
        else:
            # Convert to meters
            if gauge_unit == 'cm':
                gauge_length = gauge_value / 100
            elif gauge_unit == 'mm':
                gauge_length = gauge_value / 1000
            elif gauge_unit == 'km':
                gauge_length = gauge_value * 1000
            elif gauge_unit == 'ft':
                gauge_length = gauge_value * 0.3048
            elif gauge_unit == 'in':
                gauge_length = gauge_value * 0.0254
            elif gauge_unit == 'm':
                gauge_length = gauge_value
            else:
                raise ValueError(f"Unrecognized unit for GaugeLength: {gauge_unit}")
        
        # Remove GaugeLength part from the header
        header_line = re.sub(gauge_length_pattern, '', header_line).strip()
    
    # Split the header into columns
    header = header_line.split()

    def extract_unit(col_name):
        match = re.search(r'\(([^()]*)\)$', col_name)
        if match:
            unit = match.group(1)
            base_name = col_name[:match.start()].rstrip('/')
            base_name = re.sub(r'\([^()]*\)', '', base_name).strip()
            return base_name, unit
        return col_name, None

    # Extract base names and units for each column
    column_info = {col: extract_unit(col) for col in header}

    # Check for required columns, known and unknown columns
    required_columns = {'Y': False, 'X': False, 'Z': False, 'NoiseLevel': False}
    # Define known columns
    known_columns = ['Name', 'Latitude', 'Northing', 'Longitude', 'Easting', 'Elevation', 'NoiseLevel']
    # Define unknown columns
    unknown_columns = []
    # Default lat/lon flags
    has_lat = False
    has_lon = False    

    # Create a list to store the columns to be read
    columns_to_read = []
    
    for col, (base_name, _) in column_info.items():
        if any(known_col in base_name for known_col in known_columns):
            columns_to_read.append(col)
            if 'Latitude' in base_name or 'Northing' in base_name:
                required_columns['Y'] = True
                if 'Northing' not in base_name:
                    has_lat = True            
            elif 'Longitude' in base_name or 'Easting' in base_name:
                required_columns['X'] = True
                if 'Easting' not in base_name:                    
                    has_lon = True            
            elif 'Elevation' in base_name:
                required_columns['Z'] = True
            elif 'NoiseLevel' in base_name:
                required_columns['NoiseLevel'] = True
        else:
            unknown_columns.append(col)

    # Raise an error if any required columns are missing
    missing_columns = [col for col, present in required_columns.items() if not present]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    # Raise errors if input is ambiguous    
    if has_lon and not has_lat:
        raise ValueError("Ambiguous input: Latitude column must be present if Longitude columns is present.")
    if has_lat and not has_lon:
        raise ValueError("Ambiguous input: Longitude column must be present if Latitude columns is present.")
    # Decide for coordinate system
    if has_lon and has_lat:
        is_global = True
    else:
        is_global = False

    # Raise an error if there are unknown columns
    if unknown_columns:
        raise ValueError(f"Unknown columns found: {', '.join(unknown_columns)}")

    # Define data types for each column (float for all except 'Name' which is str)
    dtype_mapping = {col: str if 'Name' in col else float for col in columns_to_read}

    # Read the CSV file, using only the known columns
    df = pd.read_csv(file_path, delimiter=r"\s+", dtype=dtype_mapping, usecols=columns_to_read)

    # Create a mapping to rename columns
    column_mapping = {}
    for col, (base_name, _) in column_info.items():
        if 'Latitude' in base_name or 'Northing' in base_name:
            column_mapping[col] = 'Y'
        elif 'Longitude' in base_name or 'Easting' in base_name:
            column_mapping[col] = 'X'
        elif 'Elevation' in base_name:
            column_mapping[col] = 'Z'
        elif 'NoiseLevel' in base_name:
            column_mapping[col] = 'NoiseLevel'
        elif 'Name' in base_name:
            column_mapping[col] = 'Name'

    # Rename the columns
    df.rename(columns=column_mapping, inplace=True)

    # Process coordinate columns    
    coord_units = set()
    for col in ['Y', 'X', 'Z']:
        original_col = [k for k, v in column_mapping.items() if v == col][0]
        if col not in ['Y', 'X'] or not is_global:
            _, unit = column_info[original_col]
            if unit:
                coord_units.add(unit)
                # Convert to meters if necessary
                if unit == 'km':
                    df[col] *= 1000
                elif unit == 'cm':
                    df[col] *= 0.01
                elif unit == 'mm':
                    df[col] *= 0.001
                elif unit == 'ft':
                    df[col] *= 0.3048
                elif unit == 'in':
                    df[col] *= 0.0254

    # Transform coordinates
    if is_global:
        df['X'], df['Y'], refLat, refLon = global2local(df['Y'].to_numpy(), df['X'].to_numpy())
        df.attrs['reference_latitude'] = refLat
        df.attrs['reference_longitude'] = refLon

    # Check for consistency in coordinate units
    if len(coord_units) > 1:
        raise ValueError("Inconsistent units for coordinate columns")

    # Convert elevation to depth
    df['Z'] = -df['Z']

    # Process NoiseLevel column
    noise_col = [k for k, v in column_mapping.items() if v == 'NoiseLevel'][0]
    _, noise_unit = column_info[noise_col]

    # Determine noise type based on unit
    noise_types = {
        'm': 'displacement',
        'm/s': 'velocity',
        'm/s^2': 'acceleration',
        '1/s': 'strain rate',
        'none': 'strain',
        'strain': 'strain',
        '1': 'strain',
        'units': 'strain',
        'ε': 'strain'
    }

    if noise_unit in noise_types:
        noise_type = noise_types[noise_unit]
        if noise_type == 'strain':
            noise_unit = 'ε'
    else:
        raise ValueError(f"Unrecognized unit for NoiseLevel: {noise_unit}")
    
    # Edit DataFrame attributes:
    # Store original coordinate units in DataFrame attributes
    df.attrs['original_coord_units'] = list(coord_units)[0] if coord_units else None

    # Store coordainte sstem
    if is_global:
        df.attrs['coordinate_system'] = 'global'
    else:
        df.attrs['coordinate_system'] = 'local'

    # Store noise type and unit in DataFrame attributes
    df.attrs['noise_type'] = noise_type
    df.attrs['noise_unit'] = noise_unit

    # Add gauge_length to DataFrame attributes if found
    if gauge_length is not None:
        df.attrs['gauge_length'] = gauge_length

    # Add a special attribute to identify this as DataFrame that was not combined
    df.attrs['_is_combined'] = False

    return df

def read_velocity(file_path: str):
    r"""
    Read velocity model data from a whitespace-delimited CSV (ASCII) file and return a DataFrame.
    This function reads a CSV (ASCII) file containing velocity information, checks if the
    columns match the expected format, and renames the columns according to a predefined
    mapping for easier analysis.

    Parameters
    ----------
    file_path : :obj:`str`
        The path to the input CSV (ASCII) file containing the velocity data.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        A DataFrame containing the renamed columns:

        - 'Depth' for Depth;
        - 'Vp' for P-wave velocity;
        - 'VpVsRatio' for Vp/Vs ratio;
        - 'Qp' for P-wave quality factor;
        - 'Qs' for S-wave quality factor;
        - 'Rho' for density (kg/m³);
        - 'Ep' for anisotropy parameter Ep (Thomsen epsilon parameter for VTI medium, P-SV waves)
        - 'Dt' for anisotropy parameter Dt (Thomsen delta parameter for VTI medium, P-SV waves)
        - 'Gm' for anisotropy parameter Gm (Thomsen gamma parameter for VTI medium, SH waves)

        The DataFrame will also have the following attributes:

        - 'model_type' : :obj:`str`
            The type of model, either "homo" (homogeneous, single layer) or "layered" (multiple layers).
        - 'model_anisotropy' : :obj:`str`
            The anisotropy type, either "isotropic" (if all anisotropy parameters are zero) or "vti" (if any are non-zero).

    Raises
    ------
    ValueError
        If the input file columns do not match the expected format.

    Notes
    -----
    The function handles whitespace and tabs as delimiters.

    Examples
    --------
    >>> df = read_velocity("data/velocity.csv")
    >>> print(df.head())
    """
   
    # Read the file while handling spaces and tabs
    df = pd.read_csv(file_path, delimiter=r"\s+")

    # Define the expected column names
    expected_columns = [
        'Depth(m)', 'Vp(m/s)', 'Vp/Vs', 'Qp', 'Qs', 'density(kg/m^3)',
        'Ep(anisotropy)', 'Dt(anisotropy)', 'Gm(anisotropy)'
    ]

    # Check if the input file columns match the expected format
    if list(df.columns) != expected_columns:
        raise ValueError(
            "Input file columns do not match the expected format. "
            f"Expected: {expected_columns}, "
            f"Got: {list(df.columns)}"
        )

    # Define the column mapping
    column_mapping = {
        'Depth(m)': 'Depth',
        'Vp(m/s)': 'Vp',
        'Vp/Vs': 'VpVsRatio',
        'Qp': 'Qp',
        'Qs': 'Qs',
        'density(kg/m^3)': 'Rho',
        'Ep(anisotropy)': 'Ep',
        'Dt(anisotropy)': 'Dt',
        'Gm(anisotropy)': 'Gm'
    }

    # Rename columns as per the mapping
    df.rename(columns=column_mapping, inplace=True)

    # Determine the model type: "homo" if there is only one depth, "layered" if more
    if df['Depth'].nunique() == 1:
        model_type = "homo"
    else:
        #model_type = "layered"
        raise ValueError("Velocity model is not homogeneous!")

    # Determine the anisotropy type: "isotropic" if all anisotropy parameters are zero, "vti" otherwise
    if (df[['Ep', 'Dt', 'Gm']] == 0).all().all():
        model_anisotropy = "isotropic"
    else:
        model_anisotropy = "vti"

    # Store model_type and model_anisotropy in DataFrame attributes
    df.attrs['model_type'] = model_type
    df.attrs['model_anisotropy'] = model_anisotropy

    # Store description of parameters in DataFrame attributes
    parameter_descriptions = {
        'Depth': 'Depth',
        'Vp': 'P-wave velocity',
        'Vs': 'S-wave velocity',
        'VpVsRatio': 'Vp/Vs',
        'Qp': 'P-wave quality factor',
        'Qs': 'S-wave quality factor',
        'Rho': 'density',
        'Ep': 'Thomsen epsilon parameter',
        'Dt': 'Thomsen delta parameter',
        'Gm': 'Thomsen gamma parameter'
    }
    df.attrs['parameter_descriptions'] = parameter_descriptions

     # Store units of parameters in DataFrame attributes
    parameter_units = {
        'Depth': 'm',
        'Vp': 'm/s',
        'Vs': 'm/s',
        'VpVsRatio': '',
        'Qp': '',
        'Qs': '',
        'Rho': 'kg/m^3',
        'Ep': '',
        'Dt': '',
        'Gm': ''
    }
    df.attrs['parameter_units'] = parameter_units
    

    return df