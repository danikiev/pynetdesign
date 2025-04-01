from pynetdesign.modelling.io import *
from pynetdesign.visualization.utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import cmcrameri.cm as cmc
import pandas as pd
import numpy as np
import warnings

def plot_stations_view(df: pd.DataFrame,
                       view: str = 'xy',
                       x_offset: float = None,
                       y_offset: float = None,
                       z_offset: float = None,
                       offset_perc: float = 0.1,
                       plot_names: bool = True,
                       station_marker_size: int = 100,                       
                       grid_points: np.ndarray = None,
                       grid_marker_size: int = 1,
                       use_warnings: bool = True): 
    r"""
    Plots either a map view or side view of stations as filled triangles with name labels.
    Optionally plots grid points as well. Color-codes the triangles based on noise level
    and adds a colorbar.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Data frame containing station names, coordinates, and noise levels.
        Expected columns: 'Name', 'X', 'Y', 'Z' (for side view), 'NoiseLevel'.
    view : :obj:`str`, optional
        Specifies the view to plot. Options are 'xy' (X-Y), 'xz' (X-Z), or 'yz' (Y-Z).
        Default is 'xy'.
    x_offset : :obj:`float`, optional
        Offset added to the min and max X coordinates for axis limits.
        If None, defaults to 10% of the X coordinate range.
    y_offset : :obj:`float`, optional
        Offset added to the min and max Y coordinates for axis limits.
        If None, defaults to 10% of the Y coordinate range.
    z_offset : :obj:`float`, optional
        Offset added to the min and max Z coordinates for axis limits (side view only).
        If None, defaults to 10% of the Z coordinate range.
    offset_perc : :obj:`float`, optional, default: 0.1
        Percentage of coordinate range to use as offset.
        Used only if offset value is None.
    plot_names : :obj:`bool`, optional, default: True
        Flag to plot station names or not (True or False)
    station_marker_size : :obj:`int`, optional, default: 100
        Size of the station markers in the scatter plot    
    grid_points: :obj:`np.ndarray`, optional, default: None
        Array of shape (ngrid, 3) representing grid points' coordinates (X, Y, Z).
    grid_marker_size: :obj:`int`, optional, default: 10
        Size of the grid points markers in the scatter plot
    use_warnings: :obj:`bool`, optional, default: True
        If True, enable warnings, if False, disable them.

    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        The figure object containing the plot.
    ax : :obj:`matplotlib.axes.Axes`
        The axes object of the plot.
    cax : :obj:`matplotlib.axes.Axes`
        The axes object for the colorbar.
    cbar : :obj:`matplotlib.colorbar.Colorbar`
        The colorbar object.
    """
    # Validate view parameter
    if view not in ['xy', 'xz', 'yz']:
        raise ValueError("Invalid view specified. Choose 'xy', 'xz', or 'yz'.")

    # Check if required columns exist
    required_columns = ['X', 'Y', 'NoiseLevel']
    if view in ['xz', 'yz']:
        required_columns.append('Z')
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The '{col}' column is missing in the geometry data.")

    # Extract data from the dataframe
    x = df['X']
    y = df['Y']
    if view in ['xz', 'yz']:
        z = df['Z']

    # Check if noise level units are consistent
    if 'noise_unit' in df.attrs:
        noise_level = df['NoiseLevel'].to_numpy()
        clb_label = f"Noise Level ({df.attrs['noise_unit']})"
        plot_clb = True
    else:
        noise_level = np.full(len(x), 0)
        plot_clb = False    

    # Check if Name column can be used
    if plot_names:
        if 'Name' in df.columns:
            names = df['Name']
        else:
            plot_names = False
            if use_warnings:
                warnings.warn("Can't plot station names: Name column is missing in the geometry data.")

    if grid_points is not None:
        # Grid points are expected to be of shape (ngrid, 3), where each row is (x, y, z)
        gx = grid_points[:, 0]
        gy = grid_points[:, 1]
        gz = grid_points[:, 2]
        # Get limits from stations and grid points
        xmin = min([x.min(), gx.min()])
        xmax = max([x.max(), gx.max()])
        ymin = min([y.min(), gy.min()])
        ymax = max([y.max(), gy.max()])
        if view in ['xz', 'yz']:
            zmin = min([z.min(), gz.min()])
            zmax = max([z.max(), gz.max()])
    else:
        # Get limits only from stations
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        if view in ['xz', 'yz']:
            zmin = z.min()
            zmax = z.max()

    # Calculate offsets
    if x_offset is None:
        x_offset = offset_perc * (xmax - xmin)
    if y_offset is None:
        y_offset = offset_perc * (ymax - ymin)
    if view in ['xz', 'yz'] and z_offset is None:
        z_offset = offset_perc * (zmax - zmin)
        if z_offset < 0.001 * (xmax - xmin + ymax - ymin):
            z_offset = offset_perc * min(xmax - xmin, ymax - ymin)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Ensure equal scaling
    ax.set_aspect('equal', 'box')

    # First plot grid points if provided
    if grid_points is not None:        
        if view == 'xy':
            ax.scatter(gx, gy, color='gray', s=grid_marker_size, marker='s', zorder=1, label='Grid Points')
        elif view == 'xz':
            ax.scatter(gx, gz, color='gray', s=grid_marker_size, marker='s', zorder=1, label='Grid Points')
        else:  # view == 'yz'
            ax.scatter(gy, gz, color='gray', s=grid_marker_size, marker='s', zorder=1, label='Grid Points')

    # Plot based on the view
    if view == 'xy':
        scatter = ax.scatter(x, y, c=noise_level, cmap='viridis', s=station_marker_size, marker='^', zorder=2)
        ax.set_xlim(xmin - x_offset, xmax + x_offset)
        ax.set_ylim(ymin - y_offset, ymax + y_offset)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Map View of Stations')
    elif view == 'xz':
        scatter = ax.scatter(x, z, c=noise_level, cmap='viridis', s=station_marker_size, marker='^', zorder=2)
        ax.set_xlim(xmin - x_offset, xmax + x_offset)
        ax.set_ylim(zmax + z_offset, zmin - z_offset)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Z Coordinate')
        ax.set_title('Side View of Stations: X-Z')
    else:  # view == 'yz'
        scatter = ax.scatter(y, z, c=noise_level, cmap='viridis', s=station_marker_size, marker='^', zorder=2)
        ax.set_xlim(ymin - y_offset, ymax + y_offset)
        ax.set_ylim(zmax + z_offset, zmin - z_offset)
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('Z Coordinate')
        ax.set_title('Side View of Stations: Y-Z')
   
    # Add labels to the points
    if plot_names:
        for i, name in enumerate(names):
            name = name + " "  # to make a gap between the name and the marker
            if view == 'xy':
                ax.text(x[i], y[i], name, fontsize=12, ha='right', va='bottom', zorder=3)
            elif view == 'xz':
                ax.text(x[i], z[i], name, fontsize=12, ha='right', va='bottom', zorder=3)
            else:  # view == 'yz'
                ax.text(y[i], z[i], name, fontsize=12, ha='right', va='bottom', zorder=3)

    # Add grid
    ax.grid(True)

    if plot_clb:
        # Create colorbar    
        if view in ['xz', 'yz']:
            bbox = ax.get_position()
            cax_size_perc = 5 * bbox.width / bbox.height
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size=f"{cax_size_perc}%", pad=0.65)
            cbar = plt.colorbar(scatter, cax=cax, orientation="horizontal")
            cbar.set_label(clb_label, labelpad=15)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label(clb_label, rotation=270, labelpad=15)
    else:
        cax = None
        cbar = None

    plt.show()

    return fig, ax, cax, cbar
