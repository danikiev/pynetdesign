import matplotlib.colors as mcolors
import pandas as pd

def generate_colorscale(cmap: mcolors.Colormap, ncolors: int) -> list:
    r"""
    Generate a colorscale from a colormap and number of colors.

    Parameters
    ----------
    cmap : :obj:`mcolors.Colormap`
        A colormap instance from matplotlib.colors to be used for generating the colorscale.
    ncolors : :obj:`int`
        The number of colors to generate in the colorscale. Must be a positive integer.

    Returns
    -------
    colorscale : :obj:`list`
        A list of colors in the format [[position, 'rgb(r, g, b)'], ...] where `position` is a 
        float between 0 and 1 representing the color position, and 'rgb(r, g, b)' is the color 
        at that position in RGB format.

    Raises
    ------
    ValueError
        If `ncolors` is less than 2 or if `cmap` is not an instance of mcolors.Colormap.

    Examples
    --------
    >>> from matplotlib import cm
    >>> generate_colorscale(cm.viridis, 5)    
    [[0.0, 'rgb(0.267004, 0.004874, 0.329415)'], [0.25, 'rgb(0.229739, 0.322361, 0.545706)'], 
     [0.5, 'rgb(0.127568, 0.566949, 0.550556)'], [0.75, 'rgb(0.369214, 0.788888, 0.382914)'], 
     [1.0, 'rgb(0.993248, 0.906157, 0.143936)']]
    [[0.0, 'rgb(68, 1, 84)'], [0.25, 'rgb(58, 82, 139)'], [0.5, 'rgb(32, 144, 140)'],
     [0.75, 'rgb(94, 201, 97)'], [1.0, 'rgb(253, 231, 36)']]
    """
    # Input validation
    if not isinstance(cmap, mcolors.Colormap):
        raise ValueError("cmap must be an instance of matplotlib.colors.Colormap.")
    if not isinstance(ncolors, int) or ncolors < 2:
        raise ValueError("ncolors must be an integer greater than or equal to 2.")

    # Generate colorscale
    colorscale = [
        [i / (ncolors - 1), 'rgb({}, {}, {})'.format(*(int(255 * v) for v in cmap(i / (ncolors - 1))[:3]))]
        for i in range(ncolors)
    ]
    
    return colorscale

def is_notebook() -> bool:
    r"""
    Check if code is run in a notebook.

    Returns
    -------
    :obj:`bool` 
        True if yes (Jupyter notebook, qtconsole, google colab) or False if not
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'google.colab._shell':
            return True   # Google colab
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def check_spyder():
    r"""
    Check if code is run inside Spyder.

    Returns
    -------
    is_spyder : :obj:`bool` 
        True if yes or False if not
    """
    try: # fails when not ipython
        ip_name = get_ipython().__class__.__name__
    except: ip_name = -1
    is_spyder = ip_name == 'SpyderShell'
    return is_spyder