# PyNetDesign

[![DOI](https://zenodo.org/badge/958300487.svg)](https://zenodo.org/badge/latestdoi/958300487)

PyNetDesign is an open-source Python framework for testing and designing microseismic monitoring networks for homogeneous velocity models.

PyNetDesign can determine the magnitude sensitivity of the microseismic monitoring network, i.e. estimate the minimum moment magnitude detectable using the given network or receivers.
It can work with both station networks and distributed acoustic sensing (DAS) cable networks at the same time.
Magnitude sensitivity is represented in the form of horizontal or vertical slices through a 3D grid.

## Installation

### Requirements

To clone the git repository use [Git](https://git-scm.com/), a free and open source distributed version control system.

Installation requires [Conda](https://conda.io) package manager, e.g. one can use [miniforge](https://github.com/conda-forge/miniforge) implementation.

PyNetDesign works with 3.8 <= `python` <= 3.12 and requires [`pip`](https://pypi.org/project/pip/).

Some key dependencies:

- [`numpy`](https://www.numpy.org/)
- [`scipy`](https://scipy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`plotly`](https://plotly.com/python/)

The author is incredibly grateful to the developers of these and other packages used by PyNetDesign.

### Clone

First clone the git repository using

```sh
git clone https://github.com/danikiev/pynetdesign.git
```

### Install

The best way is to create a new conda environment with all the required packages:

```bash
conda env create -f environment.yml
```

**Note:** to speed up creation of the environment, use `mamba` instead of `conda`, which is a faster alternative.

Then activate the newly created environment:

```bash
conda activate pnd
```

Finally, install the package:

```bash
pip install -e .
```

### Installation using script

For quick installation, you can use the specially designed installation script which implement all of the above mentioned steps.

On Windows, in miniforge prompt run:

```cmd
install.bat
```

and select option 1 (user environment).

### Installation for development

Development environment includes additional packages for testing and building documentation.

For development please use `environment-dev.yml` instead of `environment.yml`:

```bash
conda env create -f environment-dev.yml
conda activate pnd-dev
pip install -e .
```

On Windows you can use `install.bat` and select option 2 (developer environment):

```cmd
install.bat
```

Developer environment includes more packages, e.g. for building [local documentation](#build-documentation-locally),

### Uninstall

If you need to add/change packages, deactivate the environment first:

```bash
conda deactivate
```

and then remove the appropriate environment:

```bash
conda remove -n pnd --all
```

or

```bash
conda remove -n pnd-dev --all
```

On Windows you can also run the uninstallation script:

```cmd
uninstall.bat
```

It will search for all associated Conda environments matching `pnd*` and will ask to delete each of them.

## Documentation

The latest stable documentation based on [Sphinx](https://www.sphinx-doc.org) is available online at: <>.

It features:

- Methodology
- Getting started guide
- Examples
- API reference

### Build documentation locally

Packages required for building of the documentation are included to the development environment.
To install it in the normal installation environment you have to additionally run:

```cmd
conda install sphinx pydata-sphinx-theme sphinx-gallery numpydoc
```

and

```cmd
pip install setuptools_scm
```

To build the documentation locally run:

```bash
cd docs
make html
```

If you want to rebuild the documentation:

```bash
cd docs
make clean
make html
```

To build website also the PDF file of the website, use:

```bash
make latexpdf
```

After a successful build, one can serve the documentation website locally:

```bash
cd build/html
python -m http.server
```

and open in browser: <http://localhost:8000>.

**Note:** check the exact port number in the output

On Windows you can also use the script:

```cmd
build-docs.bat
```

or, to build also the PDF file:

```cmd
build-docs.bat -pdf
```
