import re

import numpy as np
import pyproj
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from geospatial_grid.grid_database import PROJ4_MODIS
from geospatial_grid.gsgrid import GSGrid
from pyhdf.SD import SD, SDC

# NASA snow cover products meaning-value correspondences(see VNP10A1 product documentation)
NASA_CLASSES = {
    "snow_cover": range(1, 101),
    "no_snow": (0,),
    "clouds": (250,),
    "water": (237, 239),  # inland and ocean
    "no_decision": (201,),
    "night": (211,),
    "missing_data": (251,),
    "L1B_unusable": (252,),
    "bowtie_trim": (253,),
    "L1B_fill": (254,),
    "fill": (255,),
}

NODATA_NASA_CLASSES = (
    "no_decision",
    "night",
    "missing_data",
    "L1B_unusable",
    "bowtie_trim",
    "L1B_fill",
    "fill",
)

S2_CLASSES = {"snow_cover": range(1, 101), "no_snow": (0,), "clouds": (205,), "nodata": (255,), "fill": (255,)}


def get_modis_bin_size(filepath: str) -> float:
    """Find MOD10A1 resolution from product metadata"""
    hdf = SD(filepath, SDC.READ)
    fattrs = hdf.attributes(full=1)
    ga = fattrs["ArchiveMetadata.0"]
    archivemeta = ga[0]
    pattern = r"OBJECT\s*=\s*CHARACTERISTICBINSIZE.*?VALUE\s*=\s*([0-9.+-Ee]+)"

    match = re.search(pattern, archivemeta, re.DOTALL)
    bin_size = float(match.group(1))
    return bin_size


def open_modis_ndsi_snow_cover(filepath: str) -> xr.DataArray:
    """Open a MOD10A1 hdf on a georeferenced xarray. Georeferencing data are read from metadata."""
    DATAFIELD_NAME = "NDSI_Snow_Cover"

    hdf = SD(filepath, SDC.READ)

    # Read dataset.
    data2D = hdf.select(DATAFIELD_NAME)
    data = data2D[:, :].astype(np.float64)

    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]

    ul_regex = re.compile(
        r"""UpperLeftPointMtrs=\(
    (?P<upper_left_x>[+-]?\d+\.\d+)
    ,
    (?P<upper_left_y>[+-]?\d+\.\d+)
    \)""",
        re.VERBOSE,
    )
    bin_size = get_modis_bin_size(filepath=filepath)
    match = ul_regex.search(gridmeta)
    x0 = float(match.group("upper_left_x"))
    y0 = float(match.group("upper_left_y"))

    ny, nx = data.shape
    prod_coordinates = GSGrid(x0=x0, y0=y0, resolution=bin_size, width=nx, height=ny).xarray_coords
    out_data_array = xr.DataArray(data=data, coords=prod_coordinates)
    return georef_netcdf_rioxarray(xr.Dataset({DATAFIELD_NAME: out_data_array}), crs=pyproj.CRS.from_proj4(PROJ4_MODIS))
