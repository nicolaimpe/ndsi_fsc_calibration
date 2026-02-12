"""Few useful Python geospatial data treatement functions"""

import logging
from typing import Any, Dict

import geopandas as gpd
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from geospatial_grid.gsgrid import GSGrid
from pyproj import CRS, Transformer
from rasterio.features import rasterize

# Module configuration
default_logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


def gdf_to_binary_mask(gdf: gpd.GeoDataFrame, grid: GSGrid) -> xr.DataArray:
    """Convert a collection of polygons to a binary mask in xarray.

    Args:
        gdf (gpd.GeoDataFrame): a vectorial data file opened with geopandas. It must contain a polygon or a collection of polygons
        grid (GSGrid): the grid of the binary mask

    Returns:
        xr.DataArray: a binary mask
    """
    gdf = gdf.to_crs(grid.crs)
    transform = grid.affine

    # Prepare geometries for rasterization
    shapes = [(geom, 1) for geom in gdf.geometry]  # Assign a value of 1 to all polygons

    # Rasterize
    binary_mask = rasterize(
        shapes,
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,  # Background value
        dtype="uint8",
    )

    dims = ("y", "x")
    binary_mask_data_array = xr.DataArray(
        data=binary_mask,
        dims=(dims[0], dims[1]),
        coords={dims[0]: (dims[0], grid.ycoords), dims[1]: (dims[1], grid.xcoords)},
    )
    out = georef_netcdf_rioxarray(binary_mask_data_array, grid.crs)

    return out


def generate_xarray_compression_encodings(data: xr.Dataset | xr.DataArray, compression_level: int = 3) -> Dict[str, Any]:
    """Generate compression encodings to use with xarray to_netcdf.

    Args:
        data (xr.Dataset | xr.DataArray): an xarray object
        compression_level (int, optional): compression level. Defaults to 3.

    Returns:
        Dict[str, Any]: dictionary to use with kwarg encodings of xarray to_netcdf
    """
    output_dict = {}
    compression_encoding_dict = {"zlib": True, "complevel": compression_level}
    if type(data) is xr.Dataset:
        for data_var_name in data.data_vars:
            output_dict.update({data_var_name: compression_encoding_dict})
    elif type(data) is xr.DataArray:
        output_dict.update({data.name: compression_encoding_dict})
    return output_dict


def find_aoi_bounds(aoi_vectorial_file: str):
    """Find a bounding box given a vectorial file."""
    gdf = gpd.read_file(aoi_vectorial_file)
    transformer = Transformer.from_crs(crs_from=gdf.crs, crs_to=CRS.from_epsg(4326), always_xy=True)
    return transformer.transform_bounds(*gdf.union_all().bounds)
