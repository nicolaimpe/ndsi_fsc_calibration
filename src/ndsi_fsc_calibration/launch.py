import argparse
import logging
import sys
from datetime import datetime
from glob import glob

import earthaccess
import geopandas as gpd
import xarray as xr
import yaml
from geospatial_grid.grid_database import PROJ4_MODIS
from geospatial_grid.gsgrid import GSGrid
from matplotlib import pyplot as plt
from pyproj import CRS, Transformer

from ndsi_fsc_calibration.download import download_s2_fsc_pyhydroweb
from ndsi_fsc_calibration.match import Scatter
from ndsi_fsc_calibration.regrid import S2TheiaRegrid, V10Regrid
from ndsi_fsc_calibration.utils import find_aoi_bounds, gdf_to_binary_mask
from ndsi_fsc_calibration.visualization import scatter_plot_with_fit

logger = logging.getLogger()


def parse_arguments(args):
    """Parsing the arguments when you call the main program."""
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional argument
    help_date_begin = "Start date for the data collection in the format YYYYMMDD"
    help_date_end = "End date for the data collection in the format YYYYMMDD"
    help_aoi = "Area of interest file in a vectorial format (everything accepted by geopandas read_file)"
    help_eval_product = "Identifier of evaluation snow cover product to use. Available options: "
    help_s2_dir = (
        "Path to the Sentinel-2 reference data folder. Add --download_s2 option to trigger download from Hydroweb API"
    )
    help_eval_prod_dir = (
        "Path to the VIIRS/MODIS product data folder. Add --download_nasa option to trigger download from earthcaccess API"
    )
    help_output_dir = "Path to the output folder.VNP10A1, VJ110A1, VJ210A1, MOD10A1 Intermediary results will also be exported to this folder"
    help_resampling_grid_file = "If the correspondences shall be calculated on a user defined grid, select the corresponding grid configuration file (see conf/grid_conf_example.yaml)"

    parser.add_argument("-sd", "--start_date", help=help_date_begin, type=str, default=None)
    parser.add_argument("-ed", "--end_date", help=help_date_end, type=str, default=None)
    parser.add_argument("-aoi", "--aoi_file", help=help_aoi, type=str, default=None)
    parser.add_argument("-pn", "--product_name", help=help_eval_product, type=str, default="VNP10A1")
    parser.add_argument("-s2f", "--sentinel_2_folder", help=help_s2_dir, type=str, default=None)
    parser.add_argument("-pf", "--nasa_folder", help=help_eval_prod_dir, type=str, default=None)
    parser.add_argument("-of", "--output_folder", help=help_output_dir, type=str, default="./output_folder")
    parser.add_argument("-rg", "--resampling_grid_file", help=help_resampling_grid_file, type=str, default=None)
    parser.add_argument("--downlaod_s2", help=help_output_dir, action="store_true")
    parser.add_argument("--download_nasa", help=help_resampling_grid_file, action="store_true")

    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    date_start = datetime.strftime(args.start_date, format="YYYYMMdd")
    date_end = datetime.strftime(args.end_date, format="YYYYMMdd")
    aoi_bounds = find_aoi_bounds(args.aoi_file)
    prod_id = args.product_name

    if prod_id not in ("VNP10A1", "VJ110A1", "VJ210A1", "MOD10A1"):
        raise NotImplementedError(f"Unknkown product name {prod_id}")

    if args.download_nasa:
        logger.info(f"Download data for {args.product_name} to {args.nasa_folder}/{args.product_name} via NASA earthaccess")
        nasa_products = earthaccess.search_data(
            short_name=args.product_name,  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10?
            bounding_box=aoi_bounds,  # Only include files in area of interest...
            temporal=(date_start, date_end),  # ...and time period of interest
            day_night_flag="day",
        )

        files = earthaccess.download(nasa_products, f"{args.nasa_folder}/{args.product_name}")

    if args.download_s2:
        logger.info(f"Download data for Sentinel-2 snow cover fraction to {args.sentinel_2_folder} via PyHydroweb API")
        download_s2_fsc_pyhydroweb(
            start_date=date_start, end_date=date_end, bounding_box=aoi_bounds, download_folder=args.sentinel_2_folder
        )

    if args.resampling_grid_file:
        logger.info(f"Creating a resampling grid with the user input grid configuration file {args.resampling_grid_file}")
        with open(args.resampling_grid_file, "r") as src:
            grid_kwargs = yaml.safe_load(src)
            output_grid = GSGrid(**grid_kwargs)
    else:
        logger.info(f"Creating a default resampling grid on {prod_id} SIN Grid using area of interest file.")
        format = "hdf" if args.product_name == "MOD10A1" else "h5"
        eval_prod_sample = glob.glob(f"{args.nasa_folder}/{args.product_name}/*.{format}")[0]
        resolution = xr.open_dataset(eval_prod_sample, engine="netcdf4").attrs["CharacteristicBinSize"]
        transformer = Transformer.from_crs(crs_from=CRS.from_epsg(43624), crs_to=CRS.from_proj4(PROJ4_MODIS), always_xy=True)
        aoi_bounds_sin_grid = transformer.transform_bounds(*aoi_bounds)

        output_grid = GSGrid(
            x0=aoi_bounds_sin_grid[0] // resolution,
            y0=aoi_bounds_sin_grid[3] // resolution + resolution,
            resolution=(resolution, resolution),
            width=(aoi_bounds_sin_grid[2] - aoi_bounds_sin_grid[0]) / resolution,
            height=(aoi_bounds_sin_grid[3] - aoi_bounds_sin_grid[1]) / resolution,
            crs=CRS.from_proj4(PROJ4_MODIS),
        )

    vnp10a1_regridder = V10Regrid(
        output_grid=output_grid,
        data_folder=f"{args.nasa_folder}/{args.product_name}",
        output_folder=f"output_folder/{prod_id.lower()}",
    )
    s2_regridder = S2TheiaRegrid(
        output_grid=output_grid, data_folder=f"{args.sentinel_2_folder}", output_folder="output_folder/s2"
    )

    logger.info(f"Regridding {prod_id} evaluation dataset on output grid.")
    vnp10a1_regridder.create_time_series(
        roi_shapefile=args.aoi_file,
        start_date=date_start,
        end_date=date_end,
    )
    logger.info("Regridding Sentinel-2 reference data on output grid.")
    s2_regridder.create_time_series(
        roi_shapefile=args.aoi_file,
        start_date=date_start,
        end_date=date_end,
    )

    vnp10a1_regridded = xr.open_dataset(f"{args.output_folder}/{prod_id.lower()}/regridded.nc")
    s2_regridded = xr.open_dataset(f"{args.output_folder}/s2/regridded.nc")

    logger.info(f"Compute individual correspondences between {prod_id} evaluation dataset and Sentinel-2 reference dataset")
    matcher = Scatter(eval_product=prod_id, ref_product="S2")
    matcher.compute_all_correspondences(
        eval_ndsi_time_series=vnp10a1_regridded.data_vars["NDSI_Snow_Cover"],
        ref_fsc_time_series=s2_regridded.data_vars["snow_cover_fraction"],
        netcdf_export_path=f"{args.output_folder}/correspondences.nc",
    )

    logger.info("Creating resuming scatter plot with linear regression fit.")
    correspondence_dataset = xr.open_dataset(f"{args.output_folder}/correspondences.nc")
    fig, ax = plt.subplots()

    fig, ax = scatter_plot_with_fit(
        data_to_plt=correspondence_dataset.sum(dim="time").data_vars["n_occurrences"],
        fig=fig,
        ax=ax,
        eval_prod_name=prod_id,
        quantile_max=0.9,
        quantile_min=0.2,
    )

    fig.patch.set_alpha(0.0)
    logger.info(f"Exporting to {args.output_folder}/scatter_plot.png")
    fig.savefig(f"{args.output_folder}/scatter_plot.png", format="png", dpi=600)
    plt.show()
