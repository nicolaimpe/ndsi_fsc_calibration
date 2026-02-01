import abc
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from geospatial_grid.grid_database import PROJ4_MODIS
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from rasterio.enums import Resampling

from ndsi_fsc_calibration.snow_cover_products import NASA_CLASSES, S2_CLASSES
from ndsi_fsc_calibration.utils import default_logger as logger
from ndsi_fsc_calibration.utils import gdf_to_binary_mask, generate_xarray_compression_encodings


def resample_s2_to_grid(s2_dataset: xr.Dataset, output_grid: GSGrid) -> xr.DataArray:
    # 250m resolution FSC from FSCOG S2 product with a "zombie" nodata mask

    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    s2_validity_mask = reproject_using_grid(
        s2_dataset, output_grid=output_grid, resampling_method=Resampling.max, nodata=S2_CLASSES["nodata"][0]
    )

    # Aggregate the dataset at 250 m
    s2_aggregated = reproject_using_grid(
        s2_dataset.astype(np.float32),
        output_grid=output_grid,
        resampling_method=Resampling.average,
        nodata=S2_CLASSES["nodata"][0],
    )

    # Compose the mask
    s2_out_image = xr.where(s2_validity_mask <= S2_CLASSES["snow_cover"][-1], s2_aggregated.astype("u1"), s2_validity_mask)
    s2_out_image.rio.write_nodata(S2_CLASSES["fill"][0], inplace=True)

    return s2_out_image


def reprojection_l3_nasa_to_grid(nasa_snow_cover: xr.DataArray, output_grid: GSGrid) -> xr.DataArray:
    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    # nasa_dataset = nasa_dataset.where(nasa_dataset <= NASA_CLASSES["snow_cover"][-1], NASA_CLASSES["fill"][0])

    resampled_max = reproject_using_grid(
        nasa_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.max,
        nodata=NASA_CLASSES["fill"][0],
    )

    resampled_average = reproject_using_grid(
        nasa_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.average,
    )

    resampled_nearest = reproject_using_grid(
        nasa_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.nearest,
    )

    invalid_mask = resampled_max > NASA_CLASSES["snow_cover"][-1]
    water_mask = resampled_nearest == NASA_CLASSES["water"][0] | NASA_CLASSES["water"][1]
    valid_qualitative_mask = water_mask

    out_snow_cover = resampled_average.where(invalid_mask == False, resampled_max)
    # We readd water resempled with nearest
    out_snow_cover = out_snow_cover.where(valid_qualitative_mask == False, resampled_nearest)

    return out_snow_cover.astype("u1")


class RegridBase:
    def __init__(self, data_folder: str, output_folder: str, output_grid: GSGrid, product_classes: Dict[str, int | range]):
        self.grid = output_grid
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.product_classes = product_classes

    @abc.abstractmethod
    def get_all_files(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get_date_files(self, all_winter_year_files: List[str], date: datetime) -> List[str]:
        pass

    @abc.abstractmethod
    def check_date_files(self, date_files: List[str]) -> List[str]:
        pass

    @abc.abstractmethod
    def create_spatial_composite(self, date_files: List[str]) -> xr.Dataset:
        pass

    def scf_empty(self, daily_composite: xr.Dataset) -> None:
        snow_cover = (
            daily_composite.data_vars["snow_cover_fraction"]
            if "snow_cover_fraction" in daily_composite.data_vars
            else daily_composite.data_vars["NDSI_Snow_Cover"]
        )
        if (
            snow_cover.where(snow_cover <= self.product_classes["clouds"]).count()
            == snow_cover.where(snow_cover == self.product_classes["clouds"]).count()
        ):
            return True
        else:
            return False

    def low_values_screen(self, daily_composite: xr.Dataset, thresholds: Dict[str, float]) -> xr.Dataset:
        for key, value in thresholds.items():
            daily_composite.data_vars[key][:] = daily_composite.data_vars[key].where(daily_composite.data_vars[key] > value, 0)
        return daily_composite

    def export_date_data(self, date: datetime, date_data: xr.Dataset):
        out_path = f"{str(self.output_folder)}/{date.strftime('%Y%m%d')}.nc"
        daily_composite = date_data.assign_coords(dict(time=[date]))
        daily_composite.to_netcdf(out_path)

    def export_time_series(self):
        out_tmp_paths = glob(f"{str(self.output_folder)}/*.nc")
        time_series = xr.open_mfdataset(out_tmp_paths, mask_and_scale=False)
        encodings = generate_xarray_compression_encodings(time_series)
        out_path = f"{self.output_folder}/regridded.nc"
        logger.info(f"Exporting to {out_path}")
        time_series.to_netcdf(out_path, encoding=encodings)
        [os.remove(file) for file in out_tmp_paths]

    def create_time_series(
        self,
        roi_shapefile: str,
        start_date: datetime,
        end_date: datetime,
        low_value_thresholds: Dict[str, float] | None = None,
    ):
        files = self.get_all_files()
        period = pd.date_range(start=start_date, end=end_date)
        for date in period:
            logger.info(f"Processing date {date}")

            day_files = self.get_date_files(files, date=date)

            day_files = self.check_date_files(date_files=day_files)

            if len(day_files) == 0:
                logger.info(f"Skip day {date} because 0 files were found on this date")
                continue
            daily_composite = self.create_spatial_composite(date_files=day_files)
            if roi_shapefile is not None:
                roi_mask = gdf_to_binary_mask(gdf=gpd.read_file(roi_shapefile), grid=self.grid)

                # Handle numerical overflow grid misalignment
                try:
                    daily_composite = daily_composite.where(roi_mask, self.product_classes["fill"][0])
                except xr.structure.alignment.AlignmentError:
                    logger.info("Misalignment between snow cover map and ROI mask. Reproject ROI mask on output_grid.")
                    roi_mask = reproject_using_grid(roi_mask, output_grid=self.grid, resampling_method=Resampling.nearest)
                    daily_composite = daily_composite.where(roi_mask, self.product_classes["fill"][0])

                for dv in daily_composite.data_vars.values():
                    dv.rio.write_nodata(self.product_classes["fill"][0], inplace=True)
            daily_composite.to_netcdf("test_ndsi_snoc_cvoer.nc")
            if self.scf_empty(daily_composite):
                logger.info(f"Skip day {date} because only clouds are present on this date.")
                continue
            if low_value_thresholds is not None:
                daily_composite = self.low_values_screen(daily_composite=daily_composite, thresholds=low_value_thresholds)

            self.export_date_data(date=date, date_data=daily_composite)
        self.export_time_series()


class S2Regrid(RegridBase):
    def __init__(self, data_folder: str, output_folder: str, output_grid: GSGrid):
        super().__init__(
            output_grid=output_grid, data_folder=data_folder, output_folder=output_folder, product_classes=S2_CLASSES
        )

    def check_date_files(self, date_files: List[str]) -> List[str]:
        for day_file in date_files:
            try:
                xr.open_dataset(day_file).data_vars["band_data"].values
            except (OSError, rasterio.errors.RasterioIOError, rasterio._err.CPLE_AppDefinedError):
                logger.info(f"Could not open file {day_file}. Removing it from processing")
                date_files.remove(day_file)
                continue
        return date_files

    def get_date_files(self, all_winter_year_files: List[str], date: datetime):
        return [file for file in all_winter_year_files if date.strftime("%Y%m%d") in file]


class S2TheiaRegrid(S2Regrid):
    def __init__(self, output_grid: GSGrid, data_folder: str, output_folder: str, fsc_thresh: int = 51):
        super().__init__(output_grid=output_grid, data_folder=data_folder, output_folder=output_folder)
        self.fsc_thresh = fsc_thresh

    def get_all_files(self) -> List[str]:
        return glob(str(Path(self.data_folder).joinpath("LIS_S2-SNOW-FSC_*tif")))

    def create_spatial_composite(self, date_files: List[str]) -> xr.Dataset:
        day_data_array = xr.DataArray(S2_CLASSES["nodata"][0], coords=self.grid.xarray_coords).astype("u1")
        for filepath in date_files:
            logger.info(f"Processing product {Path(filepath).name}")
            s2_image = rioxarray.open_rasterio(filepath)
            s2_image = s2_image.sel(band=1).drop_vars("band")
            high_fsc_mask = (s2_image >= S2_CLASSES["snow_cover"][0]) * (s2_image < self.fsc_thresh)
            low_fsc_mask = (s2_image >= self.fsc_thresh) * (s2_image <= S2_CLASSES["snow_cover"][-1])
            s2_image = s2_image.where(1 - high_fsc_mask, 100)
            s2_image = s2_image.where(1 - low_fsc_mask, 0)
            s2_resampled_image = resample_s2_to_grid(s2_dataset=s2_image, output_grid=self.grid)
            day_data_array = day_data_array.where(day_data_array != S2_CLASSES["nodata"][0], s2_resampled_image.values)
        day_dataset = xr.Dataset({"snow_cover_fraction": day_data_array})
        return georef_netcdf_rioxarray(day_dataset, crs=self.grid.crs)


class V10Regrid(RegridBase):
    def __init__(self, output_grid: GSGrid, data_folder: str, output_folder: str):
        super().__init__(
            output_grid=output_grid, data_folder=data_folder, output_folder=output_folder, product_classes=NASA_CLASSES
        )

    def get_all_files(self) -> List[str]:
        return glob(str(Path(self.data_folder).joinpath("V*10A1*.h5")))

    def get_date_files(self, all_winter_year_files: List[str], date: datetime) -> List[str]:
        return [file for file in all_winter_year_files if date.strftime("A%Y%j") in file]

    def check_date_files(self, date_files: List[str]) -> List[str]:
        for date_file in date_files:
            try:
                xr.open_dataset(date_file, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4").data_vars[
                    "NDSI_Snow_Cover"
                ].values
            except OSError:
                logger.info(f"Could not open file {date_file}. Removing it from processing")
                date_files.remove(date_file)
                continue
        return date_files

    def create_spatial_l3_nasa_viirs_composite(self, daily_snow_cover_files: List[str]) -> xr.DataArray:
        day_data_arrays = []
        dims = ("y", "x")
        for filepath in daily_snow_cover_files:
            # try:
            logger.info(f"Processing product {Path(filepath).name}")

            product_grid_data_variable = xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D", engine="netcdf4")
            bin_size = xr.open_dataset(filepath, engine="netcdf4").attrs["CharacteristicBinSize"]
            nasa_l3_grid = GSGrid(
                resolution=bin_size,
                x0=product_grid_data_variable.coords["XDim"][0].values,
                y0=product_grid_data_variable.coords["YDim"][0].values,
                width=len(product_grid_data_variable.coords["XDim"]),
                height=len(product_grid_data_variable.coords["YDim"]),
            )
            ndsi_snow_cover = xr.open_dataset(
                filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4"
            ).data_vars["NDSI_Snow_Cover"]

            ndsi_snow_cover = ndsi_snow_cover.rename({"XDim": dims[1], "YDim": dims[0]}).assign_coords(
                coords={dims[0]: nasa_l3_grid.ycoords, dims[1]: nasa_l3_grid.xcoords}
            )

            day_data_arrays.append(georef_netcdf_rioxarray(data_array=ndsi_snow_cover, crs=pyproj.CRS.from_proj4(PROJ4_MODIS)))

        merged_day_dataset = (
            xr.combine_by_coords(day_data_arrays, data_vars="minimal", fill_value=NASA_CLASSES["fill"][0])
            .astype(np.uint8)
            .data_vars["NDSI_Snow_Cover"]
        ).rio.write_nodata(NASA_CLASSES["fill"][0])

        return georef_netcdf_rioxarray(data_array=merged_day_dataset, crs=pyproj.CRS.from_proj4(PROJ4_MODIS))

    def create_spatial_composite(self, date_files: List[str]) -> xr.Dataset:
        daily_spatial_composite = self.create_spatial_l3_nasa_viirs_composite(daily_snow_cover_files=date_files)
        nasa_snow_cover = reprojection_l3_nasa_to_grid(nasa_snow_cover=daily_spatial_composite, output_grid=self.grid)
        nasa_snow_cover.attrs.pop("valid_range")
        out_dataset = xr.Dataset({"NDSI_Snow_Cover": nasa_snow_cover})
        return out_dataset
