from typing import Dict, Tuple

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from ndsi_fsc_calibration.snow_cover_products import NASA_CLASSES, S2_CLASSES
from ndsi_fsc_calibration.utils import default_logger as logger


class Scatter:
    """Match an evaluated snow cover NDSI product with a reference FSC product.

    Assume that eval and ref have their snow values between 1 and 100 with for no snow."""

    def __init__(self, eval_product: str, ref_product: str):
        """

        Args:
            eval_product (str): identifier of evaluated product
            ref_product (str): identifier of reference product
        """
        self.eval_product = eval_product
        self.ref_product = ref_product

    def snow_cover_value_bins(self):
        """Eval data have to be normalized between 1 and 100 for snow cover."""
        return BinGrouper(
            np.array([*np.arange(-1, 101, 1), 255]),
            labels=np.array([*np.arange(0, 101, 1), 255]),
        )

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]) -> xr.DataArray:
        """Find the union of valid masks for eval and ref products and compute correspondences.

        Args:
            dataset (xr.Dataset): a geographic dataset containing eval NDSI and ref FSC on the same grid for a give time
            bins_dict (Dict[str, xr.groupers.Grouper]): FSC [1,100] and NDSI [1,100] values are used as bins

        Raises:
            NotImplementedError: if the eval product is not supported
            NotImplementedError: if the ref product is not supported

        Returns:
            xr.DataArray: an array of the individual correspondences for each (NDSI,FSC) pair
        """
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")

        if self.eval_product in ("VNP10A1", "VJ110A1", "VJ210A1", "MOD10A1"):
            quant_mask_test = dataset.data_vars["eval_ndsi"] < NASA_CLASSES["snow_cover"][-1]
            eval_prod_max_ndsi = NASA_CLASSES["snow_cover"][-1]
        else:
            raise NotImplementedError

        if self.ref_product == "S2":
            quant_mask_ref = dataset.data_vars["ref_fsc"] < S2_CLASSES["snow_cover"][-1]
            ref_prod_max_fsc = S2_CLASSES["snow_cover"][-1]
        else:
            raise NotImplementedError

        quantitative_mask_union = quant_mask_test & quant_mask_ref
        n_intersecting_pixels = quantitative_mask_union.sum()

        if n_intersecting_pixels < 2:
            logger.info("No intersection found on this day. Returning a zeros array.")
            return xr.DataArray(0, coords=xr.Coordinates({k + "_bins": v.labels for k, v in bins_dict.items()}))

        dataset.data_vars["ref_fsc"].values = (
            dataset.data_vars["ref_fsc"].where(quantitative_mask_union) * 100 / ref_prod_max_fsc
        )
        dataset.data_vars["eval_ndsi"].values = (
            dataset.data_vars["eval_ndsi"].where(quantitative_mask_union) * 100 / eval_prod_max_ndsi
        )
        scatter = dataset.groupby(bins_dict).map(self.compute_occurrences)
        return scatter

    def compute_occurrences(self, dataset: xr.Dataset) -> xr.Dataset:
        """Count number of correspondences for each (NDSI, FSC) value pair."""
        # Counting ref or test doesn't really change here
        return dataset.data_vars["ref_fsc"].count().rename("n_occurrences")

    def prepare_analysis(
        self, ndsi_time_series: xr.DataArray, fsc_time_series: xr.DataArray
    ) -> Tuple[xr.Dataset, Dict[str, BinGrouper]]:
        """Combine eval and ref time series on a Dataset and cread (NDSI,FSC) pair bins for xarray groupby based matching.

        ndsi_time_series and fsc_time_series have to be on the same grid.

        Args:
            ndsi_time_series (xr.DataArray): Spatio-temporal array of regridded NDSI_Snow_Cover from NASA products
            fsc_time_series (xr.DataArray): Spatio-temporal array of Sentinel-2 regridded FSC

        Returns:
            Tuple[xr.Dataset, Dict[str, BinGrouper]]: NDSI and FSC data combines, (NDSI,FSC) bins
        """
        combined_dataset = xr.Dataset({"eval_ndsi": ndsi_time_series, "ref_fsc": fsc_time_series})
        analysis_bin_dict = {"eval_ndsi": self.snow_cover_value_bins(), "ref_fsc": self.snow_cover_value_bins()}
        return combined_dataset, analysis_bin_dict

    def compute_all_correspondences(
        self,
        eval_ndsi_time_series: xr.DataArray,
        ref_fsc_time_series: xr.DataArray,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        """Wrap-up of this class methods.

        Args:
            eval_ndsi_time_series (xr.DataArray): Spatio-temporal array of regridded NDSI_Snow_Cover from NASA products
            ref_fsc_time_series (xr.DataArray): Spatio-temporal array of Sentinel-2 regridded FSC
            netcdf_export_path (str | None, optional): where to export the correspondence dataset. Defaults to None.

        Returns:
            xr.Dataset: the correspondence dataset
        """
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            ndsi_time_series=eval_ndsi_time_series, fsc_time_series=ref_fsc_time_series
        )
        result = combined_dataset.groupby("time").map(self.time_step_analysis, bins_dict=analysis_bin_dict)
        result = xr.Dataset({"n_occurrences": result}).rename_dims({"eval_ndsi_bins": "ndsi", "ref_fsc_bins": "fsc"})
        if netcdf_export_path:
            logger.info(f"Exporting to {netcdf_export_path}")
            result.to_netcdf(netcdf_export_path, encoding={"n_occurrences": {"zlib": True}})
        return result
