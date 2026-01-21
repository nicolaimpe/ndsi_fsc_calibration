from typing import Dict

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from ndsi_fsc_calibration.utils import default_logger as logger


class Scatter:
    def snow_cover_value_bins(self):
        """Test data have to be normalized between 1 and 100 for snow cover."""
        return BinGrouper(
            np.array([*np.arange(-1, 101, 1), 255]),
            labels=np.array([*np.arange(0, 101, 1), 255]),
        )

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")

        quant_mask_ref = self.ref_analyzer.quantitative_mask(dataset.data_vars["ref_fsc"])
        quant_mask_test = self.test_analyzer.quantitative_mask(dataset.data_vars["ndsi"])
        quantitative_mask_union = quant_mask_test & quant_mask_ref
        n_intersecting_pixels = quantitative_mask_union.sum()

        if n_intersecting_pixels < 2:
            logger.info("No intersection found on this day. Returning a zeros array.")
            return xr.DataArray(0, coords=xr.Coordinates({k + "_bins": v.labels for k, v in bins_dict.items()}))

        dataset.data_vars["ref_fsc"].values = (
            dataset.data_vars["ref_fsc"].where(quantitative_mask_union) * 100 / self.test_analyzer.max_fsc
        )
        dataset.data_vars["ndsi"].values = (
            dataset.data_vars["ndsi"].where(quantitative_mask_union) * 100 / self.test_analyzer.max_fsc
        )
        scatter = dataset.groupby(bins_dict).map(self.compute_scatter_plot)

        return scatter

    def compute_scatter_plot(self, dataset: xr.Dataset):
        # Counting ref or test doesn't really change here
        return dataset.data_vars["ref_fsc"].count().rename("n_occurrences")

    def scatter_analysis(
        self,
        ndsi_time_series: xr.DataArray,
        ref_fsc_time_series: xr.DataArray,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset = xr.Dataset({"ndsi": ndsi_time_series, "ref_fsc": ref_fsc_time_series})
        analysis_bin_dict = self.snow_cover_value_bins(
            {"ndsi": self.snow_cover_value_bins(), "ref_fsc": self.snow_cover_value_bins()}
        )

        result = combined_dataset.groupby("time").map(self.time_step_analysis, bins_dict=analysis_bin_dict)
        if netcdf_export_path:
            logger.info(f"Exporting to {netcdf_export_path}")
            result.to_netcdf(netcdf_export_path, encoding={"n_occurrences": {"zlib": True}})
        return result
