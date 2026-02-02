from datetime import datetime

import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from matplotlib import pyplot as plt
from pyproj import CRS

from ndsi_fsc_calibration.match import Scatter
from ndsi_fsc_calibration.regrid import S2TheiaRegrid, V10Regrid
from ndsi_fsc_calibration.visualization import scatter_plot_with_fit

output_grid = GSGrid(
    x0=4.2099998999999997,
    y0=46.5565352276687463,
    resolution=(0.003374578177758, 0.0033740359897170007),
    width=1000,
    height=1200,
    crs=CRS.from_epsg(4326),
)
data_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/"
start, end = datetime(year=2023, month=12, day=1), datetime(year=2023, month=12, day=30)
area_of_interest_file = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/alpes_bbox.shp"
eval_product_name = "VNP10A1"

vnp10a1_regridder = V10Regrid(
    output_grid=output_grid,
    data_folder=f"{data_folder}/V10A1/VNP10A1",
    output_folder=f"output_folder/{eval_product_name.lower()}",
)
s2_regridder = S2TheiaRegrid(output_grid=output_grid, data_folder=f"{data_folder}/S2_THEIA", output_folder="output_folder/s2")

matcher = Scatter(eval_product=eval_product_name, ref_product="S2")

vnp10a1_regridder.create_time_series(
    roi_shapefile=area_of_interest_file,
    start_date=start,
    end_date=end,
)

s2_regridder.create_time_series(
    roi_shapefile=area_of_interest_file,
    start_date=start,
    end_date=end,
)

vnp10a1_regridded = xr.open_dataset(f"output_folder/{eval_product_name.lower()}/regridded.nc")
s2_regridded = xr.open_dataset("output_folder/s2/regridded.nc")


matcher.compute_all_correspondences(
    eval_ndsi_time_series=vnp10a1_regridded.data_vars["NDSI_Snow_Cover"],
    ref_fsc_time_series=s2_regridded.data_vars["snow_cover_fraction"],
    netcdf_export_path="output_folder/correspondences.nc",
)

correspondence_dataset = xr.open_dataset("output_folder/correspondences.nc")
fig, ax = plt.subplots()

fig, ax = scatter_plot_with_fit(
    data_to_plt=correspondence_dataset.sum(dim="time").data_vars["n_occurrences"],
    fig=fig,
    ax=ax,
    eval_prod_name=eval_product_name,
    quantile_max=1,
    quantile_min=0.2,
)

fig.patch.set_alpha(0.0)
fig.savefig("output_folder/scatter_plot.png", format="png", dpi=600)
plt.show()
