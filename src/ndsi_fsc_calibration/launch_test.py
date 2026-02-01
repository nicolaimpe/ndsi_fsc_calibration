from datetime import datetime

from geospatial_grid.gsgrid import GSGrid
from pyproj import CRS

from ndsi_fsc_calibration.regrid import S2TheiaRegrid, V10Regrid

output_grid = GSGrid(
    x0=4.2099998999999997,
    y0=46.5565352276687463,
    resolution=(0.003374578177758, 0.0033740359897170007),
    width=1000,
    height=1200,
    crs=CRS.from_epsg(4326),
)
data_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/"
start, end = datetime(year=2023, month=12, day=1), datetime(year=2023, month=12, day=5)
area_of_interest_file = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/alpes_bbox.shp"


vnp10a1_regridder = V10Regrid(
    output_grid=output_grid, data_folder=f"{data_folder}/V10A1/VNP10A1", output_folder="output_folder/vnp10a1"
)
s2_regridder = S2TheiaRegrid(output_grid=output_grid, data_folder=f"{data_folder}/S2_THEIA", output_folder="output_folder/s2")


# vnp10a1_regridder.create_time_series(
#     roi_shapefile=area_of_interest_file,
#     start_date=start,
#     end_date=end,
# )

s2_regridder.create_time_series(
    roi_shapefile=area_of_interest_file,
    start_date=start,
    end_date=end,
)
