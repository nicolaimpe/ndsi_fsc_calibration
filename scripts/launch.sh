#!/bin/bash

# Launch main on the Dolomites between 1/12/2023 and 1/1/2024 for VJ210A1 (VIIRS JPSS-2 snow cover product) and 
# trigger automatic data download from earthaccess (VJ210A1) and Py Hydroweb (Sentinel-2)
python scripts/main.py \
--start_date 20231201 \
--end_date 20240101 \
--aoi_file ./data/aoi_files/dolomites.geojson \
--product_name VJ210A1 \
--sentinel_2_folder ./data/S2_THEIA/ \
--nasa_folder ./data/V10A1/VJ210A1/ \
--output_folder ./output_folder/ \
--download_nasa
--download_s2


# Launch main on the French Alps between 1/12/2023 and 1/1/2024 for MOD10A1 (MODIS Terra snow cover product) 
# without triggering automatic download -> source data are assumed to be already available in sentinel_2_folder and nasa_folder

# python scripts/main.py \
# --start_date 20231201 \
# --end_date 20240101 \
# --aoi_file ./data/aoi_files/alpes_bbox.shp \
# --product_name MOD10A1 \
# --sentinel_2_folder ./data/S2_THEIA/ \
# --nasa_folder ./data/MOD10A1/ \
# --output_folder ./output_folder/ 


# Launch main on the French Alps between 1/12/2023 and 1/1/2024 for VNP10A1 (VIIRS SNPP snow cover product) 
# without triggering automatic download and specifying a custom resampling grid

# python scripts/main.py \
# --start_date 20231201 \
# --end_date 20240101 \
# --aoi_file ./data/aoi_files/alpes_bbox.shp \
# --product_name VJ210A1 \
# --sentinel_2_folder ./data/S2_THEIA/ \
# --nasa_folder ./data/V10A1/VJ210A1/ \
# --output_folder ./output_folder/ \
# --resampling_grid_file ./conf/grid_conf_example.yaml




