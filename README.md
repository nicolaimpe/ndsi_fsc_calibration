# ndsi_fsc_calibration

![Example scatter plot with fit](./output_folder/plots/scatter_plot_example.png) 

## Description
Tool to automize moderate resolution satellite snow cover products (VIIRS, MODIS) characterization using Sentinel-2 as reference. Visualize agreement between NDSI computed from MODIS and VIIRS vs the referFit a linear model for Normalized Snow Index (NDSI) to Fractional Snow Cover (FSC) for moderate resolution sensors using Sentinel-2 as high resolution reference.

!! Will work only where Sentinel-2 FSC products are available !!"

The supported products are (VIIRS) VNP10A1, VJ110A1, VJ210A1, MOD10A1. 

Given an area of interest vectorial file and a time period, this code automizes the following ste ps:
1. Download of VIIRS/MODIS and Sentinel-2 source data on an area of interest
2. Regrid Sentinel-2 FSC on VIIRS/MODIS grid (specify another frid is also possible)
3. Match evaluation and reference datasets: for each combination of VIIRS/MODIS NDSI (0 to 100 %) and Sentinel-2 FSC (0 to 100 %), compute the number of occurrences
4.  Fit a linear regression on the (NDSI, FSC) dataset obtained in step 3
5. Visualize result with a scatter plot

Intermediate products are saved in the user defined output folder:

`<output_folder>/s2/regridded.nc` : Time series of Sentinel-2 regridded on the evaluation grid (i.e. resolution of at least MODIS/VIIRS)

`<output_folder>/<product_id>/regridded.nc` : Time series of the avaluated product on the evaluation grid (i.e. resolution of at least MODIS/VIIRS)

`<output_folder>/correspondences.nc` : array of the number of correspondences for each combination of <product_id> NDSI and reference Sentinel-2 FSC for each time step 

The processing is illustrated in this flow chart:

![Pipeline](./data/illustrations/data_flow_illustration.png) 

```bash
$ tree
.
├── conf
│   └── grid_conf_example.yaml
├── data
│   ├── aoi_files
├── output_folder
│   ├── correspondences.nc
│   ├── plots
│   │   └── scatter_plot.png
│   ├── s2
│   │   ├── regridded.nc
│   └── vnp10a1
│       └── regridded.nc
├── pdm.lock
├── pyproject.toml
├── README.md
├── scripts
│   ├── launch.sh
│   └── main.py
├── src
│   └── ndsi_fsc_calibration
│       ├── download.py
│       ├── fit_linear_model.py
│       ├── __init__.py
│       ├── match.py
│       ├── regrid.py
│       ├── snow_cover_products.py
│       ├── utils.py
└──     └── visualization.py
```

## Install

```bash
git clone git@github.com:nicolaimpe/ndsi_fsc_calibration.git
cd ndsi_fsc_calibration
pip install .
```

## Usage

See `scripts/launch.py`

Examples:

Launch main on the Dolomites between 1/12/2023 and 1/1/2024 for VJ210A1 (VIIRS JPSS-2 snow cover product) and trigger automatic data download from earthaccess (VJ210A1) and Py Hydroweb (Sentinel-2).

For this use case NASA Earthdata account is needed.



```bash
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
```

Launch main on the French Alps between 1/12/2023 and 1/1/2024 for MOD10A1 (MODIS Terra snow cover product) without triggering automatic download -> source data are assumed to be already available in sentinel_2_folder and nasa_folder

```bash
python scripts/main.py \
--start_date 20231201 \
--end_date 20240101 \
--aoi_file ./data/aoi_files/alpes_bbox.shp \
--product_name MOD10A1 \
--sentinel_2_folder ./data/S2_THEIA/ \
--nasa_folder ./data/MOD10A1/ \
--output_folder ./output_folder/ 
```

Launch main on the French Alps between 1/12/2023 and 1/1/2024 for VNP10A1 (VIIRS SNPP snow cover product) without triggering automatic download and specifying a custom resampling grid

```bash
python scripts/main.py \
--start_date 20231201 \
--end_date 20240101 \
--aoi_file ./data/aoi_files/alpes_bbox.shp \
--product_name VJ210A1 \
--sentinel_2_folder ./data/S2_THEIA/ \
--nasa_folder ./data/V10A1/VJ210A1/ \
--output_folder ./output_folder/ \
--resampling_grid_file ./conf/grid_conf_example.yaml
```

## Contributing

Contributions are welcome.

PDM is recommended for environment management.

```bash
pip install pdm
pdm install
```

To allow pdm to install packages in the virtual environment
```bash
pdm use -f $(which python)
```

To add a package to the project

```bash
pip add <your_package>
```