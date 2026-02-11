help_message = """
Download products from your hydroweb.next projects (https://hydroweb.next.theia-land.fr) using the py-hydroweb lib (https://pypi.org/project/py-hydroweb/)
This script is an example tuned for your last hydroweb.next project but feel free to adapt it for future requests.
Follow these steps:
1. If not already done, install py-hydroweb latest version using `pip install -U py-hydroweb` (WARNING: python >= 3.8 is required)
2a. Generate an API-Key from hydroweb.next portal in your user settings
2b. Carefully store your API-Key (2 options):
- either in an environment variable `export HYDROWEB_API_KEY="<your_key_here>"`
- or in below script by replacing <your_key_here>
3. You can change download directory by adding an `output_folder` parameter when calling `submit_and_download_zip` (see below). By default, current path is used.
4. You are all set, run this script `python download_script.py`

For more documentation about how to use the py-hydroweb lib, please refer to https://pypi.org/project/py-hydroweb/.
"""

import logging
import sys
import zipfile
from datetime import datetime
from importlib.metadata import version
from typing import Tuple


def download_s2_fsc_pyhydroweb(
    start_date: datetime, end_date: datetime, bounding_box: Tuple[float, float, float, float], download_folder: str
):
    try:
        import py_hydroweb
    except ImportError:
        print(help_message)
        exit(1)

    # Check py-hydroweb version
    latest_version = "1.1.0"
    if version("py_hydroweb") < latest_version:
        logging.getLogger().warning(f"""\033[33m
    /!\ Consider upgrading py-hydroweb to {latest_version} using `pip install -U py-hydroweb`
    \033[0m""")

    # Set log config
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Create a client
    #  - either using the API-Key environment variable (HYDROWEB_API_KEY)
    client: py_hydroweb.Client = py_hydroweb.Client("https://hydroweb.next.theia-land.fr/api")
    #  - or explicitly giving API-Key (comment line above and uncomment line below)
    # client: py_hydroweb.Client = py_hydroweb.Client("https://hydroweb.next.theia-land.fr/api", api_key="<your_key_here>")

    # Initiate a new download basket (input the name you want here)
    basket: py_hydroweb.DownloadBasket = py_hydroweb.DownloadBasket("my_download_basket")

    # Add collections in our basket
    basket.add_collection(
        "LIS_FSC_PREOP",
        bbox=list(bounding_box),
        query={
            "start_datetime": {"lte": f"{start_date.strftime('%Y%m%dT%H%M%S')}"},
            "end_datetime": {"gte": f"{end_date.strftime('%Y%m%dT%H%M%S')}"},
        },
    )

    # Do download (input the archive name you want here, and optionally an output folder)
    now = datetime.today().strftime("%Y%m%dT%H%M%S")
    downloaded_zip_path: str = client.submit_and_download_zip(
        basket,
        zip_filename=f"my_hydroweb_data_{now}.zip",
        output_folder=download_folder,
    )
    zipfile.ZipFile.extractall(downloaded_zip_path)
