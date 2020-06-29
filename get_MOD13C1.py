
# Script should be in TATSSI root directory, e.g. $HOME/TATSSI

from datetime import datetime
from TATSSI.download.modis_downloader import get_modis_data

platform = "MOLT"
product = 'MOD13A2.006'
tiles = ["h17v02", "h17v03", "h17v04", "h18v03"]
output_dir = "/archive/modis/MOD13A2"
start_date = datetime(2000, 1, 1)
end_date = datetime(2019, 12, 31)
n_threads = 4
username = "tatssi"
password = "Tatssi2019"

# Get the data
get_modis_data(platform, product, tiles, output_dir, start_date,
               end_date, n_threads, username, password)

