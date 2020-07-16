
"""
VIIRS downloading tool to obtain data from the
Land Processes Distributed Active Archive Center (LP DAAC).
https://lpdaac.usgs.gov/dataset_discovery/viirs/viirs_products_table

Authentication via the EarthData login.
https://urs.earthdata.nasa.gov/
"""

from functools import partial
import os
import datetime
import time
import json
import pickle
from pathlib import Path

import requests
from concurrent import futures

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)
BASE_URL = "http://e4ftl01.cr.usgs.gov/"

class WebError (RuntimeError):
    """An exception for web issues"""
    def __init__(self, arg):
        self.args = arg

def read_config():
    """
    Read downloaders config file
    """
    downloaders_dir = os.path.dirname(__file__)
    fname = os.path.join(downloaders_dir, 'config.json')
    with open(fname) as f:
        credentials = json.load(f)

    url = credentials['url']
    username = credentials['username']
    password = credentials['password']

    return url, username, password

def save_available_dates(product, avail_dates):
    """
    Save the available dates for a specific product on a pickle file
    in $HOME/.TATSSI/{product}
    """
    homedir = os.path.expanduser("~")
    config_dir = os.path.join(homedir, '.TATSSI')
    # Create TATSSI config dir
    Path(config_dir).mkdir(parents=True, exist_ok=True)

    fname = os.path.join(config_dir, product)

    with open(fname, 'wb') as f:
        pickle.dump(avail_dates, f)

    f.close()

def get_available_dates_from_cache(product):
    """
    Get the available dates for a specific product from the
    TATSSI config dir
    """
    homedir = os.path.expanduser("~")
    config_dir = os.path.join(homedir, '.TATSSI')

    fname = os.path.join(config_dir, product)

    if os.path.exists(fname) is True:
        with open(fname, 'rb') as f:
            avail_dates = pickle.load(f)
    else:
        avail_dates = []

    return avail_dates

def get_available_dates(url, product, start_date, end_date=None,
                        use_cache=False):
    """
    This function gets the available dates for a particular
    product, and returns the ones that fall within a particular
    pair of dates. If the end date is set to ``None``, it will
    be assumed it is today. If use_cache is True then first the
    available dates will be first obtained from any cache available.
    """
    if use_cache is True:
        avail_dates = get_available_dates_from_cache(product)
    else:
        avail_dates = []

    if end_date is None:
        end_date = datetime.datetime.now()
    r = requests.get(url)
    if not r.ok:
        raise WebError(
            "Problem contacting NASA server. Either server " +
            "is down, or the product you used (%s) is kanckered" %
            url)
    html = r.text

    for line in html.splitlines()[19:]:
        if line.find("[DIR]") >= 0 and line.find("href") >= 0:
            this_date = line.split("href=")[1].split('"')[1].strip("/")
            this_datetime = datetime.datetime.strptime(this_date,
                                                       "%Y.%m.%d")
            if this_datetime >= start_date and this_datetime <= end_date:
                avail_dates.append(url + "/" + this_date)

    # Save search to cache
    if len(avail_dates) > 0:
        save_available_dates(product, avail_dates)

    return avail_dates

def download_tile_list(url, tiles):
    """
    For a particular product and date, obtain the data tile URLs.
    """
    if not isinstance(tiles, type([])):
        tiles = [tiles]
    while True:
        try:
            r = requests.get(url )
            break
        except requests.execeptions.ConnectionError:
            time.sleep(240)
            
    grab = []
    for line in r.text.splitlines():
        for tile in tiles:
            if line.find ( tile ) >= 0 and line.find (".xml" ) < 0 \
                    and line.find("BROWSE") < 0:
                fname = line.split("href=")[1].split('"')[1]
                grab.append(url + "/" + fname)
    return grab

def download_tiles(url, session, username, password, output_dir):

    r1 = session.request('get', url)
    r = session.get(r1.url, stream=True)
    fname = url.split("/")[-1]
    LOG.debug("Getting %s from %s(-> %s)" % (fname, url, r1.url))

    if not r.ok:
        # raise IOError(f"Can't start download... {fname}")
        print(f"Can't start download... {fname}. Try download again.")
        return

    file_size = int(r.headers['content-length'])
    LOG.debug("\t%s file size: %d" % (fname, file_size))
    output_fname = os.path.join(output_dir, fname)

    # Save with temporary filename...
    with open(output_fname+".partial", 'wb') as fp:
        for block in r.iter_content(65536):
            fp.write(block)

    # Rename to definitive filename
    os.rename(output_fname+".partial", output_fname)
    LOG.info("Done with %s" % output_fname)
    return output_fname

def required_files (url_list, output_dir):
    """
    Checks for files that are already available in the system.
    """
    
    all_files_present = os.listdir (output_dir)
    hdf_files_present = [fich 
                        for fich in all_files_present if fich.endswith(".h5")]
    hdf_files_present = set(hdf_files_present)
    
    flist= [url.split("/")[-1] for url in url_list]
    file_list = dict(zip(flist, url_list))
    flist = set(flist)
    files_to_download = list(flist.difference(hdf_files_present))
    to_download = [ file_list[k] for k in files_to_download]

    return to_download
    
def get_viirs_data(platform, product, tiles, 
                   output_dir, start_date,
                   end_date=None, n_threads=5,
                   username=None, password=None,
                   progressBar=None,
                   use_cache=False):
    """The main workhorse of VIIRS downloading. The products are specified
    by their VIIRS code (e.g. VNP13A1.001 or VNP09A1.001).
    You need to specify a tile (or a list of tiles), as well as a starting
    and end date. If the end date is not specified, the current date will
    be chosen. Additionally, you can specify the number of parallel threads
    to use. And you also need to give an output directory to dump your files.

    Parameters
    -----------
    usearname: str
        The username that is required to download data from the VIIRS archive.
    password: str
        The password required to download data from the VIIRS archive.
    platform: str
        The platform, MOLT, MOLA or MOTA. This basically relates to the sensor
        used (or if a combination of AQUA & TERRA is used)
    product: str
        The VIIRS product. The product name should be in VIIRS format
        (VNP13A1.001, so product acronym dot collection)
    tiles: str or iter
        A string with a single tile (e.g. "h17v04") or a lits of such strings.
    output_dir: str
        The output directory
    start_date: datetime
        The starting date as a datetime object
    end_date: datetime
        The end date as a datetime object. If not specified, taken as today.
    n_threads: int
        The number of concurrent downloads to envisage. I haven't got a clue
        as to what a good number would be here...
    use_cache: boolean
        Whether to use local cache with results from previous searches

    """
        # Read config
    BASE_URL, _username, _password = read_config()
    if username is not None:
        username = _username
    if password is not None:
        password = _password

    # Ensure the platform is OK
    assert platform.upper() in ["VIIRS"], \
        "%s is not a valid platform. Valid one is VIIRS" % \
        platform

    # If output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Cook the URL for the product
    url = BASE_URL + platform + "/" + product
    # Get all the available dates in the NASA archive...
    the_dates = get_available_dates(url, product, start_date,
            end_date=end_date, use_cache=use_cache)
    
    # We then explore the NASA archive for the dates that we are going to
    # download. This is done in parallel. For each date, we will get the
    # url for each of the tiles that are required.
    the_tiles = []
    download_tile_patch = partial(download_tile_list, tiles=tiles)
    with futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        for tiles in executor.map(download_tile_patch, the_dates):
            the_tiles.append(tiles)

    # Flatten the list of lists...
    gr = [g for tile in the_tiles for g in tile]
    gr.sort()

    # Check whether we have some files available already
    gr_to_dload = required_files(gr, output_dir)
    gr = gr_to_dload

    msg = f"Will download {len(gr)} files..."
    if progressBar is not None:
        progressBar.setFormat(msg)
    
    LOG.info(msg)
    # Wait for a few seconds before downloading the data
    time.sleep(5)

    # The main download loop. This will get all the URLs with the filenames,
    # and start downloading them in parallel.
    dload_files = []
    with requests.Session() as s:
        s.auth = (username, password)
        download_tile_patch = partial(download_tiles,
                                     session=s,
                                     output_dir=output_dir,
                                     username=username,
                                     password=password )
        
        with futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            for fich in executor.map(download_tile_patch, gr):
                dload_files.append(fich)
                if progressBar is not None:
                    progressBar.setValue((len(dload_files) / len(gr)) * 100.0)
        
    return dload_files
