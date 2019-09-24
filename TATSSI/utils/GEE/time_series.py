
import ee
import ee.mapclient

ee.Initialize()
ee.mapclient.centerMap(-98.0, 19.0, 6)

dataset = ee.ImageCollection('MODIS/006/MCD43A4').filter(
              ee.Filter.date('2012-08-01', '2019-07-31'))

nir = dataset.select(['Nadir_Reflectance_Band2'])

nir_vis = {
  'min': 1.0,
  'max': 6000.0,
  'gamma': 1.4,
}

ee.mapclient.addToMap(nir, nir_vis, 'NIR')
