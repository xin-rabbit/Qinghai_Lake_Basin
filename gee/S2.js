//Sentinel-2 2025年9-10月，青海湖流域，云量优先拼接

var Qinghai = ee.Geometry.Rectangle([99, 36.2, 101.3, 37.9]);
var startDate = '2025-09-01';
var endDate = '2025-10-15';

var l5Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(Qinghai)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

var sorted = l5Collection.sort('CLOUDY_PIXEL_PERCENTAGE', false);
var mosaic = sorted.mosaic().clip(Qinghai);
var scaled = mosaic.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                  .multiply(0.0001);

Map.centerObject(Qinghai, 8);
Map.addLayer(Qinghai, {color: 'red'}, '青海湖流域');
Map.addLayer(scaled, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, '云量优先拼接');

Export.image.toDrive({
  image: scaled,
  description: 'S2_2025',
  folder: 'Qinghai_Lake_Basin',
  scale: 20,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});