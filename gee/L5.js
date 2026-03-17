//Landsat 5 2000,2005,2010年9-10月，青海湖流域，去云+去阴影拼接

var Qinghai = ee.Geometry.Rectangle([99, 36.2, 101.3, 37.9]);
var startDate = '2010-09-01';
var endDate = '2010-10-30';

// 掩膜函数：去除云和云阴影
function maskCloudAndShadow(image) {
  var qa = image.select('QA_PIXEL');
  
  // Landsat Collection 2 QA_PIXEL 位定义
  var cloudBit = 1 << 3;        // 云
  var cloudShadowBit = 1 << 4;  // 云阴影
  var cirrusBit = 1 << 2;       // 卷云
  var dilatedBit = 1 << 1;      // 膨胀云

  // 创建掩膜：云位和阴影位均为 0 的像素保留
  var mask = qa.bitwiseAnd(cloudBit).eq(0)
      .and(qa.bitwiseAnd(cloudShadowBit).eq(0))
      .and(qa.bitwiseAnd(cirrusBit).eq(0))  
      .and(qa.bitwiseAnd(dilatedBit).eq(0));  

  return image.updateMask(mask);
}

var l5Collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(Qinghai)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUD_COVER', 30));  

// 应用云+阴影掩膜，缩放
var processed = l5Collection.map(function(img) {
  var masked = maskCloudAndShadow(img);
  var scaled = masked.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
                .multiply(0.0000275).add(-0.2)
                .max(0)  // 确保反射率非负
                .copyProperties(img, ['system:time_start', 'CLOUD_COVER']);
  return scaled;
});

// 按云量升序排序（云量少的在前），再反转，使云量少的影像在最后（mosaic 时覆盖前面的）
var sorted = processed.sort('CLOUD_COVER').toList(processed.size()).reverse();
var mosaic = ee.ImageCollection.fromImages(sorted).mosaic().clip(Qinghai);

Map.centerObject(Qinghai, 8);
Map.addLayer(Qinghai, {color: 'red'}, '青海湖流域');
Map.addLayer(mosaic, {bands: ['SR_B3', 'SR_B2', 'SR_B1'], min: 0, max: 0.3}, '去云+去阴影拼接');


Export.image.toDrive({
  image: mosaic,
  description: 'L5_2010', 
  folder: 'Qinghai_Lake_Basin',
  scale: 30,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});