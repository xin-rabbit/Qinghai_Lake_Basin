//ERA5-Land 在流域范围内 2000-2025 年月平均气温(℃)、月总降水量(mm)、月总蒸发量(mm)

var roi = ee.FeatureCollection('projects/earthengine-legacy/assets/users/Luna13/test');
Map.addLayer(roi, {color: 'red'}, 'Basin');
Map.centerObject(roi, 8);

var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
  .filterDate('2000-01-01', '2025-12-31')
  .select([
    'temperature_2m',           // 月平均气温 (K)
    'total_precipitation_sum',  // 月总降水量 (m)
    'total_evaporation_sum'     // 月总蒸发量 (m)
  ]);

// 对每个影像：计算区域平均值，转换单位，添加年月信息
var monthlyFeatures = era5.map(function(img) {
  var date = ee.Date(img.get('system:time_start'));
  var year = date.get('year');
  var month = date.get('month');
  var stats = img.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 11132,          // ERA5 原始分辨率约 0.1° ≈ 11 km
    maxPixels: 1e13,
    bestEffort: true
  });
  
  // 单位转换
  // 气温：开尔文 → 摄氏度
  // 降水与蒸发：米 → 毫米
  var temp_c = ee.Image.constant(ee.Number(stats.get('temperature_2m')).subtract(273.15))
                .rename('temp_c');
  var precip_mm = ee.Image.constant(ee.Number(stats.get('total_precipitation_sum')).multiply(1000))
                  .rename('precip_mm');
  var evap_mm = ee.Image.constant(ee.Number(stats.get('total_evaporation_sum')).multiply(1000))
                .rename('evap_mm');
  
  // 将数值提取出来
  var tempVal = ee.Number(stats.get('temperature_2m')).subtract(273.15);
  var precipVal = ee.Number(stats.get('total_precipitation_sum')).multiply(1000);
  var evapVal = ee.Number(stats.get('total_evaporation_sum')).multiply(1000);
  
  var properties = {
    'year': year,
    'month': month,
    'temp_c': tempVal,
    'precip_mm': precipVal,
    'evap_mm': evapVal
  };
  
  // 返回一个点状 Feature
  return ee.Feature(null, properties);
});

// 转换为 FeatureCollection
var monthlyCollection = ee.FeatureCollection(monthlyFeatures);

Export.table.toDrive({
  collection: monthlyCollection,
  description: 'QinghaiLake_ERA5_Monthly_2000_2025',
  folder: 'Qinghai_Lake_Basin',
  fileFormat: 'CSV'
});

print(monthlyCollection.limit(5));