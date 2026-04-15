// 从 ERA5‑Land 月度数据中提取青海湖流域 2000‑2025 年逐月平均径流（mm）

var runoffColl = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');
var qinghaiBasin = ee.FeatureCollection('projects/earthengine-legacy/assets/users/Luna13/test');
Map.addLayer(qinghaiBasin, {color: 'red'}, 'Basin');
Map.centerObject(qinghaiBasin, 8);

var startYear = 2000;
var endYear = 2025;
var months = ee.List.sequence(1, 12);

function extractRunoff(year, month) {
  var startDate = ee.Date.fromYMD(year, month, 1);
  var endDate = startDate.advance(1, 'month');
  var monthlyRunoff = runoffColl
    .filterDate(startDate, endDate)
    .select('runoff_sum')
    .mean()
    .clip(qinghaiBasin);
  // 单位转换：m → mm（乘以1000）
  var runoff_mm = monthlyRunoff.multiply(1000);
  return runoff_mm.set('year', year, 'month', month);
}

var allImages = [];
for (var y = startYear; y <= endYear; y++) {
  for (var m = 1; m <= 12; m++) {
    allImages.push(extractRunoff(y, m));
  }
}

var runoffFC = ee.FeatureCollection(
  allImages.map(function(img) {
    var meanVal = img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: qinghaiBasin,
      scale: 10000,
      bestEffort: true
    }).get('runoff_sum');
    return ee.Feature(null, {
      year: img.get('year'),
      month: img.get('month'),
      runoff_mm: meanVal
    });
  })
);

Export.table.toDrive({
  collection: runoffFC,
  description: 'Qinghai_Runoff',
  folder: 'Qinghai_Lake_Basin',
  fileFormat: 'CSV'
});