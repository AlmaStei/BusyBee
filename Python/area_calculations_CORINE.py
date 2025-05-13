import processing
import csv
from qgis.core import QgsProject, QgsVectorLayer

# === User-defined file paths ===
# Path to the CORINE 2018 raster 
input_raster = 'C:/Users/Almas/QGIS_proj/CORINE_u2018_clc2018_v2020_20u1_raster100m/CORINE_2018_NS_AS.tif'
# Path to your 1km buffer zones GeoPackage (layer must contain a "Name" field)
buffer_file = 'C:/Users/Almas/QGIS_proj/NS_SA_sites/1km_buffer.gpkg'
# Path for the output CSV file
output_csv = 'C:/Users/Almas/QGIS_proj/CORINE_u2018_clc2018_v2020_20u1_raster100m/corine_area_by_buffer.csv'

# === Define valid CORINE land cover categories ===
valid_categories = [1, 2, 3, 11, 12, 18, 23, 24, 26, 29, 35]

# === Step 1. Convert CORINE Raster to Vector (Polygonize) ===
print("Converting CORINE raster to polygons...")
polygonize_params = {
    'INPUT': input_raster,
    'BAND': 1,
    'FIELD': 'DN',  # Field to store the raster pixel value (land cover class)
    'EIGHT_CONNECTEDNESS': False,
    'EXTRA': '',
    'OUTPUT': 'TEMPORARY_OUTPUT'
}
polygonize_result = processing.run("gdal:polygonize", polygonize_params)
landcover_vector = polygonize_result['OUTPUT']

# === Step 2. Load Buffer Zones Vector Layer ===
print("Loading buffer zones...")
buffer_layer = QgsVectorLayer(buffer_file, "buffer_zones", "ogr")
if not buffer_layer.isValid():
    raise Exception("Buffer layer failed to load. Check the file path and layer name.")

# === Step 3. Clip (Intersect) Land Cover with Buffer Zones ===
print("Computing intersections between buffer zones and CORINE land cover polygons...")
intersection_params = {
    'INPUT': buffer_layer,
    'OVERLAY': landcover_vector,
    'OUTPUT': 'TEMPORARY_OUTPUT'
}
intersection_result = processing.run("native:intersection", intersection_params)
intersection_layer = intersection_result['OUTPUT']

# === Step 4. Calculate Area per CORINE Land Cover Type Within Each Buffer ===
# Results dictionary structure:
# { buffer_zone_name: { land_cover_value: total_area, ... }, ... }
print("Calculating areas for each buffer and CORINE land cover type...")
results = {}
for feature in intersection_layer.getFeatures():
    # Retrieve the buffer zone name from the "Name" field
    zone_name = feature['Name']
    # Retrieve the CORINE land cover value from the polygonized output field ("DN")
    land_cover = feature['DN']
    
    # Filter out any features that do not belong to the valid CORINE categories
    if land_cover not in valid_categories:
        continue
    
    # Calculate the area of the intersected geometry (assumes CRS units in meters)
    area = feature.geometry().area()
    
    # Aggregate the area by buffer zone and land cover type
    if zone_name not in results:
        results[zone_name] = {}
    if land_cover not in results[zone_name]:
        results[zone_name][land_cover] = 0.0
    results[zone_name][land_cover] += area

# === Step 5. Write the Output Table to a CSV File ===
print("Writing results to CSV...")
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow(["Buffer", "LandCover", "Area_m2"])
    # Write one row for each combination of buffer zone and land cover type
    for zone, landcover_dict in results.items():
        for land_cover, area in landcover_dict.items():
            writer.writerow([zone, land_cover, area])

print("CSV output saved to:", output_csv)
