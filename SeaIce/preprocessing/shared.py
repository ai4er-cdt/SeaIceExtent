import rasterio
import os
from osgeo import ogr, gdal

program_path = os.getcwd()
temp_folder = r"{}\SeaIce\temp\temporary_files".format(program_path)
temp_buffer = r"{}\SeaIce\temp\temporary_buffer".format(program_path)
