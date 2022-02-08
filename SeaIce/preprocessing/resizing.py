from preprocessing.shared import *
from preprocessing.data_handling import name_file


def change_resolution(in_path, out_path, out_name, new_resolution):
# Change the resolution.
    file_name = name_file(out_path, out_name, ".tif")
    gdal.Warp(file_name, in_path, xRes=new_resolution, yRes=new_resolution)
    

def resize_to_match(image_to_change, image_to_match, out_path):
# Resize the first image to match the size of the second image.
    template_image = gdal.Open(image_to_match)
    width, length = template_image.RasterXSize, template_image.RasterYSize 
    gdal.Translate(out_path, image_to_change, width=width, height=length)
    # Clean up.
    template_image.FlushCache()
    del template_image