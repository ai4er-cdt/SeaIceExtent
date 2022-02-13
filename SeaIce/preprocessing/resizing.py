from preprocessing.data_handling import *

def change_resolution(in_path, out_path, new_resolution):
# Change the resolution.
    gdal.Warp(out_path, in_path, xRes=new_resolution, yRes=new_resolution)
    

def resize_to_match(image_to_change, image_to_match, out_path):
    """ Resize the first image to match the size of the second image.
        Parameters: image_to_change: (string) file path of the image to be resized.
                    image_to_match: (string) the file path to the reference image.
                    out_path: (string) the file path the write the resized image.
    """ 
    template_image = gdal.Open(image_to_match)
    width, length = template_image.RasterXSize, template_image.RasterYSize 
    gdal.Translate(out_path, image_to_change, width=width, height=length)
    # Clean up.
    template_image.FlushCache()
    del template_image