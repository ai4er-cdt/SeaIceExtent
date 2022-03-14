"""This is the resizing module for the Sea Ice Extent GTC Project data preprocessing step.

This module contains functions to resize and reproject GeoTIFF images in preprocessing.
"""

from preprocessing.data_handling import *


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


def halve_size(image_path, out_path):
    """Quarter the size of an image by halving both length and width.
       Parameters:
            image_path: (string) file path of large image.
            out_path: (string) file path to write the new, smaller image.
       Output: tiff image smaller copy of original.
    """
    large_image = gdal.Open(image_path) 
    width, length = large_image.RasterXSize, large_image.RasterYSize 
    gdal.Translate(out_path, image_path, width=round(width/2), height=round(length/2))
    # Clean up.
    large_image.FlushCache()
    del large_image


def change_resolution(in_path, out_path, new_resolution = 40):
    """ Change the resolution.
        Parameters:
            in_path: (string) file path of original image.
            out_path: (string) file path of new copy to write.
            new_resolution: (int) desired resolution in metres.
        Output: New tiff image.
    """
    gdal.Warp(out_path, in_path, xRes=new_resolution, yRes=new_resolution)
    

def reproj_tif(original_tif, tif_target_proj, output_tif):
    """Reproject an image.
       Parameters: original_tif: (string) file name of raster to be reprojected. 
                   tif_target_proj: (string) tif file name with projection to be used. 
                   output_tif: (string) name of output raster
    """
    target_file = gdal.Open(tif_target_proj) 
    # Use the defined target projection information to reproject the input raster to be modified
    prj_target=target_file.GetProjection() 
    input_raster = gdal.Open(original_tif)
    # Reproject and write the reprojected raster
    gdal.Warp(output_tif, input_raster, dstSRS = prj_target)


