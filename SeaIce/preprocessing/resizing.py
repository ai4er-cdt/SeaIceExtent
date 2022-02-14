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


def change_resolution(in_path, out_path, new_resolution):
# Change the resolution.
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


