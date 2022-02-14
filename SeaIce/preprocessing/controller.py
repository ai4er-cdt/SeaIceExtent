from preprocessing.data_handling import *
from preprocessing import stitching, resizing, clipping, relabelling, tiling

def preprocess(modis_paths = None, sar_path = None, shape_file_path, out_path = "temp", folder_name, resolution = 40, 
               relabel_from = [0], relabel_to = [0], relabel_scale = 1, tile_size = 512, step_size = 384):
   """
   Handles the sequence of performing all the preprocessing functions.
   Parameters: modis_paths: a list of file paths to adjoining optical images which can be empty if there are none. 
               sar_path: the file path to the radar image, set to None if there isn't one.
               shape_file_path: the file path to the .shp file associated with the images.
               out_path: the directory in which output is to be written. folder_name: the date that the images were collected.
               resolution: desired resolution of modis images. relabel_from and relabel_to: a pair of lists containing numbers to change
               labels from and to, respectively. relabel_scale: multiplier for all labelled pixel values. 
               tile_size: number of pixels of the length or width of square tiles.
               step_size: number of pixels to move along.
   Output: Tiles of whichever images are passed to the function, named by their date and image type, and a metadata entry.           
   """
   if len(modis_paths) != 0:
        if len(modis_paths) > 1:
            # Stitch the modis images together
            modis_array, modis_metadata = stitching.stitch(modis_paths)
            # Save the full image in temporary folder
            modis_file_path = save_tiff(modis_array, modis_metadata, "temp", "stitched")
        else:
            modis_file_path = modis_paths[0]
        # Upsample modis image
        upsampled_modis_path = name_file("temp", "upsampled", ".tif")
        resizing.change_resolution(modis_file_path, upsampled_modis_path, resolution)
        # Clip modis image
        modis_clipped, modis_metadata = clipping.clip(shape_file_path, upsampled_modis_path)
        modis_clipped_path = save_tiff(modis_clipped, modis_metadata, "temp", "clipped")
        # Resize the modis image to match the sar image
        if sar_path != None:
            resized_modis_path = name_file("temp", "resized", ".tif")
            resizing.resize_to_match(modis_clipped_path, sar_path, resized_modis_path)
   else:
        modis_file_path = None

    # These will be applied to either modis or sar, depending on which we have.
    # Relabel. Sar images are faster to work with. 
   labels_path = name_file("temp", "labels", ".tif") 
   if sar_path != None:
        relabelling.shp_to_tif(shape_file_path, sar_path, labels_path)
   else:
        relabelling.shp_to_tif(shape_file_path, modis_file_path, labels_path)

   relabelling.relabel(labels_path, relabel_from, relabel_to, relabel_scale)

    # Tile    
   tiled_path = name_file(out_path, folder_name, "")
   tiling.tile_images(modis_file_path, sar_path, labels_path, tiled_path, tile_size, step_size, folder_name)

   delete_temp_files()

