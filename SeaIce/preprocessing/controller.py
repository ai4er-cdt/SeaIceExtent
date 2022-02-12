from preprocessing.data_handling import *
from preprocessing import stitching, resizing, clipping, relabelling, tiling, controller

def preprocess(modis_paths, sar_path, shape_file_path, out_path, folder_name, resolution, relabel_from, relabel_to, relabel_scale, tile_size, step_size):
   
    if sar_path != None:
        has_sar = True
    else:
        has_sar = False

    if len(modis_paths) != 0:
        has_modis = True
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
        if has_sar:
            resized_modis_path = name_file("temp", "resized", ".tif")
            resizing.resize_to_match(modis_clipped_path, sar_path, resized_modis_path)
    else:
        has_modis = False

    # These will be applied to either modis or sar, depending on which we have.
    # Relabel. Sar images are faster to work with. 
    labels_path = name_file("temp", "labels", ".tif") 
    if has_sar:  
        relabelling.shp_to_tif(shape_file_path, sar_path, labels_path)
    else:
        sar_path = None
        relabelling.shp_to_tif(shape_file_path, modis_file_path, labels_path)

    # Old format: 0 = no data. 1 = ice free. 2 = sea ice. 9 = on land or ice shelf. 10 = unclassified.        
        # New format: 0 = ignore (for now). 1 = water. 2 = ice.
    # 1 is already water and 2 is already ice so there is no need to waste time checking or changing them.
    relabelling.relabel(labels_path, relabel_from, relabel_to, relabel_scale)

    # Tile    
    tiled_path = name_file(out_path, folder_name, "")
    tiling.tile_images(modis_file_path, sar_path, labels_path, tiled_path, tile_size, step_size, folder_name)

    delete_temp_files()

