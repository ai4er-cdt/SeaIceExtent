from logging import raiseExceptions
from PIL import Image
from preprocessing.data_handling import *


def tile_training_images(labels_path, out_path, tile_size, step_size, date_name, modis_path = None, sar_path = None): 
    """Divide associated images into tiles and save the tiles and their metadata.
       Parameters: labels_path: (string) file path of labelled raster.
                   out_path: (string) path to directory to write output.
                   tile_size: (int) number of pixels in length or width of square tile.
                   step_size: (int) number of pixels to move.
                   date_name: (string) the date the images were collected.
                   modis_path: (string) file path of 3 band optical image, or None.
                   sar_path: (string) file path of radar image, or None.
    """
    if modis_path == None and sar_path == None:
        raise Exception(help(tile_training_images), "No optical or radar image provided. The file path to least one of these must be supplied.")

    has_modis, has_sar = False, False

    window_shape = (tile_size, tile_size)
    labels_window = tif_to_window(labels_path, window_shape, step_size)
    if sar_path != None:
        has_sar = True
        sar_window = tif_to_window(sar_path, window_shape, step_size)
        image_data = gdal.Open(sar_path)
        image_window = sar_window

    # Modis has RGB channels so needs a different window shape to the other images.
    if modis_path != None:
        has_modis = True
        window_shape = (tile_size, tile_size, 3)
        modis_window = tif_to_window(modis_path, window_shape, step_size)
        image_data = gdal.Open(modis_path)
        image_window = modis_window

    geography = image_data.GetGeoTransform()
    top_left = geography[0], geography[3]
    image_data.FlushCache()
    del image_data
    num_shape = labels_window.shape[1]

    if has_modis and has_sar:
        for row_count, (row_modis, row_sar, row_labels) in enumerate(zip(modis_window, sar_window, labels_window)):
            for tile_count, (tile_modis, tile_sar, tile_labels) in enumerate(zip(row_modis, row_sar, row_labels)):
                tile_num = num_shape * row_count + tile_count
                # Check if the label tile contains any unclassified / no data. Discard them if so.
                if np.amin(tile_labels) == 0:
                    continue
                n_water = np.count_nonzero(tile_labels == 100)
                n_ice = np.count_nonzero(tile_labels == 200)
                # Save the tiles.
                for each_image in [["modis", tile_modis], ["sar", tile_sar], ["labels", tile_labels]]:
                    image_name = name_file("{}_tile{}_{}".format(date_name, tile_num, each_image[0]), ".npy", out_path)
                    np.save(image_name, each_image[1])
                # Update metadata. 
                generate_metadata(tile_num, date_name, n_water, n_ice, top_left, row_count, tile_count, step_size, tile_size, out_path)

    else:
        for row_count, (row_image, row_labels) in enumerate(zip(image_window, labels_window)):
            for tile_count, (tile_image, tile_labels) in enumerate(zip(row_image, row_labels)):
                tile_num = num_shape * row_count + tile_count
                # Check if the label tile contains any unclassified / no data. Discard them if so.
                if np.amin(tile_labels) == 0:
                    continue
                n_water = np.count_nonzero(tile_labels == 100)
                n_ice = np.count_nonzero(tile_labels == 200)
                # Save the tiles.       
                if modis_path != None:
                    image_name = name_file("{}_tile{}_modis".format(date_name, tile_num), ".npy", out_path)
                    np.save(image_name, tile_image)
                elif sar_path != None:
                    image_name = name_file("{}_tile{}_sar".format(date_name, tile_num), ".npy", out_path)
                    np.save(image_name, tile_image)
                image_name = name_file("{}_tile{}_labels".format(date_name, tile_num), ".npy", out_path)
                np.save(image_name, tile_labels)
                # Update metadata.
                generate_metadata(tile_num, date_name, n_water, n_ice, top_left, row_count, tile_count, step_size, tile_size, out_path)


def tile_prediction_image(image_path, image_type, out_path, tile_size): 
    """Divide image into tiles and save the tiles and their metadata.
       Parameters: image_path: (string) file path of image.
                   image_type: (string) modis or sar.
                   out_path: (string) path to directory to write output.
                   tile_size: (int) number of pixels in length or width of square tile.
    """
    step_size = tile_size
    if image_type == "modis":
        window_shape = (tile_size, tile_size, 3)
    elif image_type == "sar":
        window_shape = (tile_size, tile_size)
    image_window = tif_to_window(image_path, window_shape, step_size)
    image_data = gdal.Open(image_path)
    geography = image_data.GetGeoTransform()
    top_left = geography[0], geography[3]
    image_data.FlushCache()
    del image_data
    num_shape = image_window.shape[1]
    for row_count, (row_image) in enumerate(image_window):
        for tile_count, (tile_image) in enumerate(row_image):
            tile_num = num_shape * row_count + tile_count
            # Save the tiles.            
            image_name = name_file("{}_tile{}_row{}_col{}".format(image_type, tile_num, row_count, tile_count), ".npy", out_path)
            np.save(image_name, tile_image)
            # Update metadata. 
            generate_metadata(tile_num, image_type, 1, 1, top_left, row_count, tile_count, step_size, tile_size, out_path)


def tif_to_window(tif_path, window_shape, step_size):
    Image.MAX_IMAGE_PIXELS = 660000000
    image_tif = Image.open(tif_path)
    image_array = np.asarray(image_tif)
    del image_tif
    image_window = np.lib.stride_tricks.sliding_window_view(x=image_array, window_shape=(window_shape))[::step_size, ::step_size]
    return image_window


def reconstruct_from_tiles(tiles_path):
    os.chdir(tiles_path)
    file_names = os.listdir()
    num_rows, num_cols = 0, 0
    for file_name in file_names:
        if "row0" in file_name:
            num_cols += 1
        if "col0" in file_name:
            num_rows += 1
    row, next_row = 0, 1
    full_array = []
    for row in range(num_rows-1):
        row_array = []
        for col in range(num_cols):
            file_num = (row * (num_cols + 1)) + col + 1
            tile_file = file_names[file_num]
            tile_array = np.load(tile_file)
            tile_array = tile_array[0]
            if len(row_array) == 0:
                row_array = tile_array
            else:
                row_array = np.concatenate((row_array, tile_array), axis=1)
        if len(full_array) == 0:
            full_array = row_array
        else:
            full_array = np.concatenate((full_array, row_array), axis=0)
    mosaic_name = name_file("reconstructed", ".npy", tiles_path)
    np.save(mosaic_name, full_array)

   
        