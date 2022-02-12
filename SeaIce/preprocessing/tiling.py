from PIL import Image
from preprocessing.data_handling import *


def tile_images(modis_path, sar_path, labels_path, out_path, tile_size, step_size, date_name): 

    has_modis, has_sar = False, False

    window_shape = (tile_size, tile_size)
    labels_window = tif_to_window(labels_path, window_shape, step_size)
    if sar_path != None:
        has_sar = True
        sar_window = tif_to_window(sar_path, window_shape, step_size)
        image_data = gdal.Open(sar_path)

    # Modis has RGB channels so needs a different window shape to the other images.
    if modis_path != None:
        has_modis = True
        window_shape = (tile_size, tile_size, 3)
        modis_window = tif_to_window(modis_path, window_shape, step_size)
        image_data = gdal.Open(modis_path)

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
                    print("Invalid label")
                    continue
                n_water = np.count_nonzero(tile_labels == 100)
                n_ice = np.count_nonzero(tile_labels == 200)
                # Save the tiles.        
                modis_name = "{}_tile{}_modis.npy".format(out_path, tile_num)
                np.save(modis_name, tile_modis)
                
                sar_name = "{}_tile{}_sar.npy".format(out_path, tile_num)
                np.save(sar_name, tile_sar)

                labels_name = "{}_tile{}_labels.npy".format(out_path, tile_num)    
                np.save(labels_name, tile_labels)
                # Update metadata.
                generate_metadata(out_path, tile_num, date_name, n_water, n_ice, top_left, row_count, tile_count, step_size, tile_size)

    else:
        try:
            image_window = modis_window
        except:
            image_window = sar_window
        for row_count, (row_image, row_labels) in enumerate(zip(image_window, labels_window)):
            for tile_count, (tile_image, tile_labels) in enumerate(zip(row_image, row_labels)):
                tile_num = num_shape * row_count + tile_count
                print(tile_num)
                # Check if the label tile contains any unclassified / no data. Discard them if so.
                if np.amin(tile_labels) == 0:
                    continue
                n_water = np.count_nonzero(tile_labels == 100)
                n_ice = np.count_nonzero(tile_labels == 200)
                # Save the tiles.       
                if modis_path != None:
                    modis_name = "{}_tile{}_modis.npy".format(out_path, tile_num)
                    np.save(modis_name, tile_image)
                elif sar_path != None:
                    sar_name = "{}_tile{}_sar.npy".format(out_path, tile_num)
                    np.save(sar_name, tile_image)

                labels_name = "{}_tile{}_labels.npy".format(out_path, tile_num)   
                np.save(labels_name, tile_labels)
                # Update metadata.
                generate_metadata(out_path, tile_num, date_name, n_water, n_ice, top_left, row_count, tile_count, step_size, tile_size)


def tif_to_window(tif_path, window_shape, step_size):
    image_tif = Image.open(tif_path)
    image_array = np.asarray(image_tif)
    del image_tif
    image_window = np.lib.stride_tricks.sliding_window_view(x=image_array, window_shape=(window_shape))[::step_size, ::step_size]
    return image_window

