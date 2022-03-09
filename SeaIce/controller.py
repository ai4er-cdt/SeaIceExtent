from preprocessing.data_handling import *
from preprocessing import stitching, resizing, clipping, relabelling, tiling, rebanding
from unet.predict import make_predictions


def preprocess_training(shape_file_path, folder_name, modis_paths = None, sar_path = None, out_path = temp_files, resolution = 40, 
               relabel_from = [0], relabel_to = [0], relabel_scale = 1, tile_size = 512, step_size = 384):
   """
   Handles the sequence of performing all the preprocessing functions.
   Parameters: modis_paths: (list of strings) file paths to adjoining optical images which can be empty if there are none. 
               sar_path: (string) the file path to the radar image, set to None if there isn't one.
               shape_file_path: (string) the file path to the .shp file associated with the images.
               out_path: (string) the directory in which output is to be written. folder_name: the date that the images were collected.
               resolution: (int) desired resolution of modis images. 
               relabel_from and relabel_to: (lists of ints) a pair of lists containing numbers to change labels from and to, respectively. 
               relabel_scale: (int) multiplier for all labelled pixel values. 
               tile_size: (int) number of pixels of the length or width of square tiles.
               step_size: (int) number of pixels to move along.
   Output: Numpy array tiles of whichever images are passed to the function, named by their date and image type, and a metadata entry.           
   """
   try:
       fiona.open(shape_file_path)
   except:
       raise Exception(help(preprocess_training), "Shapefile could not be opened. A .shp file must be provided.")
   if modis_paths == None and sar_path == None:
       raise Exception(help(preprocess_training), "No optical or radar image provided. The file path to least one of these must be supplied.")
   if len(modis_paths) != 0:
        if len(modis_paths) > 1:
            # Stitch the modis images together
            modis_array, modis_metadata = stitching.stitch(modis_paths)
            # Save the full image in temporary folder
            modis_file_path = save_tiff(modis_array, modis_metadata, "stitched", temp_files)
        else:
            modis_file_path = modis_paths[0]
        # Upsample modis image
        upsampled_modis_path = name_file("upsampled", ".tif", temp_files)
        resizing.change_resolution(modis_file_path, upsampled_modis_path, resolution)
        # Clip modis image
        modis_clipped, modis_metadata = clipping.clip(shape_file_path, upsampled_modis_path)
        modis_clipped_path = save_tiff(modis_clipped, modis_metadata, "clipped", temp_files)
        # Resize the modis image to match the sar image
        if sar_path != None:
            resized_modis_path = name_file("resized", ".tif", temp_files)
            resizing.resize_to_match(modis_clipped_path, sar_path, resized_modis_path)
            modis_file_path = resized_modis_path
   else:
        modis_file_path = None

    # These will be applied to either modis or sar, depending on which we have.
    # Relabel. Sar images are faster to work with. 
   labels_path = name_file("labels", ".tif", temp_files) 
   if sar_path != None:
        relabelling.shp_to_tif(shape_file_path, sar_path, labels_path)
   else:
        relabelling.shp_to_tif(shape_file_path, modis_file_path, labels_path)

   relabelling.relabel(labels_path, relabel_from, relabel_to, relabel_scale)

    # Tile    
   tiling.tile_training_images(labels_path, out_path, tile_size, step_size, folder_name, modis_file_path, sar_path)

   delete_temp_files()


def preprocess_prediction(image_path, image_type, resolution, tile_size):
   """Perform preprocessing on new images to be consistent with training.
      Parameters: 
            image_path: (string) file path to image.
            image_type: (string) "sar" or "modis".
            resolution: (int) required resolutin in metres.
            tile_size: (int) height and width of image tiles.
      Output: preprocessed images in temporary folders, and tiles of image. 
   """
   if image_type == "modis" and resolution != None:
           # Alter resolution of modis image
           new_resolution_path = name_file("new_resolution", ".tif", temp_preprocessed)
           resizing.change_resolution(image_path, new_resolution_path, resolution)
           # The image will be too large
           new_size_path = name_file("resized", ".tif", temp_preprocessed)
           resizing.halve_size(new_resolution_path, new_size_path)
           image_path = new_size_path
   # Tile
   tiling.tile_prediction_image(image_path, image_type, temp_tiled, tile_size)


def start_prediction(image_path):
    """Controls pipeline of taking in a new image and passing it through prediction procedure.
       Parameter: image_path: (string) file path to tiff image to be predicted, or path of folder containing tiff images to be predicted.
       Output: two png images next to the original image's location(s); one for classes and one for probabilities. 
    """
    # Check if the provided path is to a folder or an individual image.
    if image_path.endswith(".tif"):
        # Individual image.
        image_paths = [image_path]
        image_path = image_path[::-1]
        folder = image_path.split("\\", 1)
        filename, folder = folder[0], folder[1]
        filename, folder = filename[::-1], folder[::-1]
        image_names = [filename]
    else:
        # Folder containing images.
        folder = image_path
        image_names, image_paths = get_contents(image_path, ".tif", "suffix")
    for i in range(len(image_paths)):
        image = image_paths[i]
        filename = image_names[i]
        filename = filename.split(".")[0]
        image_path = r'{}'.format(image)
        # Find out if the image is modis or sar.
        open_image = gdal.Open(image_path)
        image_type = "modis"
        model = model_modis
        if open_image.RasterCount == 1:
            image_type = "sar"
            model = model_sar
        elif open_image.RasterCount > 3:
            # name rebanded image path.
            image_path = name_file("rebanded", ".tif", folder)
            rebanding.select_bands(open_image, image_path)
        # Tile and do any necessary resizing.
        preprocess_prediction(image_path, image_type, 40, 512)
        # Pass the tiles into the model.
        make_predictions(model, "raw", image_type, temp_tiled, temp_binary, temp_probabilities, save = True)
        # Construct a mosaic of tiles to match the original image.
        out_path = name_file("{}_predicted_classes".format(filename), ".png", folder)
        tiling.reconstruct_from_tiles(temp_binary, out_path)
        out_path = name_file("{}_predicted_probabilities".format(filename), ".png", folder)
        tiling.reconstruct_from_tiles(temp_probabilities, out_path)
        # Clean up.
        delete_temp_files()
        del open_image