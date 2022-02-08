from preprocessing.shared import *
import fiona


def clip(shapefile, image_large, image_small, temp_path, out_path):
# Clips images to the same bounds. Had to combine rasterio and gdal to deal with broken shapefiles and bugs in gdal. 
    # Get the bounding box from the polygon file.
    with fiona.open(shapefile, "r") as polygon:
        shapes = [feature["geometry"] for feature in polygon]
    # Clip the large image to the bounding box.
    with rasterio.open(image_large) as large_image:
        out_image, out_transform = rasterio.mask.mask(large_image, shapes, crop=True)
        out_meta = large_image.meta
    out_meta.update({"driver": "GTiff", "transform": out_transform})
    with rasterio.open(temp_path, "w", **out_meta) as destination:
        destination.write(out_image)
    

    # MOVE ALL THIS CODE. See resizing.resize_to_match
    # Resize the clipped large image to match the small image.
    small_image = gdal.Open(image_small)
    width, length = small_image.RasterXSize, small_image.RasterYSize 
    gdal.Translate(out_path, temp_path, width=width, height=length)
    # Clean up.
    small_image.FlushCache()
    del small_image


    os.remove(temp_path)