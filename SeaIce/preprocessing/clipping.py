from preprocessing.shared import *
import rasterio.mask
import fiona


def clip(shapefile, image):
# Clips images to the same bounds. Had to combine rasterio and gdal to deal with broken shapefiles and bugs in gdal. 
    # Get the bounding box from the polygon file.
    with fiona.open(shapefile, "r") as polygon:
        shapes = [feature["geometry"] for feature in polygon]
    # Clip the large image to the bounding box.
    with rasterio.open(image) as large_image:
        out_image, out_transform = rasterio.mask.mask(large_image, shapes, crop=True)
        out_meta = large_image.meta
    out_meta.update({"driver": "GTiff", "transform": out_transform})
    return out_image, out_meta