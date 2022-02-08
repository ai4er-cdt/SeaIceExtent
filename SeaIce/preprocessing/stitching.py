from preprocessing.shared import *
import rasterio.merge


def stitch(image_portions):
    # Stick images together.
    open_portions = []
    for portion in image_portions:
        opened = rasterio.open(portion) 
        open_portions.append(opened)
        # Not sure if this next line will work... remove if not!
        opened.close()
    full_image, transform = rasterio.merge.merge(open_portions) 
    # New metadata.
    out_meta = opened.meta.copy()
    out_meta.update({"driver": "GTiff", "height": full_image.shape[1], 
                     "width": full_image.shape[2], "transform": transform})
    return full_image, out_meta
