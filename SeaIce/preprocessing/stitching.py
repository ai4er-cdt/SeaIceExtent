"""This is the stitching module for the Sea Ice Extent GTC Project data preprocessing step.

This module contains a function to stitch together two modis images and creates new metadata.
"""

import rasterio.merge


def stitch(image_portions):
    """Stick images together.
       Parameters: image_portions: (list of strings) all the images to be joined to form a mosaic.
       Returns: full_image: open rasterio raster of joined images. out_meta: metadata for the new image.
    """
    open_portions = []
    for portion in image_portions:
        opened = rasterio.open(portion) 
        open_portions.append(opened)
    full_image, transform = rasterio.merge.merge(open_portions) 
    # New metadata.
    out_meta = opened.meta.copy()
    out_meta.update({"driver": "GTiff", "height": full_image.shape[1], 
                     "width": full_image.shape[2], "transform": transform})
    return full_image, out_meta
