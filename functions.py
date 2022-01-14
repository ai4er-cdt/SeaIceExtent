from osgeo import ogr, gdal


def shp2tif(shape_file, sar_raster, output_raster_name):
    """GTC Code to rasterise an input shapefile. Requires as inputs: shapefile, reference tiff, output raster name.
Adapted from: https://opensourceoptions.com/blog/use-python-to-convert-polygons-to-raster-with-gdal-rasterizelayer/
Requires 'ogr' and 'gdal' packages from the 'osgeo' library.
    """

    # Note: throughout ds = 'dataset'

    # Reading in the template raster data (i.e. the SAR tiff).
    sar_raster_ds = gdal.Open(sar_raster)

    # Reading in the shapefile data (should use the 'polygon90.shp' for this).
    shape_file_ds = ogr.Open(shape_file)
    sf_layer = shape_file_ds.GetLayer()

    # Getting the geo metadata from the SAR tiff.
    sar_metadata = sar_raster_ds.GetGeoTransform()

    # Setting the new tiff driver to be a geotiff.
    new_tiff_driver = gdal.GetDriverByName("GTiff")

    # Creating the new tiff from the driver and size from the SAR tiff.
    output_raster_ds = new_tiff_driver.Create(utf8_path=output_raster_name, xsize=sar_raster_ds.RasterXSize,
                                              ysize=sar_raster_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    # Setting the output raster to have the same metadata as the SAR raster.
    output_raster_ds.SetGeoTransform(sar_metadata)

    # Rasterising the shapefile polygons and adding to the new raster.
    # Inputs: new raster, band number of new raster (i.e. the only band), the polygon layer, attribute (preserves class
    # labels given as 'type'.
    gdal.RasterizeLayer(output_raster_ds, [1], sf_layer, options=['ATTRIBUTE=type'])  # , outputType=gdal.GDT_Byte, burnValues=255)

    output_raster_ds.GetRasterBand(1).SetNoDataValue(0.0)
    # Not entirely sure what this does, but it's needed.
    output_raster_ds = None


