# Some of this code has been copied from here:
# https://gis.stackexchange.com/questions/190724/remove-subset-raster-bands-in-python-gdal

from preprocessing.data_handling import *
import struct
from osgeo import osr


def select_bands(open_image, out_path):
    """Copies light bands 3, 6 and 7 to a new image so that extra bands are not included.
       Parameters: 
            open_image: (gdal opened image) the image which contains extra bands.
            out_path: (string) the file path of the new image to write.
       Output: A new image.
    """
    numerical_sizes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}

    #Get projection
    prj = open_image.GetProjection()

    #finding the right bands
    band3 = open_image.GetRasterBand(3)
    band6 = open_image.GetRasterBand(6)
    band7 = open_image.GetRasterBand(7)

    geotransform = open_image.GetGeoTransform()

    # Create gtif file with rows and columns from parent raster 
    driver = gdal.GetDriverByName("GTiff")

    columns, rows = (band3.XSize, band3.YSize)

    BandType = gdal.GetDataTypeName(band3.DataType)

    new_band_1, new_band_2, new_band_3 = [], [], []
    bands = [band3, band6, band7]
    new_raster = [new_band_1, new_band_2, new_band_3]

    for i in range(3):
        band = bands[i]
        raster = new_raster[i]
        for y in range(band.YSize):

            scanline = band.ReadRaster(0, y, band.XSize, 1, band.XSize, 1, band.DataType)
            values = struct.unpack(numerical_sizes[BandType] * band.XSize, scanline)
            raster.append(values)

        #flattened list of raster values
        raster = [ item for element in raster for item in element ]

        #transforming list in array
        raster = np.asarray(np.reshape(raster, (rows,columns)))

    dst_ds = driver.Create(out_path, columns, rows, 1, band3.DataType)    

    ##writting output raster
    dst_ds.GetRasterBand(1).WriteArray(new_band_1)
    dst_ds.GetRasterBand(2).WriteArray(new_band_2)
    dst_ds.GetRasterBand(3).WriteArray(new_band_3)

    #setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(geotransform)

    # setting spatial reference of output raster 
    srs = osr.SpatialReference(wkt = prj)
    dst_ds.SetProjection( srs.ExportToWkt() )

    #Close output raster dataset 
    dst_ds = None

    #Close main raster dataset
    open_image = None