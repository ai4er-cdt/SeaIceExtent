This file is to document decision making throughout the project.

January 7 - 21, 2022
 - Tile size of 128 x 128 pixels: we had originally picked a tile size of 100 x 100 pixels to create a 4 km by 4 km square for training based on visual inspection of the imagery as well as looking at other papers. This size was increased to accomodate the reduction in size by halving through pooling. 
 - Tile overlap of 25% (32 pixels) was chosen to have some overlap between tiles and creating a manageable tile set size. 
 - Starting our model with SAR imagery because most existing work uses only SAR imagery and because SAR has one band vs three in the MODIS optical imagery (simplicity). 
 - Decided to model with SAR and optical seperately and combine later on, time allowing, to build complexity in the model and gain a more robust understanding of the CNN with SAR before using both data types. We were also doubtful of the reasonableness of having regular matching SAR and optical imagery pairs. Full notes in our Meeting 4 Minutes. 
 - We rasterized the polygon90 data because it was in the correct projection and we could create a raster at 40m (appropriate for use with the SAR imagery). 
 - We decided to remap the classified values to 1 for ice/land, 2 for water and 0 for no data (instead of ice, water, land, no data, and null). We noted that there is ver little classified data for each image pair. 
