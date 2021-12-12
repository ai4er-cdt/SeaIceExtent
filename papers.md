Notes on papers read to inform the sea ice group team challenge. 
Format: please use citation in APA format and place alphabetically by author's last name in file. Use a bullet point with initials to add comments of take-aways from paper.



Boulze, Hugo, Anton Korosov, and Julien Brajard. 2020. "Classification of Sea Ice Types in Sentinel-1 SAR Data Using Convolutional Neural Networks" Remote Sensing 12, no. 13: 2165. https://doi.org/10.3390/rs12132165
* MCL: used CNN to classify for multiple ice ages and water with greatest uncertainty with young/new ice and water boundary - this could justify keeping the land/ice distinction in the original shapefiles BUT only using pixels with x pixels of boarder distinction (i.e. boundary buffering). Found that CNN is less sensitive to noise - can be jused to justify not treating Sentinel-1 product with filter for speckle. But does use denoising as done by Park (2019). Validates using measurements in two directions. Validates using additional random speckle filter with gaussian noise. 
CNN input data was of shape N×K×K×2 where N is number of samples, K is dimension of 50 pixels and 2 is for the hv and hh bands. Used training labeling with one-hot encoding.  Then used nearest neighbor to deal with single pixel anomalies in classifying. Uses 70-30 train test split. 
“CNN is composed of 2 batch-norm layers, 3 convolutional layers, 2 max-pooling layers, 3 hidden dense layers, 4 dropout layers (used only for the training) and one output layer.” I'm not sure what the CNN description means right now but wanted to note. 

Murashkin, D., Spreen, G., Huntemann, M., & Dierking, W. (2018). Method for detection of leads from Sentinel-1 SAR images. Annals of Glaciology, 59(76pt2), 124-136. doi:10.1017/aog.2018.6
* MCL: Describes formerly accepted approach of using a random forest or SVM for classification of the sea ice. Problematizes the observation angle of the satellite and the co-polarization reading and describes a method for correction of angle of incidence: corrected backscatter  = backscatter + 0.49*(incidence angle - min(incidence angle)). Used in Wang and Li (2021)


Park, J. W., Korosov, A. A., Babiker, M., Sandven, S., & Won, J. S. (2017). Efficient thermal noise removal for Sentinel-1 TOPSAR cross-polarization channel. IEEE Transactions on Geoscience and Remote Sensing, 56(3), 1555-1565.
* MCL: proposal for further denoising of Sentinel-1 using azimuth descalloping, noise scaling and interswath power balancing, ad local residual noise power compensation. This describes the method used by Boulze et al (2020). 

Tom, M., Kälin, U., Sütterlin, M., Baltsavias, E., & Schindler, K. (2018). Lake ice detection in low-resolution optical satellite images. International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 4, 279-286.
* MCL: The MODIS bands kept/considered for ice detection at the beginning of the work were B1-B4, B6, B17-B20, B22, B23, B25. Because the lakes of interest were so small, the data was resampled and reprojected into the local UTM32N coordinates. This article problematizes the confusion between snow/ice and cloud cover in MODIS products, and uses the MODIS binary cloud-mask and combines the cloudy and uncertain clear classifications to create a conservative masking approach. Used xgboost to perform supervised variable importance analysis. Found that B2 alone or B2 and B22 together worked well.


Wang, Y.-R. and Li, X.-M.: Arctic sea ice cover data from spaceborne synthetic aperture radar by deep learning, Earth Syst. Sci. Data, 13, 2723–2742, https://doi.org/10.5194/essd-13-2723-2021, 2021.
* MCL: Offers justification to use cross polarization instead of co polarization if only using one band. Uses Sun and Li (2020) method for thermal noise reduction and de-scalloping. Downsamples to 400m pixels to shrink file size/make more manageable. Resales HV cross polarization to 0-255 like with optical bands and discards the 2% extremas at both ends of spectrum. Used U-net architecture. Reinforces prior observations that very smooth sea ice can have a backscatter similar to the ocean, and high wind sea surface can have backscatter similar to sea ice. 

