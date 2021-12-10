Notes on papers read to inform the sea ice group team challenge. 
Format: please use citation in APA format and place alphabetically by author's last name in file. Use a bullet point with initials to add comments of take-aways from paper.



Boulze, Hugo, Anton Korosov, and Julien Brajard. 2020. "Classification of Sea Ice Types in Sentinel-1 SAR Data Using Convolutional Neural Networks" Remote Sensing 12, no. 13: 2165. https://doi.org/10.3390/rs12132165
* MCL: used CNN to classify for multiple ice ages and water with greatest uncertainty with young/new ice and water boundary - this could justify keeping the land/ice distinction in the original shapefiles BUT only using pixels with x pixels of boarder distinction (i.e. boundary buffering). Found that CNN is less sensitive to noise - can be jused to justify not treating Sentinel-1 product with filter for speckle. But does use denoising as done by Park (2019). Validates using measurements in two directions. Validates using additional random speckle filter with gaussian noise. 
CNN input data was of shape N×K×K×2 where N is number of samples, K is dimension of 50 pixels and 2 is for the hv and hh bands. Used training labeling with one-hot encoding.  Then used nearest neighbor to deal with single pixel anomalies in classifying. Uses 70-30 train test split. 
“CNN is composed of 2 batch-norm layers, 3 convolutional layers, 2 max-pooling layers, 3 hidden dense layers, 4 dropout layers (used only for the training) and one output layer.” I'm not sure what the CNN description means right now but wanted to note. 

Park, J. W., Korosov, A. A., Babiker, M., Sandven, S., & Won, J. S. (2017). Efficient thermal noise removal for Sentinel-1 TOPSAR cross-polarization channel. IEEE Transactions on Geoscience and Remote Sensing, 56(3), 1555-1565.
* MCL: proposal for further denoising of Sentinel-1 using azimuth descalloping, noise scaling and interswath power balancing, ad local residual noise power compensation. Method used by Boulze et al (2020). 

Wang, Y.-R. and Li, X.-M.: Arctic sea ice cover data from spaceborne synthetic aperture radar by deep learning, Earth Syst. Sci. Data, 13, 2723–2742, https://doi.org/10.5194/essd-13-2723-2021, 2021.
* MCL: Use cross polarization instead of co polarization if only using one band. 

