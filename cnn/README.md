# Notes on CNN

For me, it´s not difficult to understand what´s going on in CNN, but to get to know what we want to do in this exercise as I know little about computer vision or picture processing.

Thus, I´d like to write this note to remind myself as well as for references to others who might occur into the same situation.

Ok, 
* here we treat a color image as a 3d matrix, (imageChannels,row,col), where imageChannels refers to the R,G,B.
* convolvedFeature is a 4D matrix, (numFeatures,numImage,imageDim,imageDim),
where numFeatures is the 400 features we extracted from last exercise.
* In the cnnConvolve function, the *feature* here refers to the *weights* extracted from last exercise.
* **No** sigmoid function provided in cnnConvolved.m! Be careful.
* I run the codes on a PC with 2GB memory and the out-of-memory comes out.
