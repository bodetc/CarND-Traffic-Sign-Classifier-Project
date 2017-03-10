#**Traffic Sign Recognition** 

[//]: # (Image References)

[histogram]: ./writeup/histogram.png "Histogram of the classes"
[before]: ./writeup/before.png "Before preprocessing"
[y]: ./writeup/y.png "YUV-image"
[u]: ./writeup/u.png "YUV-image"
[v]: ./writeup/v.png "YUV-image"
[after]: ./writeup/after.png "After preprocessing"
[23205]: ./writeup/yield/23205.png "Yield"
[23206]: ./writeup/yield/23206.png "Yield"
[23207]: ./writeup/yield/23207.png "Yield"
[23208]: ./writeup/yield/23208.png "Yield"
[23209]: ./writeup/yield/23209.png "Yield"
[lenet]: ./writeup/lenet.png "LeNet"
[test1full]: ./test_images/GRMN0001.JPG "Test Image"
[test2full]: ./test_images/GRMN0004.JPG "Test Image"
[test3full]: ./test_images/GRMN0006.JPG "Test Image"
[test4full]: ./test_images/GRMN0016.JPG "Test Image"
[test5full]: ./test_images/GRMN0048.JPG "Test Image"
[test1]: ./test_images/test1.jpg "Test Image"
[test2]: ./test_images/test2.jpg "Test Image"
[test3]: ./test_images/test3.jpg "Test Image"
[test4]: ./test_images/test4.jpg "Test Image"
[test5]: ./test_images/test5.jpg "Test Image"

###Writeup / README

This is my written report for the second project of the first term of the self-driving car nanodegree. 
The code for this project can also be found on my [GitHub](https://github.com/bodetc/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) account.


The goal of this project is to build a classifier capable of categorizing German traffic sign.
After a short exploration of the dataset,
I will discuss the design, training and testing the model architecture.
After, the architecture will be tested with five new images.
The results of those new images will be discussed in more details.

###Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 pixels
* They are 43 unique classes/labels in the data set 

#### Exploratory visualization

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.
It is an histogram showing the number of occurrence of each class in the training dataset.


![alt text][histogram]

The classes do not have an even rate of occurrence.
It appears that the occurrence rate of each sign follows real-life distribution of those traffic sign.
For instance, the "Keep right (38)" sign has much more occurrence than the "Keep left (39)" sign.
As Germany is a right-hand drive country, the "Keep right (38)" is common on median dividers and therefore occurs more often than its "Keep left (39)" counterpart.
Other common traffic signs include speed limitations, priority regulation sign, and caution signs.

Furthermore, I exported each image of the dataset to the filesystem and visually explored them.
The images are still taken from onboard video, and several consecutive picture can be from the same video clip and from the same physical traffic sign.
This emphasize the need to shuffle the training data, as to avoid unwanted correlations.

![alt text][23205]
![alt text][23206]
![alt text][23207]
![alt text][23208]
![alt text][23209]

###Design and Test a Model Architecture

#### Preprocessing

The code for this step is contained in the fourth to sixth code cell of the IPython notebook.

As a first step, I decided to convert the images to YUV in order to separate the luminosity of the image from its luminosity.
I did not convert to grayscale as the color is an important factor for identifying traffic sign, and the the type of traffic sign.
As all driving school student must learn, blue means obligation or indication and red means interdiction or danger. 
Blue and red colors are located on opposite corners of the UV square. 

The Y channel was normalized by applying min-max normalization on each image.
The values of the remaining two channels were then divided by 255 to improve the numerical stability of the algorithm.

Here is an example of a traffic sign image before preprocessing, then each channel of the YUV transformation, and finally the normalized image converted back to RGB.

![alt text][before]
![alt text][y]
![alt text][u]
![alt text][v]
![alt text][after]

#### Training, validation and testing data.

The training, validation and testing sets used correspond to the three datasets provided [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).
Those sets contain 34799, 4410 and 12630 elements, respectively.

#### Architecture and training

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV normalized image   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride, outputs 14x14x6    				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride, outputs 5x5x16    				|
| Flattening            | Converts 5x5x16 into 400 linear layer         |
| Fully connected		| Input 400, output 120 						|
| RELU					| Activation									|
| Dropout				| Keep probability for training: 80%			|
| Fully connected		| Input 120, output 84   						|
| RELU					| Activation									|
| Dropout				| Keep probability for training: 80%			|
| Fully connected		| Input 84, output 43   						|

This architecture is based on the LeNet architecture introduced in the LeNet lab:
![alt text][lenet]

The architecture was adapted for this project.
The input layer has three color channels instead of one, 
and the final output was enlarge to 43 in order to fit the number of classes.
Furthermore, in order to prevent overfitting two dropout layers were added after the fully connected layers (C5 and F6 in the picture above).

The code for training the model is located in the eight cell of the ipython notebook.
The structure of code code is the same as what presented in the previous Labs of this nanodegree.
The model is trained using an Adam optimizer (Adaptive moment estimation). The hyperparameters of the optimizer are as follows:

| Hyperparameter   		| Value         	        					| 
|:---------------------:|:---------------------------------------------:| 
| Batch size       		| 4096                             				|
| Learning rate    		| 0.006                            				|
| Number of epochs 		| 25                             				|

#### Results

The code for calculating the accuracy of the model on the test set is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 98.6%
* validation set accuracy of 91.8%
* test set accuracy of 90.3%

The LeNet architecture was chosen as a basis for this project.
It was chosen 
Furthermore, a similar model was used with success in [this article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by P. Sermanet and Y. LeCun.
The article was considering the dataset of the GTSRB competition, which was is also the basis of the dataset of the current project.
The architecture presented in the article is more advance than the LeNet architecture, but is is close enough to consider the LeNet architecture as a base for this project.

After manual adjustments to the architecture, I decided to add two dropout layer on the fully connected layers in order to avoid overfitting.
A keep probability of 80% was chosen after manual tuning as it provides a good balance between preventing overfitting and loosing too much accuracy.
Other adjustments were tried, such as removing layers or changing their sizes, but they did not lead to significant increase of the accuracy.
They are therefore not included in the final architecture.

The final accuracy of the model on the training, validation are all over 90%, indicating that the model is working well.

###Test a Model on New Images

For this section, I used five German traffic sign pictures that I capture using the dashcam mounted on my car.
Those pictures were taken during my daily driving in the vicinity of Munich.
The full picture are attached to this report.
I would appreciate if they are not published as German privacy laws are very strict, and not yet well defined regarding the usage of dashcams. 

![alt text][test1full]
![alt text][test2full]
![alt text][test3full]
![alt text][test4full]
![alt text][test5full]

I manually cropped and resized those image to 32x32 pixels. The resulting images to be fed to the classifier are:

![alt text][test1] ![alt text][test2] ![alt text][test3] 
![alt text][test4] ![alt text][test5]

The code for loading the new images is located in the tenth cell of the Ipython notebook.

The first image is expected to be easy for the model to classify.
The yield sign was specifically designed to be easily recognisable even in difficult situations.
Its shape is unique (triangle on its tip), so that it can be recognized even if covered in snow.

The second and third images are speed limits, and present the challenge of having to parse the number sitting inside the traffic sign.
The fourth image might present difficulties as it was taken at night, and present a blue sign on a blue background.
The fifth image could pose a problem as several sticker are present on the traffic sign.

#### Predictions

The code for making predictions on my final model is located in the eleventh cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield           		| Yield   			    						| 
| Speed limit 70 km/h	| Speed limit 50 km/h							|
| Speed limit 80 km/h	| Speed limit 80 km/h							|
| Turn left only		| Turn left only								|
| Keep right      		| Keep right					 				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
This is lower than the accuracy on the test set of 90.5%, but still in line with it.
The number of new datapoint is too low to make any meaningful statistical comparison.

#### Softmax
The code for making looking at the softmax probabilities is located in the 11th cell of the Ipython notebook.

##### First image

For the first image, the model is certain that the image is a yield sign.
The rounded probability is 1, and the other softmax probability are negligible (below 10<sup>-8</sup>).
This confirms the previous supposition that yield traffic signs are easy to recognize.

##### Second image

The model wrongly predicts that the second image is a 'Speed limit 50 km/h' with a low certainty of around 50%.
The correct answer is only third, with a softmax probability of 20.7%.
The model is however confident that it is a speed limitation, as all the prediction in the top five are speed limits.

The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50.5%        			| Speed limit 50 km/h   						| 
| 21.2%        			| Speed limit 30 km/h   						| 
| 20.7%        			| Speed limit 70 km/h   						| 
| 2.7%        			| Speed limit 120 km/h   						| 
| 2.0%        			| Speed limit 100 km/h   						| 

##### Third image

The next image is also a speed limit traffic sign.
The sign is correctly recognized by the model with high certainty of 96%.
The following three most probable predictions are also speed limitations.

The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 96.0%        			| Speed limit 80 km/h   						| 
| 2.8%        			| Speed limit 60 km/h   						| 
| 0.7%        			| Speed limit 100 km/h   						| 
| 0.1%        			| Speed limit 30 km/h   						| 
| 0.1%        			| Wild animals crossing   						|
 
 ##### Fourth image
 
 The following image is correctly recognized as a 'Turn left only' traffic sign.
 The model give a certainty of 99.8% for its prediction.
 
 The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.8%        			| Turn left only   						        | 
| 0.2%        			| Keep right   						            | 
| 7.4 10<sup>-6</sup>	| Go straight or right   						| 
| 6.8 10<sup>-6</sup>	| Ahead only   			            			| 
| 7.6 10<sup>-7</sup>	| End of all limitations   						| 
 
 ##### Fifth image
 
 For the fifth image, the mode is also very certain that the image is a 'keep right' with an uncertainty in the order of 10<sup>-6</sup>.
 Three of the four remaining predictions are also obligation sign, but all have very small probability.
 This indicate that the model was not disturbed by the stickers present on the sign. 