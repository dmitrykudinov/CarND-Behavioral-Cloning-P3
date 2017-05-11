# Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* ./success/1/model.py containing the script to create and train the model
* ./success/1/drive.py for driving the car in autonomous mode
* ./success/1/model.h5 containing a trained convolution neural network 
* ./success/1/output.mp4 video of a successful run around the FIRST track tracks
* ./success/2/model.py containing the script to create and train the model
* ./success/2/drive.py for driving the car in autonomous mode
* ./success/2/model.h5 containing a trained convolution neural network 
* ./success/2/output.mp4 video of a successful run around the SECOND track tracks
* ./README.md this writeup summarizing the results

Additional model to illustrate the note on image quality down below
* ./success/1-dirt-road/model.py
* ./success/1-dirt-road/model.h5
* ./success/1-dirt-road/output.mp4 video of the 1 track lap which includes portion of the dirt road shortcut

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model & Training: overview

#### 1. An appropriate model architecture has been employed

My model architecture was inspired by the NVidia architecture described in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ and consists of five convolutional layers of 5x5 and 3x3 filter sizes, depths between 24 and 64, all with RELU activations. 

The fully connected layers mimic the above NVidia architecture with the only difference, that I added a Dropout with probability of 0.2 to prevent too much overfitting.

The model includes data normalization using a Keras lambda layer, where color channels are squeezed into the -16..+16 range to keep the default starting learning rate of Adam optimizer small compared to the input data. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting with probability of 0.2. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Other hyperparameters which have been tuned:
* side cameras correction angle
* cropping frame
* drop-out rate
* normalization range

#### 4. Appropriate training data

I have made a several attempts to collect the data, starting small and asking the model to generalize / guess the best course of action. At some point I got an interesting result on the first track with the rendering details set to the "Fantastic" level but very few training data.

For details about how I created the training data, see the next section. 

### Model & Training: details

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model, then adding convolutional layers one by one and gradually increasing the number of neurons in the fully connected layers watching the validation loss going down.

My first step was to use a convolutional neural network model similar to the AlexNet, then VGG, then the NVidia architecture. The latter one appeared to be most promising to me because of the simplicity, speed of training, and loss on validation set.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that right after the last convolutional layer result was flattened, a Dropout layer with 0.2 probability was fit in before the result goes to the fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I made additional laps in the opposite direction (1 - 2 total), and specifically drove around the problematic spots for a couple times each.

I also used the images from the side cameras, flipped horizontally central camera, playing with the `correction` parameter values, trying to find which ones yield best loss value.

Another hyper-parameter which I was tuning - clipping of the input images which was also a quite useful tool to reduce the amount of unnecessary information.

During the experiments, I also played with the various settings of the Simulator Renderer quality: at some point I set the quality to "Fantastic" level and drove only one single lap in the opposite direction on the 1st track. The result was quite surprising: the car was able to successfully complete the lap! The only thing is that it found a legitimate shortcut through a dirt road which it took, and then successfully returned back to the track. 
Just to demonstrate this case, I am attaching the model and the mp4 file in ./success/1-dirt-road/ folder.

After adding a lap driven in the "correct" direction, and a couple extra turns to the training set, the neural network was able to complete a full lap on the first track.

"Fantastic" quality though made things worse on the second track: "Fantastic" adds very dark shadows casted by the mountainous relief, which immediately throws off the model's predictions. A good option to cope with the dark shadows could be some local image normalization with a small kernel size. I didn't try it here as I could not find an easy way to do it in Keras, otherwise it would require some modifications to the drive.py so it does the same preprocessing before feeding the data into the model during inference...

So, for the second track, to make the model complete the lap, I used the "Fastest" quality recommended by the course materials which doesn't have shadows drawn.
Second track successful lap can be seen here: ./success/2/


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| layer | size | activation |
| --- | --- | --- |
| Lambda | | |
| Cropping2D | ((62,20),(0,0)) | |
| Conv2D | 24x5x5 | ReLu|
| Conv2D | 36x5x5 | ReLu|
| Conv2D | 48x5x5 | ReLu|
| Conv2D | 64x3x3 | ReLu|
| Conv2D | 64x3x3 | ReLu|
| Flatten | | |
| Dropout | 0.2 | |
| Dense | 1164 | |
| Dense | 100 | |
| Dense | 50 | |
| Dense | 10 | |
| Dense | 1 | |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from unexpected positions.
Then I repeated this process on track two in order to get more data points.

To augment the dataset, I also flipped images and angles, and used side cameras thinking that this would increase the size of the training dataset and make the cases with fast turning better represented.

After the collection process and data augmentation, I had about 50k of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by stabilized validation_loss after the 3rd-4th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
