import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D, SpatialDropout2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_folder = "../linux_sim/collected_data/"
img_folder = data_folder + "IMG/"
batch_size = 768

#reading CSV with the datapoints line-by-line
data = []
with open(data_folder + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
	if (line[0].find("\\") > -1 ):
	    cntr = line[0].split("\\")[-1]
	    left = line[1].split("\\")[-1]
	    rght = line[2].split("\\")[-1]
	else:
	    cntr = line[0].split("/")[-1]
	    left = line[1].split("/")[-1]
	    rght = line[2].split("/")[-1]

	sa = float(line[3])
	data.append([cntr, left, rght, sa])

def imreadRGB(file_name):
    #cv2 ny default reads images in BGR channels.
    #flipping channels to RGB as expected by the drive.py
    image = cv2.imread(img_folder + file_name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def trimImage(image):
    #trimming the top and bottom to remove unnecessary pixels,
    #which don't much contribute to the useful signal
    return image[62:140,:,:]

def generator(samples, batch_size=32):
    #using a generator so the model can be trained
    #on a large dataset which doesn't fit into physical memory
    correction = 0.08
    while 1:
	samples = shuffle(samples)
	images = []
	angles = []
	for i in range(batch_size):
	    #central camera
	    image = trimImage(imreadRGB(samples[i][0]))
	    angle = samples[i][3]
	    images.append(image)
	    angles.append(angle)
	    #flipped central camera
	    image = np.fliplr(image)
	    angle = -angle
	    images.append(image)
	    angles.append(angle)
	    #left camera
	    image = trimImage(imreadRGB(samples[i][1]))
	    angle = samples[i][3] + correction
	    images.append(image)
	    angles.append(angle)
	    #right camera
	    image = trimImage(imreadRGB(samples[i][2]))
	    angle = samples[i][3] - correction
	    images.append(image)
	    angles.append(angle)
	X = np.array(images)
	y = np.array(angles)
	yield (X, y)

train_samples, valid_samples = train_test_split(data, test_size=0.2)

train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(valid_samples, batch_size=batch_size)

#Model is inspired by the NVidia architecure 
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

model = Sequential()
#making relative learning rate smaller by normalizing 
#input color channels to the -16..+16 range
#model.add(Cropping2D(cropping=((62,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 8. - 16., input_shape=(78,320,3)))

model.add(Conv2D(24, 5, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(36, 5, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(48, 5, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(64, 3, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))
model.add(SpatialDropout2D(0.1))

model.add(Conv2D(64, 3, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))
model.add(SpatialDropout2D(0.1))

model.add(Flatten())
#adding dropout so the model doesnt overfit too much
#model.add(Dropout(0.25))

model.add(Dense(1164))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(50))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(10))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(1))

#Adam optimizer with the default settings
model.compile(loss="mse", optimizer='adam')

#training for 5 epochs, 0.3 validation/train split
#model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=6)
model.fit_generator(train_generator, 
		    steps_per_epoch=len(train_samples) / batch_size,
		    validation_data=valid_generator,
		    validation_steps=len(valid_samples) / batch_size,
		    epochs=20)

#saving the result
model.save("model.h5")
