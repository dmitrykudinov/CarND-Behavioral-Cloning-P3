import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from keras.layers.advanced_activations import LeakyReLU

data_folder = "../linux_sim/collected_data/"
img_folder = data_folder + "IMG/"

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

images = []
measurements = []

#This model was trained on less than 17k data points (50k images) which allows to put
#the entire dataset into physical RAM on a 16Gb desktop machine.
#For bigger size datasets, the recommeded way to load the data in is through
#Generators coroutine well described in 
# https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/b602658e-8a68-44e5-9f0b-dfa746a0cc1a 
for r in data:
    image = cv2.imread(img_folder + r[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(r[3])
    #flipped horizontally
    images.append(np.fliplr(image))
    measurements.append(-r[3])
    #left
    correction = 0.08
    image = cv2.imread(img_folder + r[1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(r[3] + correction)
    #right
    image = cv2.imread(img_folder + r[2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(r[3] - correction)


X_train = np.array(images)
y_train = np.array(measurements)

#Model is inspired by the NVidia architecure 
#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

model = Sequential()
#making relative learning rate smaller by normalizing 
#input color channels to the -16..+16 range
model.add(Cropping2D(cropping=((62,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 8. - 16.))

model.add(Conv2D(24, 5, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(36, 5, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(48, 5, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(64, 3, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Conv2D(64, 3, strides=(2,2)))
model.add(LeakyReLU(alpha=0.01))

model.add(Flatten())
#adding dropout so the model doesnt overfit too much
model.add(Dropout(0.25))

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
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=2)

#saving the result
model.save("model.h5")
