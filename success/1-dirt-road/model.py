import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D

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

for r in data:
    image = cv2.imread(img_folder + r[0])
    images.append(image)
    measurements.append(r[3])
    #flipped horizontally
    images.append(np.fliplr(image))
    measurements.append(-r[3])
    #left
    correction = 0.1
    image = cv2.imread(img_folder + r[1])
    images.append(image)
    measurements.append(r[3] + correction)
    #right
    image = cv2.imread(img_folder + r[2])
    images.append(image)
    measurements.append(r[3] - correction)


X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Conv2D(24, 5, strides=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(36, 5, strides=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(48, 5, strides=(2,2), activation='relu'))
model.add(Conv2D(64, 3, strides=(2,2), activation='relu'))
model.add(Conv2D(64, 3, strides=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=2)

model.save("model.h5")
