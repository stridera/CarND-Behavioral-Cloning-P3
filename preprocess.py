#!/usr/bin/env python

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2, csv, numpy as np

def crop(x):
	h, w, c = x.shape
	crop_size = int(h * 0.40)
	x = x[crop_size:h-crop_size-20, 0:w] 
	return cv2.resize(x, (w, h), interpolation = cv2.INTER_AREA)

def preprocess(x):
	x = crop(x)	
	h, w, c = x.shape

	pnt = int(w*0.15)
	src = np.array([
		[pnt, 0],
		[w-pnt, 0],
		[w, h],
		[0, h]
	], dtype = "float32")
 
	dst = np.array([
		[0, 0],
		[w, 0],
		[w, h],
		[0, h]
	], dtype = "float32") 

	# calculate the perspective transform matrix and warp
	# the perspective to grab the screen
	M = cv2.getPerspectiveTransform(src, dst)
	warp = cv2.warpPerspective(x, M, (w, h))

	return warp

# load data
lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for i, line in enumerate(lines):
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = 'data/IMG/' + filename
	image = cv2.imread(current_path)
	measurement = float(line[3])
	# if not i % 2:
	# 	image = np.fliplr(image)
	# 	measurement = -measurement
	images.append(image)
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# define data preparation
datagen = ImageDataGenerator(
	preprocessing_function=preprocess, 
)
image = np.vstack((crop(X_train[0]), preprocess(X_train[0])))
imgsize = image.shape
print('imagesize:', imgsize)
# pyplot.imshow(image)
# pyplot.show()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vid = cv2.VideoWriter("preprocessing.avi", fourcc, 20.0, (imgsize[0], imgsize[1]))
for i, img in enumerate(X_train):
	cropped = crop(img)
	proc = preprocess(img)
	stacked = np.vstack((cropped, proc))
	cv2.imshow('frame', stacked)
	vid.write(stacked)

vid.release()
# vid.write_videofile("preprocessing.mp4", audio=false)

# # fit parameters from data
# datagen.fit(X_train)

# # configure batch size and retrieve one batch of images
# for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# # create a grid of 3x3 images
	# for i in range(0, 9):
	# 	pyplot.subplot(330 + 1 + i)
	# 	# pyplot.imshow(X_train[i])
	# 	pyplot.imshow(X_batch[i])
	# # show the plot
	# pyplot.show()
	# break