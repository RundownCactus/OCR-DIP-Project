import os
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import time
import cv2
import math
import matplotlib.pyplot as plt
import h5py

# import tensorflow as tf

from keras.models import load_model

import csv
import pickle

from skimage.feature import hog

from sklearn.svm import LinearSVC, SVC

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

from matplotlib import pyplot as plt

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

## Function for curating dataset
def files():
    letterPath = "C:\\Users\\numan98khan\\Desktop\\images\\ch2_training_localization_transcription_gt"
    imagePath = "C:\\Users\\numan98khan\\Desktop\\solving_captchas_code_examples\\generated_captcha_images"

    listImg = os.listdir(imagePath)
    listLabel = listImg.copy()
    
    for i in range(0, len(listLabel)):
        listLabel[i] = listLabel[i][:4]
        # print(listLabel[i][:4])
        
    print(listLabel)

    print(listImg)

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File("train_data.h5", mode='w')
    hdf5_file.create_dataset("train_img", (len(listImg)*4, 20, 20, 1), np.uint8)

    # hdf5_file.create_dataset("train_img_hog", (noOfImages, 784), np.float64)

    hdf5_file.create_dataset('train_lab', (len(listImg)*4, 1), np.int8)

    print(len(listImg))
    # print(listLabel)

    imgCounter = 0
    for imgNo in range(0, len(listImg)):
        image = cv2.imread(imagePath+"\\"+listImg[imgNo])
        print(imagePath+"\\"+listImg[imgNo])
        # print(image)
        orig = image.copy()
        (rows, cols) = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Adding some extra padding around the image
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # applying threshold
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

        # creating empty list for holding the coordinates of the letters
        letter_image_regions = []
        
        # finding the contours
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # checking if any counter is too wide
            # if countour is too wide then there could be two letters joined together or are very close to each other
            if w / h > 1.25:
                # Split it in half into two letter regions
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:  
                letter_image_regions.append((x, y, w, h))

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

            # Save out each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions, listLabel[imgNo]):
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

            # Get the folder to save the image in
            # save_path = os.path.join(OUTPUT_FOLDER, letter_text)

            # print("letter", letter_text)
            # print(letter_image.shape)

            # cv2.imshow("Text Detection", letter_image)
            # cv2.waitKey(0)

            # letter_image = np.array(letter_image, dtype=np.uint8)
            # print(letter_image)

            # letter_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)

            # Resize the letter so it fits in a 20x20 pixel box
            image = resize_to_fit(letter_image, 20, 20)

            # Add a third channel dimension to the image to make Keras happy
            image = np.expand_dims(image, axis=2)

            # print(image)
            # Grab the name of the letter based on the folder it was in
            # label = image_file.split(os.path.sep)[-2]

            # continue

            hdf5_file["train_img"][imgCounter, ...] = image[None]
            hdf5_file["train_lab"][imgCounter, ...] = ord(letter_text)

            imgCounter += 1
        
# files()       

# exit(1)

## Function for getting the whole curated dataset
def getData(path, sample):
    # open the hdf5 file
    hdf5_file = h5py.File(path, "r")

    images = hdf5_file["train_img"]
    labels = hdf5_file["train_lab"]
    # hogs = hdf5_file["train_img_hog"]

    if (sample):
        data = images[0]
        print(labels[0])
        cv2.imshow("Text Detection", data)
        cv2.waitKey(0)

    return images, labels

def train():
    hdf5_path = "train_data.h5"
    images, labels = getData(hdf5_path, False)

    test = 0

    print(images[test])

    # print(chr(labels[test]))
    # cv2.imshow("Text Detection", images[test])
    # cv2.waitKey(0)

    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)

    MODEL_FILENAME = "model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"

    # Split the training data into separate train and test sets
    (X_train, X_test, Y_train, Y_test) = train_test_split(images, labels, test_size=0.25, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    # Build the neural network!
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # Output layer with 32 nodes (one for each possible letter/number we predict)
    model.add(Dense(33, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the neural network
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=2, verbose=1)

    # Save the trained model to disk
    model.save(MODEL_FILENAME)

def getBoxes(imageName, padding_crop, showImage):
	# load
	image = cv2.imread(imageName)
	orig = image.copy()
	(rows, cols) = image.shape[:2]

	# resized
	image = cv2.resize(image, (320, 320))
	(newrows, newcols) = image.shape[:2]

	## Code segment that finds bounding boxes for text in images
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")
	blob = cv2.dnn.blobFromImage(image, 1.0, (newrows, newcols),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])


	boxes = []
	predictions = []

	## This is a decode function provided with EAST Algorithm
	for y in range(0, scores.shape[2]):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, scores.shape[3]):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			boxes.append((startX, startY, endX, endY))
			predictions.append(scoresData[x])

	newBoxes = non_max_suppression(np.array(boxes), probs=predictions)

	cropped_images = []
	if (showImage):
		
		padding = padding_crop
		# loop over the bounding boxes
		for (startX, startY, endX, endY) in newBoxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * (cols/float(newcols)))
			startY = int(startY * (rows/float(newrows)))
			endX = int(endX * (cols/float(newcols)))
			endY = int(endY * (rows/float(newrows)))

			# draw the bounding box on the frame
			# cv2.rectangle(orig, (startX-padding, startY-padding), (endX+padding, endY+padding), (0, 255, 0), 2)

			crop_img = orig[startY-padding:endY+padding, startX-padding:endX+padding]
			cropped_images.append(crop_img)
			cv2.imshow("cropped", crop_img)
			cv2.waitKey(0)

		# show the output frame
		# cv2.imshow("Text Detection", orig)
		# cv2.waitKey(0)

	return cropped_images

# cropped_images = getBoxes("texttotest2.png", 10, True)


def check():
	hdf5_path = "train_data.h5"
	MODEL_FILENAME = "model.hdf5"
	MODEL_LABELS_FILENAME = "model_labels.dat"

	images, labels = getData(hdf5_path, False)


	# Load up the model labels (so we can translate model predictions to actual letters)
	with open(MODEL_LABELS_FILENAME, "rb") as f:
		lb = pickle.load(f)

	# Load the trained neural network
	# model = load_model(MODEL_FILENAME)

	
	cropped_images = getBoxes("C:\\Users\\numan98khan\\Desktop\\texttotest2.png", 10, True)
	# cropped_images = getBoxes("test_ocr.png", 10, True)

	for img_crop in cropped_images:
		# image = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

		gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
		# Adding some extra padding around the image
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

		# applying threshold
		thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

		# creating empty list for holding the coordinates of the letters
		letter_image_regions = []

		# finding the contours
		contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# print(contours)
		# print(hierarchy)
		for contour in contours:
			# Get the rectangle that contains the contour
			(x, y, w, h) = cv2.boundingRect(contour)

			# checking if any counter is too wide
			if w / h > 1.25:
				# Split two letter regions
				half_width = int(w / 2)
				letter_image_regions.append((x, y, half_width, h))
				letter_image_regions.append((x + half_width, y, half_width, h))
			else:  
				letter_image_regions.append((x, y, w, h))

		letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

		# Create an output image and a list to hold our predicted letters
		output = cv2.merge([img_crop] * 3)
		predictions = []

		# cv2.imshow("Output", output)
		# cv2.waitKey(0)

		# loop over the lektters
		for letter_bounding_box in letter_image_regions:
			# Grab the coordinates of the letter in the image
			x, y, w, h = letter_bounding_box

			# Extract the letter from the original image with a 2-pixel margin around the edge
			letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

			# Re-size the letter image to 20x20 pixels to match training data
			letter_image = resize_to_fit(letter_image, 20, 20)

			plt.imshow(letter_image)
			plt.title('my picture')
			plt.show()

			# cv2.imshow("Output", letter_image)
			# cv2.waitKey(0)


			# Turn the single image into a 4d list of images to make Keras happy
			letter_image = np.expand_dims(letter_image, axis=2)
			letter_image = np.expand_dims(letter_image, axis=0)

			# # Ask the neural network to make a prediction
			prediction = model.predict(letter_image)
			print(prediction)


			# # Convert the one-hot-encoded prediction back to a normal letter
			letter = lb.inverse_transform(prediction)[0]
			predictions.append(letter)

			# draw the prediction on the output image
			# cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
			# cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
		for pred in predictions:
			print(chr(pred))

	# captcha_image_files = np.random.choice(images, size=(10,), replace=False)

check()
# train()