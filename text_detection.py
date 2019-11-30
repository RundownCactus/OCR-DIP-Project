from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import math

def getBoxes(imageName, showImage):
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

	newBoxes = boxes

	# newBoxes = non_max_suppression(np.array(boxes), probs=predictions)

	if (showImage):
		# loop over the bounding boxes
		for (startX, startY, endX, endY) in newBoxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * (cols/float(newcols)))
			startY = int(startY * (rows/float(newrows)))
			endX = int(endX * (cols/float(newcols)))
			endY = int(endY * (rows/float(newrows)))

			# draw the bounding box on the frame
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

		# show the output frame
		cv2.imshow("Text Detection", orig)
		cv2.waitKey(0)

	return newBoxes

newBoxes = getBoxes("car_wash.png", True)
print(newBoxes)
newBoxes = getBoxes("sal.jpeg", True)
print(newBoxes)