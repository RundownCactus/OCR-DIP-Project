import os
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import math
import matplotlib.pyplot as plt
import h5py

import csv


## Function for curating dataset
def files():
    labelPath = "ch2_training_localization_transcription_gt"
    imagePath = "ch2_training_images"

    listImg = os.listdir(imagePath)
    listLabel = os.listdir(labelPath)

    noOfImages = 0
    strLabelList = []
    for imgNo in range(0, len(listLabel)):
        with open(labelPath+"\\"+listLabel[imgNo]) as csvFile:
            readCsv = csv.reader(csvFile)
            for row in readCsv:
                if (row[-1] != "###"):
                    strLabelList.append(row[-1])
                    noOfImages+=1

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File("train_data.h5", mode='w')
    hdf5_file.create_dataset("train_img", (noOfImages, 112, 224, 3), np.uint8)

    asciiList = [n.encode("ascii", "ignore") for n in strLabelList]
    hdf5_file.create_dataset('train_lab', (len(asciiList),1),'S10', asciiList)

    print(len(listImg))
    # print(listLabel)

    imgCounter = 0
    for imgNo in range(0, len(listImg)):
        image = cv2.imread(imagePath+"\\"+listImg[imgNo])
        # print(image)
        orig = image.copy()
        (rows, cols) = image.shape[:2]


        with open(labelPath+"\\"+listLabel[imgNo]) as csvFile:
            readCsv = csv.reader(csvFile)
            for row in readCsv:
                if (row[-1] == "###"):
                    continue
                print(labelPath+"\\"+listLabel[imgNo])
                print(row)
                y0 = int(row[1])
                y1 = int(row[5])
                if ('ï»¿' in row[0]):
                    row[0] = row[0][3:]
                x0 = int(row[0])
                x1 = int(row[4])
                print(y0, y1, x0, x1)
                crop_img = orig[y0:y1, x0:x1]

                # print("esketit", crop_img)
                if (crop_img == []):
                    continue

                try:
                    crop_img = cv2.resize(crop_img, (224, 112), interpolation=cv2.INTER_CUBIC)
                except cv2.error:
                    print(crop_img)
                    cv2.imshow("Text Detection", crop_img)
                    cv2.waitKey(0)

                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                
                hdf5_file["train_img"][imgCounter, ...] = crop_img[None]

                imgCounter += 1

               

## Function for getting the whole curated dataset
def getData(path, sample):
    # open the hdf5 file
    hdf5_file = h5py.File(hdf5_path, "r")

    images = hdf5_file["train_img"]
    labels = hdf5_file["train_lab"]

    if (sample):
        data = images[0]
        print(labels[0])
        cv2.imshow("Text Detection", data)
        cv2.waitKey(0)

    return images, labels


hdf5_path = "train_data.h5"
images, labels = getData(hdf5_path, True)
