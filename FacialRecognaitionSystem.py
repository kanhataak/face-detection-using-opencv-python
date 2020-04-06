# Face recognition using OpenCV
# Once we have acquired some data, we’ll need to read it in our program
import cv2
import numpy as np
from os import listdir
from os.path import isfile,join
# grab the paths to the input images in our dataset
pathData = "F:/PycharmProjects/opencv/faceCollection/"

#get the images names that are inside the given subject directory
files = [f for f in listdir(pathData) if isfile(join(pathData, f))]

#list to hold all subject faces
#list to hold labels for all subjects
trainingData, labels = [], []
for i, file in enumerate(files):
    imgPath = pathData + files[i]
    images = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    trainingData.append(np.asarray(images, dtype=np.uint8))
    # add label for this face
    labels.append(i)

labels = np.asarray(labels, dtype=np.int32)

# model = cv2.face.LBPHFaceRecognizer_create()
recognizerModel = cv2.face.LBPHFaceRecognizer_create()

recognizerModel.train(np.asarray(trainingData), np.asarray(labels))

print("Model trainend")

