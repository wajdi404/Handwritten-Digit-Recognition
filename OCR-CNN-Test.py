import math
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.models import Model,Sequential
from keras.layers import Flatten,Dropout,Dense,Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model


########## CREATE CAMERA OBJECT
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

########## LOAD THE TRAINNED MODEL
model = load_model( "myModel")

##### PREPOSSESSING FUNCTION FOR IMAGES FOR TRAINING #####
def preProcessing(Img):
    Img = cv2.cvtColor( Img, cv2.COLOR_BGR2GRAY)
    Img = cv2.equalizeHist(Img)
    Img = Img/255.0
    return Img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image", img)
    img = img.reshape(1,32,32,1)

    #### PREDICT
    #classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal = np.amax(predictions)
    classIndex = list(predictions[0]).index(probVal)
    #print(classIndex,probVal)
    #print( " classIndex : ", classIndex, " probVal : ", probVal)
    msg = str(classIndex) + " " + str(math.trunc(probVal*100)) + "%"
    if probVal> 0.65:
       cv2.putText( imgOriginal, msg,
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


'''imgOriginal = cv2.imread("myData/7/img008-00105.png")
img = np.asarray(imgOriginal)
img = cv2.resize(img,(32,32))
img = preProcessing(img)
cv2.imshow("Processsed Image",img)
img = img.reshape(1,32,32,1)

############## PREDICT

#classIndex = int(model.predict_classes(img))
#print(classIndex)
predictions = model.predict(img)
#print(predictions)
probVal = np.amax(predictions)
classIndex = list(predictions[0]).index(probVal)
print( "Predictions List : ", predictions, " classIndex : ", classIndex, " probVal : ", probVal)
'''
