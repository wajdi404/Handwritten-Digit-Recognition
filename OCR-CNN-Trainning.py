
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

#import pickle

##### IMPORTING DATA/IMAGES FROM FOLDERS #####
Images = []     # LIST CONTAINING ALL THE IMAGES
Labels = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES

dirList = os.listdir("myData")

print("Total Classes Detected : ", len(dirList))

print("Importing Labels/N°Images .......")
for Cl in range (0,len(dirList)):

    myImgList = os.listdir( "myData/" + str(Cl) )

    for Im in myImgList:
        curImg = cv2.imread( "myData/" + str(Cl) + "/" + Im )
        curImg = cv2.resize( curImg, (32,32) )
        Images.append(curImg)
        Labels.append(Cl)
    print( "(", Cl, "/", len(myImgList), ")", end= " ; ")

print(" ")
print("Total Image in Images List = ", len(Images))
print("Total Label in Labels List = ", len(Labels))
print(" ")

############ CONVERT TO NUMPY ARRAY #############
Images = np.array(Images)
Labels = np.array(Labels)
print( "Images List shape", Images.shape)
print( "Labels List shape", Images.shape)
print(" ")

############ SPLITTING THE DATA ##################
X_train, X_test, Y_train, Y_test = train_test_split( Images, Labels, test_size=0.2)
X_train, X_validation, Y_train, Y_validation = train_test_split( X_train, Y_train, test_size=0.2)

##### PREPOSSESSING FUNCTION FOR IMAGES FOR TRAINING #####
def preProcessing(Img):
    Img = cv2.cvtColor( Img, cv2.COLOR_BGR2GRAY)
    Img = cv2.equalizeHist(Img)
    Img = Img/255.0
    return Img

X_train = np.array( list(map(preProcessing,X_train)) )
X_test = np.array( list(map(preProcessing,X_test)) )
X_validation = np.array( list(map(preProcessing,X_validation)) )

################# RESHAPE IMAGES #################
print( "Before RESHAPE : ", X_train.shape)
X_train = X_train.reshape( X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape( X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape( X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
print( "After RESHAPE : ", X_train.shape)

################# IMAGE AUGMENTATION ################
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

################## ONE HOT ENCODING OF MATRICES ############
print( "Before ONE HOT ENCODING : ", Y_train.shape)
Y_train = to_categorical( Y_train, len(dirList))
Y_test = to_categorical( Y_test, len(dirList))
Y_validation = to_categorical( Y_validation, len(dirList))
print( "After ONE HOT ENCODING : ", Y_train.shape)

################### CREATING THE MODEL ###################
def creatModel():

    model = Sequential()
    model.add((Conv2D(60, (5,5), input_shape=(32,32, 1), activation='relu')))
    model.add((Conv2D(60, (5,5), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add((Conv2D(60 // 2, (3,3), activation='relu')))
    model.add((Conv2D(60 // 2, (3,3), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense( 10, activation='softmax'))
    model.compile( optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy']
                  )
    return model

model = creatModel()
print(model.summary())

################## STARTING THE TRAINING PROCESS ###############
History = model.fit_generator(   dataGen.flow( X_train, Y_train, batch_size = 50 ),
                                 steps_per_epoch = 2000,
                                 epochs = 10,
                                 validation_data = ( X_validation, Y_validation),
                                 shuffle = 1
                              )

print(History.history)

################## PLOT THE TRAINING RESULTS #################
plt.figure(1)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()


#### EVALUATE USING TEST IMAGES
score = model.evaluate( X_test, Y_test, verbose=0)
print( 'Test Score = ', score[0])
print( 'Test Accuracy =', score[1])


# Calling `save('myModel')` creates a SavedModel folder `myModel`.
model.save("myModel")


