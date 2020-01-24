"""def pArray(x):  #reshapes the picture array to be in picture dimentions
    from numpy import array
    from numpy import reshape
    x = x.reshape((int(x.size/784),784))
    return x
def lArray(x): #reshapes the labels array to be 2D
    from numpy import array
    from numpy import reshape
    x = x.reshape(1,-1)
    return x
   
    
def plotArray(x): #takes the array and plots the picture
    imgplot = plt.imshow(x,cmap='gray')
    plt.show()

def ArraytoPlot(x,array): #choose index of array, returns an array of one picture chosen
    temp= []
    for j in range(784):
        temp.append(array[x][j])
    temp = np.array(temp)
    temp = temp.reshape(28,28)
    return temp
    
def false_predictions(x,y):
    counter=0
    for i in range(x.size):
        if (y[i] != x[i]):
            counter+=1
    return counter               

def Fitting(type,arr1,arr2):
    from sklearn.neighbors import KNeighborsClassifier
    if type == "knn":
        #Using KNN
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(arr1,arr2)
    else:
        #Using SVM
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(arr1,arr2)
    return classifier

def ConfusionMatrix(arr1,arr2):
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(arr1,arr2)
    return cm"""

#-----------------------------------------------------------------------------------------------------#

#MAIN

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os

#Change directory
#os.chdir("D:\Machine learning course\Competition\Digit_CNN")

# Importing train
dataset_train = pd.read_csv('train.csv')
labels_train = dataset_train.iloc[:, 0].values
picture_train = dataset_train.iloc[:,1:785].values

#Importing dataset of test
dataset_test = pd.read_csv('test.csv')
picture_test = dataset_test.iloc[:,:].values

picture_train = picture_train.reshape(picture_train.shape[0], 28, 28, 1).astype('float32')
picture_test = picture_test.reshape(picture_test.shape[0], 28, 28, 1).astype('float32')
picture_train = picture_train/255
picture_test = picture_test/255

#Encoding
from keras.utils.np_utils import to_categorical
labels_train = to_categorical(labels_train)

#Creating ANN
# Create your classifier here
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#Initializing classifier
classifier = Sequential()

#Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (28, 28, 1), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Convolution 2
classifier.add(Conv2D(16, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

#Pooling 2
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Drop
classifier.add(Dropout(0.2))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 50, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting
classifier.fit(picture_train, labels_train, batch_size = 10, epochs = 100)


#Save Model
filename = 'Digit_CNN_Final_Model.sav'
classifier.save(filename)


"""#load model
from keras.models import load_model
classifier = load_model('Digit_CNN_Final_Model.sav')

# Predicting the Test set results
label_pred = classifier.predict(picture_test)
#y_pred = (y_pred>0.5)
predicted_values = np.argmax(label_pred,1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#outputing in csv file
Submition = pd.read_csv('sample_submission.csv')
#Submition = np.array(Submition)
temp = []
for i in range(predicted_values.size):
    for j in range(2):
        if j == 0:
            temp.append(i+1)
        else:
            temp.append(predicted_values[i])

temp = np.array(temp)
temp = temp.reshape(predicted_values.size,2)
np.savetxt("predicted.csv", temp, delimiter=",")

#Empty the csv
f = open("sample_submission.csv", "w+")
f.close()

titles = {'ImageId':1,'Label':2}

import csv
b = open("sample_submission.csv","a")
a = csv.writer(b,lineterminator = '\n')
a.writerow(titles)
a.writerows(temp)
b.close()
"""





"""n = 5550
k = ArraytoPlot(n,picture_test)
plotArray(k)
predicted_values[n]"""


