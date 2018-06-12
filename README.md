# Number Recognition using KNN

Use the K-Nearest Neighbour classifier to classify image of number

# Requirement to run
python3.5.2 
opencv3.2

Install OpenCV command in conda:

pip install opencv-python
# Basic of k-nearest neighbors(KNN)  algorithm
![image](https://github.com/wangjinlong9788/NumberRecognitionKNN/blob/master/279px-KnnClassification.svg.png)

The test sample (green circle) should be classified either to the first class of blue squares or to the second class of red triangles. If k = 3 (solid line circle, k is a user-defined constant) it is assigned to the second class because there are 2 triangles and only 1 square inside the inner circle. If k = 5 (dashed line circle) it is assigned to the first class (3 squares vs. 2 triangles inside the outer circle). Here the number of squares and triangles could be seens as distance. A commonly used distance metric for continuous variables is Euclidean distance.

The time complexity of k-nearest neighbors(KNN)  algorithm is O(n). 
# Example of KNN in OpenCV
import cv2 as cv 

import numpy as np

import matplotlib.pyplot as plt

#Feature set containing (x,y) values of 25 known/training data

trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

#Labels each one either Red or Blue with numbers 0 and 1

responses = np.random.randint(0,2,(25,1)).astype(np.float32)

#Take Red families and plot them

red = trainData[responses.ravel()==0]

plt.scatter(red[:,0],red[:,1],80,'r','^')

#Take Blue families and plot them

blue = trainData[responses.ravel()==1]

plt.scatter(blue[:,0],blue[:,1],80,'b','s')

newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)

plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv.ml.KNearest_create()

knn.train(trainData, cv.ml.ROW_SAMPLE, responses)

ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)

print( "result:  {}\n".format(results) )

print( "neighbours:  {}\n".format(neighbours) )

print( "distance:  {}\n".format(dist) )

plt.show()

# The result of the example code is:

![image](https://github.com/wangjinlong9788/NumberRecognitionKNN/blob/master/cvresult.png)
