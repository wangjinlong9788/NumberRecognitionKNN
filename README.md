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
