import numpy as np
import cv2

#read image  and convert color
img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#Blue Green Red

#divide the image to 5000 pieces, each of which has 20x20 
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

#convert to numpy arrays 
x = np.array(cells)

#half data for training and half to be tested 
train = x[:,:50].reshape(-1,400).astype(np.float32)
test = x[:,50:100].reshape(-1,400).astype(np.float32)

#create tag of train and test 
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
np.savez('data.npz',train=train, train_labels=train_labels)
with np.load('data.npz') as data:
    print data.files
    train = data['train']
    train_labels = data['train_labels']
#create a K-Nearest Neighbour classifer and train data then test  
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

#check the accuracy  
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0 / result.size
print(accuracy)
