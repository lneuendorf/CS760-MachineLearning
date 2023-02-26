import math
import numpy as np
import pandas as pd

# fixed-length queue storing (distance,label) pairs sorted by distance in desceding order
class Queue():
    def __init__(self, length):
        self.length = length
        self.max_dist = None
        self.queue = []
        self.isEmpty = True

    def insert(self, distance, y):
        # remove element with largest distance if queue full
        if len(self.queue) == self.length:
            self.queue.pop()
            if len(self.queue) == 0:
                self.isEmpty = True
            
        # insert new element
        if self.isEmpty:
            self.queue.append([distance,y])
            self.isEmpty = False
        else:
            for i in range(0,len(self.queue)):
                if self.queue[i][0] > distance:
                    self.queue.insert(i,[distance,y])
                    break
                if  (i+1) == len(self.queue):
                    self.queue.insert(i+1,[distance,y])
        self.max_dist = self.queue[-1][0]

class KNNClassifier():
    def __init__(self, X_train=None, y_train=None):
        if (X_train is None) or (y_train is None):
            raise Exception("Failed to provide KNNClassifier with training set.")
        if len(X_train) != len(y_train):
            raise Exception("X_train and y_train are of different lengths.")
        self.X_train = X_train
        self.y_train = y_train
        self.train_samples = len(y_train)
    
    def euclidean_distance(self, arr1, arr2):
        sum_dif_sqrd = 0
        for x1,x2 in zip(arr1[0],arr2[0]):
            sum_dif_sqrd += (x2-x1)**2
        return math.sqrt(sum_dif_sqrd)

    def predict(self, X, k):
        y_pred = pd.DataFrame(np.zeros(len(X)),columns=['Y_pred'])
        for i in range(0,len(X)):
            q = Queue(length=k)
            sample = X.iloc[i:i+1,:].to_numpy()
            for j in range(0,self.train_samples):
                train_sample = self.X_train.iloc[j:j+1,:].to_numpy()
                dist = self.euclidean_distance(sample, train_sample)
                if (q.max_dist is None) or (q.length > len(q.queue)):
                    q.insert(dist, int(self.y_train.iloc[j]))
                elif q.max_dist > dist:
                    q.insert(dist, int(self.y_train.iloc[j]))
            #find majority class for sample
            sum_y=0
            for z in range(0,q.length):
                sum_y += q.queue[z][1]
            y_ratio = sum_y / q.length
            if y_ratio > 0.5:
                y_pred.iloc[i] = 1
            else:
                y_pred.iloc[i] = 0
        return y_pred.astype('int').to_numpy()

    def confusion_matrix(self, y_true, y_pred):
        df = pd.concat([y_true,y_pred], axis=1,ignore_index=True) 
        df.columns = ['y_true','y_pred']
        TP = len(df[(df.y_true == 1) & (df.y_pred == 1)])
        FP = len(df[(df.y_true == 0) & (df.y_pred == 1)])
        FN = len(df[(df.y_true == 1) & (df.y_pred == 0)])
        TN = len(df[(df.y_true == 0) & (df.y_pred == 0)])
        return np.matrix([[TP,FP],[FN,TN]])
