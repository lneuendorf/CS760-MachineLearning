import numpy as np

class LogisticRegressor():
    def __init__(self, n_features):
        self.n_features = n_features
        self.theta = np.zeros([n_features,1], dtype=np.float) # model weights
        
    def sigmoid(self, x):
        return 1. /(1. + np.exp(-(np.dot(self.theta.T,x))))

    def gradient_loss(self, x, y):
        return np.dot(x,(self.sigmoid(x).T-y))

    def update_weights(self, x, y, learning_rate):
        self.theta = self.theta - (learning_rate * self.gradient_loss(x.T,y))

    def train(self, x, y, learning_rate, n_epochs):
        for i in range(0,n_epochs):
            self.update_weights(x,y,learning_rate)
            
    def predict(self, x_pred):
        return self.sigmoid(x_pred.T)
    
    def confusion_matrix(self, y_true, y_pred):
        df = pd.concat([pd.DataFrame(y_true),pd.DataFrame(y_pred)], axis=1,ignore_index=True) 
        df.columns = ['y_true','y_pred']
        TP = len(df[(df.y_true == 1) & (df.y_pred == 1)])
        FP = len(df[(df.y_true == 0) & (df.y_pred == 1)])
        FN = len(df[(df.y_true == 1) & (df.y_pred == 0)])
        TN = len(df[(df.y_true == 0) & (df.y_pred == 0)])
        return {'TP':TP, 'FP':FP, 'FN':FN, 'TN':TN}

