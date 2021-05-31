# -*- coding: utf-8 -*-
"""
@author: zihan

Thanks for the code provided by the author.
If there is any infringement, I will delete it immediately
"""

import numpy as np
seed = 1
np.random.seed(seed)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import pandas as pd
import data_processing as dp

def sigmoid(x):
    return 1/(1+np.exp(-x))
    # （需要填写的地方，输入x返回sigmoid(x)，x可以是标量、向量或矩阵）
    
    
def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
    # （需要填写的地方，输入x返回sigmoid(x)在x点的梯度，x可以是标量、向量或矩阵）
    
    
def mse_loss(y_true, y_pred):
    return np.sum(np.power(y_pred-y_true,2))
    # （需要填写的地方，输入真实标记和预测值返回他们的MSE（均方误差，不需要除以2）,其中真实标记和预测值维度都是(n_samples,) 或 (n_samples, n_outputs)）
    
    
def accuracy(y_true, y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0] 
    # （需要填写的地方，输入真实标记和预测值返回Accuracy，其中真实标记和预测值是维度相同的向量）
    
    
def to_onehot(y):
    # 输入为向量，转为onehot编码
    y = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    return y


class NeuralNetwork():
    def __init__(self, d, q, l):
        # weights
        self.v = np.random.randn(d, q)
        self.w = np.random.randn(q, l)
        # biases
        self.gamma = np.random.randn(q)
        self.theta = np.random.randn(l)
        # 以上为神经网络中的权重和偏置，其中具体含义见西瓜书P101

    def predict(self, X):
        '''
        X: shape (n_samples, d)
        returns: shape (n_samples, l)
        '''
        b=sigmoid(np.dot(X,self.v)-self.gamma)
        yp=sigmoid(np.dot(b,self.w)-self.theta)
        return yp
        # （需要填写的地方，输入样本，输出神经网络最后一层的输出值）
        
    
    def train(self, X, y, learning_rate = 1, epochs = 500):
        '''
        X: shape (n_samples, d)
        y: shape (n_samples, l)
        输入样本和训练标记，进行网络训练
        '''
        for epoch in range(epochs):
            # （以下部分为向前传播过程，请完成）
            b=sigmoid(np.dot(X,self.v)-self.gamma)
            yp=sigmoid(np.dot(b,self.w)-self.theta)
            
            # （以下部分为计算梯度，请完成）
            # 输出层梯度
            yg=yp-y
            betag=np.multiply(yp,1-yp)
            g=-np.multiply(yg,betag)
            
            # 隐层梯度
            bg=np.dot(-g,self.w.T)
            alphag=np.multiply(b,1-b)
            e=-np.multiply(bg,alphag)
            
            # 更新权重和偏置
            self.w+=learning_rate*np.dot(b.T,g)/X.shape[0]
            self.theta-=learning_rate*np.mean(g,axis=0).ravel()
            self.v+=learning_rate*np.dot(X.T,e)/X.shape[0]
            self.gamma-=learning_rate*np.mean(e,axis=0).ravel()

            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = self.predict(X)
                loss = (mse_loss(y, y_preds))
                print("Epoch %d loss: %.3f"%(epoch, loss))
    
if __name__ ==  '__main__':
    # 获取数据集，训练集处理成one-hot编码
    data=pd.read_csv('dataset-credit-default.csv',header=0,index_col=0)
    data=np.array(data,dtype=np.float64)
    X=data[:,1:]
    y=data[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed,test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, test_size=0.25)
    y_train = to_onehot(y_train)
    
    
    # 训练网络（可以对下面的n_hidden_layer_size，learning_rate和epochs进行修改，以取得更好性能）
    n_features = X.shape[1]
    n_hidden_layer_size = 100
    n_outputs = len(np.unique(y))
    network = NeuralNetwork(d = n_features, q = n_hidden_layer_size, l = n_outputs)
    network.train(X_train, y_train, learning_rate =1, epochs =1000)

    # 预测结果
    y_pred = network.predict(X_val)
    y_pred_class = np.argmax(y_pred, axis=1)
    mse = mse_loss(to_onehot(y_val), y_pred)
    print("\nValidation MSE: {:.3f}".format(mse))
    acc = accuracy(y_val, y_pred_class) * 100
    print("\nValidation Accuracy: {:.3f} %".format(acc))

    y_pred = network.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    mse = mse_loss(to_onehot(y_test), y_pred)
    print("\nTesting MSE: {:.3f}".format(mse))
    acc = accuracy(y_test, y_pred_class) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))

    print(classification_report(y_true=y_test, y_pred=y_pred_class))