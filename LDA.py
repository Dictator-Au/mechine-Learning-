# -*- coding: utf-8 -*-
"""
@author: zihan

Thanks for the code provided by the author.
If there is any infringement, I will delete it immediately
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

path='dataset-credit-default.csv'
data=pd.read_csv(path,header=0,index_col=0)
data=np.array(data,dtype=np.float64)
data = np.array(data, dtype=np.float64)
X = data[:, 1:]
y = data[:,:1]
y[y==0]=-1
seed = 1
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, test_size=0.25)

lda = LDA()
x_train_lda = lda.fit_transform(X_train, y_train)  
y_pre = lda.fit_transform(X_test, y_test)  
for m in range(y_pre.shape[0]):
    y_pre[m]=1 if y_pre[m]>0 else -1

acc1=np.sum(y_pre==y_val)/X_test.shape[0]
print('Validation Accuraccy:',acc1)
acc2=np.sum(y_pre==y_test)/X_test.shape[0]
print('Testing Accuraccy:',acc2)

print(classification_report(y_true=y_test, y_pred=y_pre))