# -*- coding: utf-8 -*-
"""
@author: zihan

Thanks for the code provided by the author.
If there is any infringement, I will delete it immediately
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# read data
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
#SVM
alpha=np.zeros([X.shape[0],1])
b=0

def SMO(X,y,Margin='hard',kernel_type='linear',maxiter=100,C=1,toler=1e-5):
    alpha=np.zeros([X.shape[0],1])
    b=0
    trigger1=0
    i=-1
    for iter in range(maxiter):
        trigger=1
        fx=np.zeros(y.shape)
        for col in range(X.shape[0]):
            for alpha_num in range(alpha.shape[0]):
                if alpha[alpha_num,0]!=0:
                    fx[col]+=alpha[alpha_num,0]*y[alpha_num,0]*kernel(X[col],X[alpha_num],kernel_type)
            fx[col]+=b
        error=np.zeros(alpha.shape[0])
        for m in range(alpha.shape[0]):
            error[m]=fx[m,0]-y[m,0]
        if Margin=='hard':   #硬间隔分类器
            while (i<alpha.shape[0]-1):
                i+=1
                if (y[i,0]*fx[i,0])-1<0 or (alpha[i,0]>0 and np.abs(y[i,0]*fx[i,0]-1)>toler):     #不满足KKT条件则对alpha[i]进行更新
                    if trigger1==1 and (y[i,0]*fx[i,0])-1>=-toler:
                        continue
                    trigger1=0
                    delta=np.zeros(alpha.shape[0])
                    for j in range(alpha.shape[0]):
                        if (y[j,0]*fx[j,0])<1 or (alpha[j,0]>0 and np.abs(y[j,0]*fx[j,0]-1)>toler):
                            delta[j]=np.abs(error[j]-error[i])
                    if (delta==0).all()==True:
                        for j in range(alpha.shape[0]):
                            if alpha[j]!=0:
                                delta[j]=np.abs(error[j]-error[i])
                    for j in range(alpha.shape[0]):
                        if i!=j and delta[j]==np.max(delta):
                            break                          #选出距离最远的alpha[j]进行更新
                    constant=alpha[i,0]*y[i,0]+alpha[j,0]*y[j,0]
                    eta=kernel(X[i],X[i],kernel_type)+kernel(X[j],X[j],kernel_type)-2*kernel(X[i],X[j],kernel_type)
                    old_alphai=alpha[i,0]
                    old_alphaj=alpha[j,0]
                    temp=alpha[j,0]+y[j,0]*(error[i]-error[j])/eta
                    if y[i,0]*y[j,0]==-1:
                        H=-1
                        L=np.max([0,alpha[j,0]-alpha[i,0]])
                    elif y[i,0]*y[j,0]==1:
                        H=alpha[j,0]+alpha[i,0]
                        L=0
                    if temp>H and H>=0:
                        alpha[j,0]=H
                        #alpha[j,0]=temp
                    elif temp<L:
                        alpha[j,0]=L
                    else:
                        alpha[j,0]=temp
                    print(i,j,temp,iter)
                    if np.abs(alpha[j,0] - old_alphaj)<=toler:
                        continue
                    alpha[i,0]=y[i,0]*(constant-y[j,0]*alpha[j,0])
                    trigger=0
                    break
        elif Margin=='soft':    #软间隔分类器
            while (i<alpha.shape[0]-1):
                #if i==alpha.shape[0]-1 and trigger1==0:
                #    i=-1
                i+=1
                if ((alpha[i,0]==0 and (y[i,0]*fx[i,0])<1) or (alpha[i,0]>0 and alpha[i]<C and (y[i,0]*fx[i,0])!=1)
                    or (alpha[i,0]==C and (y[i,0]*fx[i,0])>1)):     #不满足KKT条件则对alpha[i]进行更新
                    trigger1=0
                    delta=np.zeros(alpha.shape[0])
                    for j in range(alpha.shape[0]):
                        if ((alpha[j,0]==0 and (y[j,0]*fx[j,0])<1) or (alpha[j,0]>0 and alpha[j]<C and (y[j,0]*fx[j,0])!=1)
                            or (alpha[j,0]==C and (y[j,0]*fx[j,0])>1)):
                            delta[j]=np.abs(error[j]-error[i])
                    if (delta==0).all()==True:
                        for j in range(alpha.shape[0]):
                            if alpha[j]!=0:
                                delta[j]=np.abs(error[j]-error[i])
                    for j in range(alpha.shape[0]):
                        if i!=j and delta[j]==np.max(delta):
                            break                          #选出距离最远的alpha[j]进行更新
                    constant=alpha[i,0]*y[i,0]+alpha[j,0]*y[j,0]
                    eta=kernel(X[i],X[i],kernel_type)+kernel(X[j],X[j],kernel_type)-2*kernel(X[i],X[j],kernel_type)
                    old_alphai=alpha[i,0]
                    old_alphaj=alpha[j,0]
                    temp=alpha[j,0]+y[j,0]*(error[i]-error[j])/eta
                    if y[i,0]*y[j,0]==-1:
                        H=np.min([C,C+alpha[j,0]-alpha[i,0]])
                        L=np.max([0,alpha[j,0]-alpha[i,0]])
                    elif y[i,0]*y[j,0]==1:
                        H=np.min([C,alpha[j,0]+alpha[i,0]])
                        L=np.max([0,alpha[j,0]+alpha[i,0]-C])
                    if temp>H:
                        alpha[j,0]=H
                    elif temp<L:
                        alpha[j,0]=L
                    else:
                        alpha[j,0]=temp
                    print(i,j,temp,iter)
                    if np.abs(alpha[j,0] - old_alphaj)<=toler:
                        continue
                    alpha[i,0]=y[i,0]*constant-y[i,0]*y[j,0]*alpha[j,0]
                    trigger=0
                    break
        if trigger==1 and i==(alpha.shape[0]-1) and trigger1==0:
            i=-1
            trigger=0
            trigger1=1
        if trigger==0 and (delta==0).all()==False:
            tempi=tempj=0
            for alpha_num in range(alpha.shape[0]):
                tempi+=alpha[alpha_num,0]*y[alpha_num,0]*kernel(X[i],X[alpha_num],kernel_type)
                tempj+=alpha[alpha_num,0]*y[alpha_num,0]*kernel(X[j],X[alpha_num],kernel_type)
            bi=y[i,0]-float(tempi)
            bj=y[j,0]-float(tempj)
            if alpha[i]> 0:
                b=bi
            elif alpha[j]> 0:
                b=bj
            else:
                b=(bi+bj)/2
        else:
            break   #完成迭代
    print(fx)
    return b,alpha,iter
#linear or Radial Basis Function
def kernel(a,b,kernel_type):
    if kernel_type=='linear':
        return np.inner(a,b)
    if kernel_type=='rbf':
        sigma=1 #核函数参数
        return np.exp(-(np.inner((a-b),(a-b))/(2*np.power(sigma,2))))

def plot(X,Y,Margin,kernel_type):
    b,alpha,iter=SMO(X,Y,Margin,kernel_type)
    print(iter,alpha)
    x1=np.linspace(min(X[:,0])-(max(X[:,0])-min(X[:,0]))/20,max(X[:,0])+(max(X[:,0])-min(X[:,0]))/20,100)
    x2=np.linspace(min(X[:,1])-(max(X[:,1])-min(X[:,1]))/20,max(X[:,1])+(max(X[:,1])-min(X[:,1]))/20,100)
    x1,x2=np.meshgrid(x1,x2)
    x12=np.array([x1,x2])
    f=np.zeros(x1.shape)
    for row in range(x12.shape[1]):
        for col in range(x12.shape[2]):
            for alpha_num in range(alpha.shape[0]):
                if alpha[alpha_num,0]!=0:
                    f[row,col]+=alpha[alpha_num,0]*Y[alpha_num,0]*kernel(x12[:,row,col],X[alpha_num],kernel_type)
            f[row,col]+=b
    if Margin=='hard':
        plt1=plt.contour(x1,x2,f,0,colors='orange')
        plt.clabel(plt1, fmt='hard')
    elif Margin=='soft':
        plt2=plt.contour(x1,x2,f,0,colors='green')
        plt.clabel(plt2, fmt='soft')
    for alpha_num in range(alpha.shape[0]):
        if Y[alpha_num,0]==1:
            plt.scatter(X[alpha_num, 0], X[alpha_num, 1],s=50,color='b')
        else:
            plt.scatter(X[alpha_num, 0], X[alpha_num, 1],s=50,color='r')
        plt.text(X[alpha_num, 0], X[alpha_num, 1],alpha_num)
        if alpha[alpha_num,0]!=0:
            if Margin=='hard':
                plt.scatter(X[alpha_num, 0], X[alpha_num, 1], marker='o', c='', edgecolors='orange', s=150,
                       label='support_vectors')
            elif Margin=='soft':
                plt.scatter(X[alpha_num, 0], X[alpha_num, 1], marker='o', c='', edgecolors='green', s=150,
                       label='support_vectors')

def assess(X,y,X_test,y_test,b,alpha,kernel_type):
    yp=np.zeros(y_test.shape)
    fx=np.zeros(X_test.shape[0])
    
    for col in range(X_test.shape[0]):
        for alpha_num in range(alpha.shape[0]):
            if alpha[alpha_num,0]!=0:
                fx[col]+=alpha[alpha_num,0]*y[alpha_num,0]*kernel(X_test[col],X[alpha_num],kernel_type)
        fx[col]+=b
        if fx[col]>0:
            yp[col,0]=1
        else:
            yp[col,0]=-1
    acc=np.sum(yp==y_test)/X_test.shape[0]

    print(classification_report(y_true=y_test, y_pred=yp))
    return acc

    
#plot(X,y,Margin='hard',kernel_type='linear')
#plot(X,y,Margin='hard',kernel_type='rbf')
#plot(X,y,Margin='soft',kernel_type='linear')
#plot(X,y,Margin='soft',kernel_type='rbf')

b,alpha,iter=SMO(X_train,y_train,Margin='hard',kernel_type='rbf')
acc1=assess(X_train,y_train,X_val,y_val,b,alpha,kernel_type='rbf')
print('Validation Accuraccy:',acc1)
acc2=assess(X_train,y_train,X_test,y_test,b,alpha,kernel_type='rbf')
print('Testing Accuraccy:',acc2)