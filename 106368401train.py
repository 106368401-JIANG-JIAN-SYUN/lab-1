import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error


data_1=pd.read_csv("/media/lab21010/lab21010_5tb/train-v3.csv")
X_train=data_1.drop(['price','id'],axis=1).values
Y_train=data_1['price'].values

data_2=pd.read_csv("/media/lab21010/lab21010_5tb/valid-v3.csv")
X_valid=data_2.drop(['price','id'],axis=1).values
Y_valid=data_2['price'].values

data_3=pd.read_csv("/media/lab21010/lab21010_5tb/test-v3.csv")
X_test=data_3.drop('id',axis=1).values


def normalize(train,valid,test):
    tmp=train
    mean=tmp.mean(axis=0)
    std=tmp.std(axis=0)    
    print("tmp.shape=",tmp.shape)
    print("mean.shape=",mean.shape)
    print("std.shape=",std.shape)
    print("mean=",mean)
    print("std=",std)
    train=(train-mean)/std
    valid=(valid-mean)/std
    test=(test-mean)/std
    return train,valid,test

X_train,X_valid,X_test=normalize(X_train,X_valid,X_test)


from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(50,input_dim=21,init='normal',activation='relu'))
model.add(Dense(116,input_dim=50,init='normal',activation='relu'))
model.add(Dense(158,input_dim=116,init='normal',activation='relu'))
model.add(Dense(144,input_dim=158,init='normal',activation='relu'))
model.add(Dense(90,input_dim=144,init='normal',activation='relu'))
model.add(Dense(60,input_dim=90,init='normal',activation='relu'))
model.add(Dense(1 ,init='normal'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,Y_train,batch_size=20,nb_epoch=210,validation_data=(X_valid,Y_valid))



Y_predict=model.predict(X_test) #test data pred price

valid_predict=model.predict(X_valid) #valid data true price
train_predict=model.predict(X_train) #train data true price


n=len(Y_predict)+1
for i in range(1,n):
    b= np.arange(1,n,1)   
b=np.transpose([b])
Y=np.column_stack((b,Y_predict))


plt.figure()
plt.scatter(Y_train, train_predict , c='r')

MAE_train = mean_absolute_error(Y_train, train_predict)
print("\n")
print (MAE_train)

plt.figure()
plt.scatter(Y_valid, valid_predict )


MAE_valid = mean_absolute_error(Y_valid, valid_predict)
print("\n")
print (MAE_valid)


np.savetxt('106368401_test.csv',Y,delimiter=',',fmt='%i')