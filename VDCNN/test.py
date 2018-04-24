import os
import csv
import numpy as np
import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import load_model

from run import get_input_data
from utils import to_categorical, get_comment_ids

input_file = "./ag_news_csv/test.csv"
max_feature_length = 4096
num_classes = 4
embedding_size = 16
learning_rate = 0.001
batch_size = 100
num_epochs = 1000

X_train, y_train = get_input_data(input_file)
size = len(y_train)
y_train = to_categorical(y_train, num_classes)
X = []
for row in X_train:
    X.append(get_comment_ids(row))
X_train = np.zeros([size,512])
for i in range(size):
    X_train[i] = X[i]

model = load_model("./model.h5",custom_objects={"tf": tf})
output = model.predict(X_train,batch_size=100,verbose=0,steps=None)
result = np.zeros((size,2), dtype=int)
for i in range(size):
	result[i][0] = int(i)
	result[i][1] = int(np.argmax(output[i]))

#print(result)
writer=csv.writer(open('result.csv','w'))
title = ['id', 'prediction']
writer.writerow(title)
for row in result:
	writer.writerow(row)

count = 0
for i in range(size):
    for j in range(4):
        if(y_train[i][j] != 0):
            re = j
    if(result[i][1] == re):
        count = count + 1
accuracy = (count/size)*100
print("The accuracy is",accuracy,"%.")
