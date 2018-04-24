import os
import csv
import numpy as np
import keras

from run import train

input_file = "./ag_news_csv/train.csv"
max_feature_length = 4096 
num_classes = 4
embedding_size = 16
learning_rate = 0.001
batch_size = 100
num_epochs = 250

train(input_file,max_feature_length,num_classes,embedding_size,learning_rate,batch_size,num_epochs)