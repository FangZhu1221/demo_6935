import os
import csv
import numpy as np
from keras.callbacks import ModelCheckpoint
from utils import to_categorical, get_comment_ids
from vdcnn import build_model

csv.field_size_limit(100000000)

def get_input_data(file_path):
	datas = []
	labels = []

	with open(file_path) as f:
		reader = csv.reader(f)
		for row in reader:
			labels.append(row[0])
			datas.append(row[1])
	return datas,labels



def train(input_file, max_feature_length, num_classes, embedding_size, learning_rate, batch_size, num_epochs, save_dir=None, print_summary=False):
    # Stage 1: Convert raw texts into char-ids format && convert labels into one-hot vectors
    X_train, y_train = get_input_data(input_file)
    y_train = to_categorical(y_train, num_classes)
    X = []
    for row in X_train:
        X.append(get_comment_ids(row));
    X_train = np.zeros([len(y_train),512])
    for i in range(len(y_train)):
    	X_train[i] = X[i]
    # Stage 2: Build Model
    num_filters = [64, 128, 256, 512]

    model = build_model(num_filters=num_filters, num_classes=num_classes, embedding_size=embedding_size, learning_rate=learning_rate)

    # Stage 3: Training
    #save_dir = save_dir if save_dir is not None else 'checkpoints'
    #filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    #if print_summary:
        #print(model.summary())

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.33,
        #callbacks=[checkpoint],
        shuffle=True,
        verbose=True
    )
    score,acc = model.evaluate(X_train, y_train, batch_size=batch_size)
    print('Test score:',score)
    print('Test accuracy:',acc)
    model.save("./model.h5")

