import os
import re
import numpy as np
from keras import utils
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# define parameters
n_features = 3
seq_length = 50

# load dataset
dataset_path = './dataset'
file_list = os.listdir(dataset_path)

# sort files by the number in file name
file_list.sort(key=lambda f: int(re.sub('\D', '', f)))

# load all data
data_list = []
for file in file_list:
    data_path = os.path.join(dataset_path, file)
    data = np.load(data_path, allow_pickle=True)
    data_list.append(data)

# concatenate all data
data = np.concatenate(data_list, axis=0)

# prepare sequences
X, y = [], []
for i in range(0, data.shape[0] - seq_length, seq_length):
    X.append(data[i:i+seq_length, :n_features])
    y.append(data[i+seq_length, :n_features])

X = np.array(X)
y = np.array(y)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# define model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, n_features)))
model.add(Dense(n_features, activation='linear'))

# compile model
model.compile(loss='mse', optimizer='adam')

# train model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# save model
model.save('model.h5')
