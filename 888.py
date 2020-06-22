from math import sqrt
import tensorflow as tf
from numpy import concatenate
from matplotlib import pyplot
import pathlib
import os
import lzma
import numpy as np
from pandas import read_csv
from keras.models import Model
import torch
import pandas as pd
from keras.layers import Input
from sklearn import metrics
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
import librosa
import librosa.display
from keras.layers import LSTM

dataset = pd.read_csv('D:/lernen/ai/data/mpi_roof_2016b.csv')
#dataset.drop((dataset.columns[0:1]), axis=1,inplace=True)
#dataset.to_csv('D:/lernen/ai/data/jena.csv')
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = dataset[features_considered]
features.index = dataset['Date Time']
features.head()


future_target = 20
past_history = 120
STEP = 6
TRAIN_SPLIT = 50000
tf.random.set_random_seed(13)
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  pyplot.title(title)
  for i, x in enumerate(plot_data):
    if i:
      pyplot.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
        pyplot.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  pyplot.legend()
  pyplot.xlim([time_steps[0], (future+5)*2])
  pyplot.xlabel('Time-Step')
  return pyplot

def create_time_steps(length):
  return list(range(-length, 0))

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  pyplot.figure()

  pyplot.plot(epochs, loss, 'b', label='Training loss')
  pyplot.plot(epochs, val_loss, 'r', label='Validation loss')
  pyplot.title(title)
  pyplot.legend()

  pyplot.show()

BATCH_SIZE = 256
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 200
EPOCHS = 10


x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()



def multi_step_plot(history, true_future, prediction):
  pyplot.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  pyplot.plot(num_in, np.array(history[:, 1]), label='History')
  pyplot.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    pyplot.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  pyplot.legend(loc='upper left')
  pyplot.show()


@tf.function
def com_loss(y_true, y_pred):
    return bytes(len(lzma.compress(bytes(y_true-y_pred))))

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(20))
multi_step_model.compile(loss=com_loss, optimizer='adam')

#for x, y in val_data_multi.take(1):
#  print (multi_step_model.predict(x).shape)

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

#plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')



for x, y in val_data_multi.take(3):

  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


true = y_val_multi
predict = multi_step_model.predict(x_val_multi)
n = len(true)
error = true - predict
print(len(true-predict))
print(len(lzma.compress(true-predict)))
mae = sum(np.abs(true - predict)) / n
pyplot.plot(mae, label='mae')
pyplot.legend()
pyplot.show()



'''
sequence = error
print(' error‘s shape : {}'.format(sequence.shape))
# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((n_in,20, 1))
# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(4, activation='relu', input_shape=(20,1)))
model.add(tf.keras.layers.RepeatVector(20))
model.add(tf.keras.layers.LSTM(4, activation='relu', return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(sequence, sequence, epochs=35,batch_size=1000, verbose=2)
model1 = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[0].output)
# demonstrate recreation
yhat = model1.predict(sequence)
new = model.predict(sequence).reshape(n_in,20)
print('compressed error‘s shape : {}'.format(yhat.shape))

newpre= true - new
newtrue= predict + new
maae = sum(np.abs(new)) / n
pyplot.plot(maae, label='mae')
pyplot.legend()
pyplot.show()

for x, y in val_data_multi.take(3):
  pre1=multi_step_model.predict(x)[0]
  error1=y - pre1
  sequence1= error1
  sequence1 = np.array(sequence1)
  sequence1= sequence1.reshape((256,20, 1))
  new1 = model.predict(sequence1).reshape(256, 20)
  newpre1 = y - new1
  newpre1 = np.array(newpre1)
  multi_step_plot(x[0], y[0], newpre1[0])
'''

