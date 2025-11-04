#  import modules
import math
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FixedLocator, FixedFormatter)
from pandas.core.frame import DataFrame
from numpy import genfromtxt
import os
import glob 
import random
import sys
import os
import time
# print(os.getcwd())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
#import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
sess = tf.Session(config=tf.ConfigProto(device_count={'gpu': 0}))
tf.compat.v1.disable_eager_execution()

# %tensorflow_version 1.x

start_time = time.time()

RMSE_list = []
MAE_list = []
MAPE_list = []

# prepare the input data: all explanatory vars and lncase
# df_mode ahead week prediction
df_mode = 0

# data_mode
# 1: historical dengue: input_size = 6
# 2: explanatory factors: input_size = 6
# 3：historical dengue + explanatory factors: input_size = 6
data_mode = "1"

is_train = True

# network mode
# 1: LSTM
# 2: LSTM-ATT
net_mode = "2"

for df_mode in range(1, 5): # #-ahead week prediction
  for data_mode in range(2, 4): # whether use historical dengue cases
    data_mode = str(data_mode)
    for i in range(1, 2):
      print("iteration：", str(i))
      print(str(df_mode) + '-week ahead, data_mode: ' + data_mode)
      seed = 25 + i
      random.seed(seed)
      np.random.seed(seed)
      tf.set_random_seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      tf.reset_default_graph()
      tf.set_random_seed(seed)



      # read the data
      file_name = 'DF.csv'
      # file_name = 'Fortaleza.csv'
      df = pd.read_csv('./data/' + file_name)
     

      date_list = df['date']
      # fill NaN in lncase_ using interpolate
      df.interpolate(method='linear', direction='forward', inplace=True)

      # prepare the input data: all explanatory vars and lncase
      # columns_list = ['Evap_mm_mean', 'pressure_pa_mean', 'Rain_mm_mean', 'relativehumidity_mean', 'Tair_c_mean', 'windspeed_ms_mean', 'lncase_0']
      columns_list = ['rain', 'Tair', 'rh', 'Mean_EVI', 'lncase_0']
      if df_mode != 0:
          columns_list.append("lncase_"+str(df_mode))

      df0 = pd.DataFrame(df, columns=columns_list)



      # Models and parameters  
      num_layers = 1   # the number of hidden layers
      rnn_unit = 64    # the number of rnn units
      attention_size = 64

      time_step = 12   # time step for lstm
      output_size = 1  # output dimension
      lr = 0.005     # learning rate
      keep_prob = 0.5  # dropout=1-keep_out
      batch_size = 12  
      is_training = True
      epochs = 1000

      if net_mode == '1': # LSTM
        if data_mode == '2':
          epochs = 1000
          lr = 0.005
          keep_prob = 0.2
        elif data_mode == '3':
          epochs = 1500
          lr = 0.003
          keep_prob = 0.3

      elif net_mode == '2': # LSTM-ATT
        if data_mode == '2':
          epochs = 1500
          lr = 0.005
          keep_prob = 0.2
        
        elif data_mode == '3':
          epochs = 2000
          lr = 0.003
          keep_prob = 0.3



      #    prepare the training, validation and testing dataset
      #    total number of time series is the multiples of batch size
      #    split time series data into training and testing sets
      #    the number of each dataset should be the multiples of batch size
      # 0-500,  500 for training
      train_begin = 0
      train_end = 314

      # 500-678, 178 for testing
      test_begin = 314
      test_end = 418 - df_mode

      data = df0.iloc[:, 0:].values
      # input_size = 6
      input_size = 4


      #   #################################### split data #################################
      def read_data():

        batch_index = []

        # 1. training data
        data_train = data[train_begin:train_end]
        train_y_plot = data_train[:, -1]
        train_mean = np.mean(train_y_plot)
        train_std = np.std(train_y_plot)
        normalized_train_data = (data_train-np.mean(data_train, axis=0))/np.std(data_train, axis=0)
        # normalized_train_data = data_train
        # print(normalized_train_data)

        train_x, train_y = [], []
        for i in range(len(normalized_train_data)-time_step):
          if i % batch_size == 0:
            batch_index.append(i)
            # 1: historical epi data
          if data_mode == "1":
            x = normalized_train_data[i:i+time_step, input_size]
            y = normalized_train_data[i+time_step-1, -1]
            # 2: explanatory variables
          elif data_mode == "2":
            x = normalized_train_data[i:i+time_step, :input_size]
            y = normalized_train_data[i+time_step-1, -1]
            # 3: historical epi data + explanatory variables
          else:
            x = normalized_train_data[i:i+time_step, :input_size+1]
            y = normalized_train_data[i+time_step-1, -1]

          train_x.append(x.tolist())
          train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data)-time_step))
        # print(normalized_train_data)
        # print(train_x)
        # print(train_y)
        # sys.exit(1)

        # 2. testing data
        data_test = data[test_begin:test_end]
        test_y = data_test[:, -1]

        mean_ = np.mean(test_y)
        std_ = np.std(test_y)

        normalized_test_data = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)
        test_x = []
        test_y = []
        for i in range(len(normalized_test_data)-time_step):
          if data_mode == "1":
            x = normalized_test_data[i:i+time_step, input_size]
            y = normalized_test_data[i+time_step-1, -1]
          elif data_mode == "2":
            x = normalized_test_data[i:i+time_step, :input_size]
            y = normalized_test_data[i+time_step-1, -1]
          else:
            x = normalized_test_data[i:i+time_step, :input_size+1]
            y = normalized_test_data[i+time_step-1, -1]

          test_x.append(x.tolist())
          test_y.append(y.tolist())

        train_x = np.array(train_x).reshape((len(normalized_train_data) - time_step,time_step,-1))
        train_y = np.array(train_y).reshape((len(normalized_train_data) - time_step, -1))
        test_x = np.array(test_x).reshape((len(normalized_test_data)-time_step,time_step,-1))
        test_y = np.array(test_y).reshape((len(normalized_test_data)-time_step,-1))

        return train_x, train_y, test_x, test_y, train_mean, train_std, mean_, std_, batch_index
      train_x, train_y, test_x, test_y, train_mean, train_std, mean_, std_, batch_index = read_data()




      #   #################################### train lstm #################################
      def lstm(X):
        #  rnn_unit
        features = train_x.shape[2]
        #print("features",features)
        weights = {
            'in': tf.Variable(tf.random_normal([features, rnn_unit])),
            'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = weights['in']
        b_in = biases['in']

        input = tf.reshape(X, [-1, features])
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])

        #  LSTM
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0, state_is_tuple=True)

        #  dropout
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)

        #  tf.nn.rnn_cell.MultiRNNCell
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)

        output = tf.reshape(output_rnn[:,-1,:], [-1, rnn_unit])
        #print(output)
        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

      def lstm_attention(X):
        features = train_x.shape[2]
        #print("features",features)
        weights = {
            'in': tf.Variable(tf.random_normal([features, rnn_unit])),
            'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

        batch_size = tf.shape(X)[0]
        time_step = train_x.shape[1]
        w_in = weights['in']
        b_in = biases['in']

        input = tf.reshape(X, [-1, features])
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0, state_is_tuple=True)

        if is_training:
          lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_cell, output_keep_prob=keep_prob)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        # output_rnn : batch siez * time step * unit number
        # print(output_rnn.shape)
        # print(final_states[0].shape)
        

        # attention
        W_a = tf.Variable(tf.random_normal([rnn_unit, attention_size], stddev=0.1))
        b_a = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        b_a2 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.tensordot(output_rnn, W_a, axes=1) + b_a)
        vu = tf.tensordot(v, b_a2, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        # the result has (B,D) shape
        output = tf.reduce_sum(output_rnn * tf.expand_dims(alphas, -1), 1)

        # W_q = tf.Variable(tf.random_normal([rnn_unit, attention_size], stddev=0.1))
        # W_k = tf.Variable(tf.random_normal([rnn_unit, attention_size], stddev=0.1))
        # W_v = tf.Variable(tf.random_normal([rnn_unit, rnn_unit], stddev=0.1))
        # query = tf.tensordot(output_rnn, W_q, axes=1)
        # key = tf.tensordot(output_rnn, W_k, axes=1)
        # value = tf.tensordot(output_rnn, W_v, axes=1)
        # att_weight = tf.matmul(query, tf.transpose(key, perm=[0,2,1])) / sqrt(attention_size)
        # att_weight = tf.nn.softmax(att_weight)
        # output = tf.matmul(att_weight, value)
        # output = tf.reduce_sum(output, 1)

        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states


      def train_lstm(sess, train_x, train_y, batch_index):
        losses_train = []
        losses_val = []
        epoch_start=time.time()
        for i in range(epochs):
          loss_epoch = 0.
          loss_val_epoch = 0.
          for step in range(len(batch_index)-1):
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            loss_val = sess.run([loss], feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            loss_epoch += loss_
            loss_val_epoch += loss_val[0]
          losses_val.append(loss_epoch/step)
          losses_train.append(loss_val_epoch/step)
          if (i+1)%50 == 0:
            print("Number of epochs:", i+1, " loss:", loss_epoch/step, "val loss:", loss_val_epoch/step,
                  'running time:', time.time()-epoch_start)
            # saver.save(sess, 'my_model', global_step=i+1)
        print("The train has finished")
        saver.save(sess, 'lstm_model', global_step=epochs)
        return losses_train, losses_val



      # train and save lstm models
      X = tf.placeholder(tf.float32, shape=[None, time_step, train_x.shape[2]])
      Y = tf.placeholder(tf.float32, shape=[None, output_size])
      if net_mode == "1":
          pred,_ = lstm(X)
      else:
          # pred,_ = lstm_attention(X)
          pred,_ = lstm(X)
      loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
      train_op = tf.train.AdamOptimizer(lr).minimize(loss)


      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=150000)
      if is_train:
        losses_train, losses_val = train_lstm(sess, train_x, train_y, batch_index)
      else:
        ckpt = tf.train.get_checkpoint_state('')
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)

      #  ########################## test lstm ########################
      def test_lstm(sess, test_x):
        predict_y = []
        for step in range(len(test_x)):
            pred_ = sess.run(pred, feed_dict={X:[test_x[step]]})
            predict = pred_.reshape((-1))
            predict_y.append(predict[-1])
        return np.array(predict_y)

      # use trained model to produce pred_y
      # the first pred_y is df[test_begin:test_end]['date'][time_step-1:]
      pred_y = test_lstm(sess, test_x)
      
      pred_y_test_plot = pred_y * std_ + mean_

      pred_y_train = test_lstm(sess, train_x)
      pred_y_train_plot = pred_y_train * train_std + train_mean

      def test_loss(pred_y,test_y):
        n = 0
        y_rmse = 0
        y_mae = 0
        y_mape = 0

        x = []
        y1 = []
        y2 =[]
        for id in range(len(pred_y)):
            x.append(id)
            result = list(pred_y)[id]
            origial = list(test_y)[id][0]
            y1.append(result)
            y2.append(origial)
            #print(result,origial)
            y_rmse = y_rmse + pow((origial - result), 2)
            y_mae = y_mae + abs(origial - result)
            y_mape = y_mape + (abs(origial - result) / abs(origial))
            n += 1

        RMSE = sqrt(y_rmse / n)
        MAE = y_mae / n
        MAPE = y_mape / n
        print("RMSE:", round(RMSE, 4))
        print("MAE:", round(MAE, 4))
        # print("MAPE:", round(MAPE, 4))
        return RMSE, MAE, MAPE

      RMSE, MAE, MAPE = test_loss(pred_y,test_y)
      RMSE_list.append(RMSE)
      MAE_list.append(MAE)
      MAPE_list.append(MAPE)
      df_out = pd.DataFrame()

      df_out['date'] = date_list[test_begin+time_step:test_end]
      df_out["pred_y"] = pred_y_test_plot
      df_out["test_y"] = test_y * std_ + mean_
      df_out = df_out.set_index("date")

      df_out.to_csv('./output_lstm/' + file_name + "_LSTM_"+str(df_mode)+"-week-ahead"+"_input_"+str(data_mode)+"_split_"+str(train_end)+"_round_"+str(i)+".csv")

print('OK')
print('running time: ', time.time()-start_time)
