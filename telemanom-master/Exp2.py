import numpy as np
import os
import logging
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.io
from scipy.signal import savgol_filter
from tensorflow.keras.layers import Input, Dense, Lambda, RepeatVector, TimeDistributed, LSTM, Dense, Activation, Dropout
from tensorflow.keras.callbacks import History, EarlyStopping, Callback
# from tensorflow.keras.objectives import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import History, EarlyStopping, Callback, LearningRateScheduler, History
# from tensorflow.keras.layers.recurrent import LSTM
# from tensorflow.keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.losses import mse, binary_crossentropy
import time
# import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard






def AE_generate(arr, chan_id, train=True, l=5, e=5, t=50):


    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    tim = TimeHistory()
    history= History()
    stopper = EarlyStopping(monitor='loss', patience=4,  min_delta=0.0003, verbose=0)


    tf.compat.v1.disable_eager_execution()

    # cbs = [History(), EarlyStopping(monitor='loss',
    #                             min_delta=0.0003,
    #                             verbose=0)]

    l_s = t
    n_predictions = 10
    latent_dim = l
    data = []
    for i in range(len(arr) - l_s):
        data.append(arr[i:i + l_s])
    data = np.array(data)
    assert len(data.shape) == 3

    start_timer1=time.time()
    intermediate_dim = int(round((data.shape[2]+l)/2))

    input_img = Input(shape=(data.shape[1], data.shape[2]))
    encoded = LSTM(intermediate_dim, activation='relu',
                   return_sequences=True)(input_img)
    # encoded = RepeatVector(data.shape[1])(encoded)
    encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(encoded)

    decoded = RepeatVector(data.shape[1])(encoded)
    decoded = LSTM(intermediate_dim, activation='relu',
                   return_sequences=True)(decoded)
    # decoded = RepeatVector(data.shape[1])(decoded)
    decoded = LSTM(data.shape[2], activation='relu', return_sequences=True)(decoded)
    
    decoded = TimeDistributed(Dense(data.shape[2]))(decoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.summary()
    autoencoder.compile(loss='mse', optimizer='adam')
    history = autoencoder.fit(data, data,
                                batch_size=64,
                                epochs=e,
                                verbose=True,
                                callbacks = [tim, history, stopper])
                                # callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./tmp/autoencoder')])                                # callbacks=[tim, history])
                                # callbacks=cbs)
    recon = autoencoder.predict(data)
    data = encoder.predict(data)

    total = time.time()-start_timer1
    loss = history.history['loss']
    epochs = list(range(1, e+1))
    History_data = [epochs, loss, [total]]
    History_data = np.array(History_data)

    if train:
        np.save(os.path.join("data", "AE", "Exp1", "train", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "AE", "Exp1", "train", "history", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)
        np.save(os.path.join("data", "AE", "Exp1", "train", "recon", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), recon)

    else:
        np.save(os.path.join("data", "AE", "Exp1", "test", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "AE", "Exp1", "test", "history", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)
        np.save(os.path.join("data", "AE", "Exp1", "test", "recon", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), recon)


def VAE_generate(arr, chan_id, train=True, l=5, e=5, t=50):

    # cbs = [History(), EarlyStopping(monitor='loss',
    #                                 patience=4,
    #                                 min_delta=0.0003,
    #                                 verbose=0)]

    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    tim = TimeHistory()
    history= History()
    stopper = EarlyStopping(monitor='loss', patience=4, min_delta=0.0003, verbose=0)


    tf.compat.v1.disable_eager_execution()
    l_s = t
    n_predictions = 10
    print("arr Shape:  {} ".format(arr.shape))

    data = []
    for i in range(len(arr) - l_s):
        data.append(arr[i:i + l_s])
    data = np.array(data)
    assert len(data.shape) == 3


    intermediate_dim = int(round((data.shape[2]+l)/2))
    original_dim = data.shape[2]
    timesteps = data.shape[1]
    batch_size = 64
    latent_dim = l
    epochs = e
    epsilon_std=1.

    start_timer1=time.time()

    def sampling(args):
        z_mean, z_log_sigma = args
        # Uses batch size trick from [reference] trick from guy
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon


    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        # Removed axis -1 param
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        return xent_loss + kl_loss

    x = Input(shape=(data.shape[1], data.shape[2]))

    # Intermediate Node
    h = LSTM(intermediate_dim, activation='relu', return_sequences=True)(x)

    # Mean and Covarience Node
    z_mean = LSTM(latent_dim, activation=None, return_sequences=False)(h)
    z_log_sigma = LSTM(latent_dim, activation=None, return_sequences=False)(h)

    # Sampling Node
    z = Lambda(sampling)([z_mean, z_log_sigma])
    # Decoder
    z_repeat_vector = RepeatVector(timesteps)(z)

    # Decoder, train using outputted mean
    h_decoded = LSTM(intermediate_dim, activation='relu', return_sequences=True)(z_repeat_vector)
    x_decoded_mean = LSTM(original_dim, activation='sigmoid', return_sequences=True)(h_decoded)
    
    vae = Model(x, x_decoded_mean)
    encoder = Model(x, z_mean)

    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.summary()
    history = vae.fit(data, data,
                                batch_size=64,
                                epochs=e,
                                verbose=True,
                                callbacks=[tim, history, stopper])
                                # callbacks = cbs)
                    #   callbacks=[tim, history])
                                # callbacks=[tim, history])

    recon = vae.predict(data)
    data = encoder.predict(data)


    total = time.time()-start_timer1
    loss = history.history['loss']
    epochs = list(range(1,e+1))
    History_data = [epochs, loss, [total]]
    History_data = np.array(History_data)


    if train:
        np.save(os.path.join("data", "VAE", "Exp1", "train", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "VAE", "Exp1", "train", "history", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)
        np.save(os.path.join("data", "VAE", "Exp1", "train", "recon", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), recon)

    else:
        np.save(os.path.join("data", "VAE", "Exp1", "test", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "VAE", "Exp1", "test", "history", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)
        np.save(os.path.join("data", "VAE", "Exp1", "test", "recon", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), recon)



# For experiment 2  - encoder from trained data is used to predict test data - all combined

def AE_generateOG1(arr, arr1, chan_id, train=True, l=5, e=5, t=50):

    # class Timer(Callback):
    #     def __init__(self, logs={}):
    #         self.times=[]
    #     def start(self, batch, logs={}):
    #         self.start_timer=time()
    #     def end(self, batch, logs={}):
    #         self.times.append(time()-self.start_timer)

    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    tim = TimeHistory()
    history= History()
    stopper = EarlyStopping(monitor='loss', patience=4,  min_delta=0.0003, verbose=0)
    cbs = [tim, history, stopper]

    tf.compat.v1.disable_eager_execution()

    # cbs = [History(), EarlyStopping(monitor='loss',
    #                             min_delta=0.0003,
    #                             verbose=0)]


    l_s = t
    n_predictions = 0
    latent_dim = l


    data = []
    for i in range(len(arr) - l_s - n_predictions):
        data.append(arr[i:i + l_s + n_predictions])
    data = np.array(data)
    assert len(data.shape) == 3


    data1 = []
    for i in range(len(arr1) - l_s - n_predictions):
        data1.append(arr1[i:i + l_s + n_predictions])
    data1 = np.array(data1)
    assert len(data1.shape) == 3


    intermediate_dim = int(round((data.shape[2]+l)/2))

    start_timer1 = time.time()
    input_img = Input(shape=(data.shape[1], data.shape[2]))
    encoded = LSTM(intermediate_dim, activation='relu',
                   return_sequences=True)(input_img)
    # encoded = RepeatVector(data.shape[1])(encoded)
    encoded = LSTM(latent_dim, activation='relu',
                   return_sequences=False)(encoded)

    decoded = RepeatVector(data.shape[1])(encoded)
    decoded = LSTM(intermediate_dim, activation='relu',
                   return_sequences=True)(decoded)
    # decoded = RepeatVector(data.shape[1])(decoded)
    decoded = LSTM(data.shape[2], activation='relu',
                   return_sequences=True)(decoded)

    decoded = TimeDistributed(Dense(data.shape[2]))(decoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.summary()
    autoencoder.compile(loss='mse', optimizer='adam')
    history = autoencoder.fit(data, data,
                              batch_size=64,
                              epochs=e,
                              verbose=True,
                              callbacks=[tim, history, stopper])
                            #   callbacks=cbs)
                            
    data = encoder.predict(data)
    total = time.time()-start_timer1
    # times = tim.times
    loss = history.history['loss']
    epochs = list(range(1, e+1))
    # History_data = [epochs, loss, times, [total]]
    History_data = [epochs, loss, [total]]
    History_data = np.array(History_data)

    traindata = np.load(os.path.join("data", "train", "{}.npy".format(chan_id))) 
    arr = data
    traindata = traindata[t:]
    arr = np.insert(arr, 0, 0, axis=1)
    for i in range(0, arr.shape[0]):
        arr[i][0] = traindata[i][0]
    np.save(os.path.join("data", "AE", "Exp2", "train", "{}.npy".format(chan_id)), arr)
    np.save(os.path.join("data", "AE", "Exp2", "train", "history","{}.npy".format(chan_id)), History_data)


    data = encoder.predict(data1)
    traindata = np.load(os.path.join("data", "test", "{}.npy".format(chan_id)))  
    arr = data
    traindata = traindata[t:]
    arr = np.insert(arr, 0, 0, axis=1)
    for i in range(0, arr.shape[0]):
        arr[i][0] = traindata[i][0]
    np.save(os.path.join("data", "AE", "Exp2", "test", "{}.npy".format(chan_id)), arr)
    np.save(os.path.join("data", "AE", "Exp2", "test", "history","{}.npy".format(chan_id)), History_data)



def VAE_generateOG1(arr, arr1, chan_id, train=True, l=5, e=5, t=50):

    # class Timer(Callback):
    #     def __init__(self, logs={}):
    #         self.times=[]
    #     def start(self, batch, logs={}):
    #         self.start_timer=time()
    #     def end(self, batch, logs={}):
    #         self.times.append(time()-self.start_timer)

    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    tim = TimeHistory()
    history= History()
    stopper = EarlyStopping(monitor='loss', patience=4,  min_delta=0.0003, verbose=0)
    cbs = [tim, history, stopper]
    # cbs = [EarlyStopping(monitor='loss',
    #                      patience=4,
    #                      min_delta=0.0003,
    #                      verbose=0)]

    tf.compat.v1.disable_eager_execution()


    l_s = t
    n_predictions = 0
    latent_dim = l
    data = []
    for i in range(len(arr) - l_s - n_predictions):
        data.append(arr[i:i + l_s + n_predictions])
    data = np.array(data)
    assert len(data.shape) == 3


    data1 = []
    for i in range(len(arr1) - l_s - n_predictions):
        data1.append(arr1[i:i + l_s + n_predictions])
    data1 = np.array(data1)
    assert len(data1.shape) == 3


    intermediate_dim = int(round((data.shape[2]+l)/2))
    original_dim = data.shape[2]
    timesteps = data.shape[1]
    batch_size = 64
    latent_dim = l
    epochs = e
    epsilon_std=1.

    start_timer1=time.time()

    def sampling(args):
        z_mean, z_log_sigma = args
        # Uses batch size trick from [reference] trick from guy
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon


    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        # Removed axis -1 param
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        return xent_loss + kl_loss

    x = Input(shape=(data.shape[1], data.shape[2]))

    # Intermediate Node
    h = LSTM(intermediate_dim, activation='relu', return_sequences=True)(x)

    # Mean and Covarience Node
    z_mean = LSTM(latent_dim, activation=None, return_sequences=False)(h)
    z_log_sigma = LSTM(latent_dim, activation=None, return_sequences=False)(h)

    # Sampling Node
    z = Lambda(sampling)([z_mean, z_log_sigma])
    # Decoder
    z_repeat_vector = RepeatVector(timesteps)(z)

    # Decoder, train using outputted mean
    h_decoded = LSTM(intermediate_dim, activation='relu', return_sequences=True)(z_repeat_vector)
    x_decoded_mean = LSTM(original_dim, activation='sigmoid', return_sequences=True)(h_decoded)
    
    vae = Model(x, x_decoded_mean)
    encoder = Model(x, z_mean)

    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.summary()
    history = vae.fit(data, data,
                                batch_size=64,
                                epochs=e,
                                verbose=True,
                                callbacks=[tim, history, stopper])





    total = time.time()-start_timer1
    # times = tim.times
    loss = history.history['loss']
    epochs = list(range(1, e+1))
    # History_data = [epochs, loss, times, [total]]
    History_data = [epochs, loss, [total]]
    History_data = np.array(History_data)

    data = encoder.predict(data)
    traindata = np.load(os.path.join("data", "train", "{}.npy".format(chan_id))) 
    arr = data
    traindata = traindata[t:]
    arr = np.insert(arr, 0, 0, axis=1)
    for i in range(0, arr.shape[0]):
        arr[i][0] = traindata[i][0]
    np.save(os.path.join("data", "VAE", "Exp2", "train", "{}.npy".format(chan_id)), arr)
    np.save(os.path.join("data", "VAE", "Exp2", "train", "history","{}.npy".format(chan_id)), History_data)


    data = encoder.predict(data1)
    traindata = np.load(os.path.join("data", "test", "{}.npy".format(chan_id)))  
    arr = data
    traindata = traindata[t:]
    arr = np.insert(arr, 0, 0, axis=1)
    for i in range(0, arr.shape[0]):
        arr[i][0] = traindata[i][0]
    np.save(os.path.join("data", "VAE", "Exp2", "test", "{}.npy".format(chan_id)), arr)
    np.save(os.path.join("data", "VAE", "Exp2", "test", "history","{}.npy".format(chan_id)), History_data)













def AE_generateOG(arr, chan_id, train=True, l=5, e=5, t=50):

    # class Timer(Callback):
    #     def __init__(self, logs={}):
    #         self.times=[]
    #     def start(self, batch, logs={}):
    #         self.start_timer=time()
    #     def end(self, batch, logs={}):
    #         self.times.append(time()-self.start_timer)

    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    tim = TimeHistory()
    history= History()
    stopper = EarlyStopping(monitor='loss', patience=4,  min_delta=0.0003, verbose=0)

    # cbs = [EarlyStopping(monitor='loss',
    #                      patience=4,
    #                      min_delta=0.0003,
    #                      verbose=0)]

    tf.compat.v1.disable_eager_execution()

    # cbs = [History(), EarlyStopping(monitor='loss',
    #                             min_delta=0.0003,
    #                             verbose=0)]


    l_s = t
    n_predictions = 10
    latent_dim = l
    # history = History()
    data = []
    for i in range(len(arr) - l_s):
        data.append(arr[i:i + l_s])
    data = np.array(data)
    assert len(data.shape) == 3


    intermediate_dim = int(round((data.shape[2]+l)/2))


    start_timer1 = time.time()
    input_img = Input(shape=(data.shape[1], data.shape[2]))
    encoded = LSTM(intermediate_dim, activation='relu',
                   return_sequences=True)(input_img)
    # encoded = RepeatVector(data.shape[1])(encoded)
    encoded = LSTM(latent_dim, activation='relu',
                   return_sequences=False)(encoded)

    decoded = RepeatVector(data.shape[1])(encoded)
    decoded = LSTM(intermediate_dim, activation='relu',
                   return_sequences=True)(decoded)
    # decoded = RepeatVector(data.shape[1])(decoded)
    decoded = LSTM(data.shape[2], activation='relu',
                   return_sequences=True)(decoded)

    decoded = TimeDistributed(Dense(data.shape[2]))(decoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.summary()
    autoencoder.compile(loss='mse', optimizer='adam')
    history = autoencoder.fit(data, data,
                              batch_size=64,
                              epochs=e,
                              verbose=True,
                            #   validation_split=0.2,
                              callbacks=[tim, history, stopper])
                            #   callbacks=cbs)
                            
    data = encoder.predict(data)

    total = time.time()-start_timer1
    loss = history.history['loss']
    epochs = list(range(1, e+1))
    History_data = [epochs, loss, [total]]
    History_data = np.array(History_data)

    if train:
        np.save(os.path.join("data", "AE", "train", "encoded", "OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "AE", "train", "history", "OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)

    else:
        np.save(os.path.join("data", "AE", "test", "encoded" ,"OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "AE", "test", "history", "OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)


def VAE_generateOG(arr, chan_id, train=True, l=5, e=5, t=50):

    # cbs = [History(), EarlyStopping(monitor='loss',
    #                                 patience=4,
    #                                 min_delta=0.0003,
    #                                 verbose=0)]

    # cbs = [EarlyStopping(monitor='loss',
    #                      patience=4,
    #                      min_delta=0.0003,
    #                      verbose=0)]

    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    tim = TimeHistory()
    history = History()
    stopper = EarlyStopping(monitor='loss', patience=4, min_delta=0.0003, verbose=0)
    cbs = [tim, history, stopper]
    tf.compat.v1.disable_eager_execution()

    l_s = t
    n_predictions = 10
    print("arr Shape:  {} ".format(arr.shape))

    data = []
    for i in range(len(arr) - l_s):
        data.append(arr[i:i + l_s])
    data = np.array(data)
    assert len(data.shape) == 3
    # history = History()

    epsilon_std = 1.

    start_timer1 = time.time()

    intermediate_dim = int(round((data.shape[2]+l)/2))

    def sampling(args):
        z_mean, z_log_sigma = args
        # Uses batch size trick from [reference] trick from guy
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon


    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        # Removed axis -1 param
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        return xent_loss + kl_loss

    x = Input(shape=(data.shape[1], data.shape[2]))

    # Intermediate Node
    h = LSTM(intermediate_dim, activation='relu', return_sequences=True)(x)

    # Mean and Covarience Node
    z_mean = LSTM(latent_dim, activation=None, return_sequences=False)(h)
    z_log_sigma = LSTM(latent_dim, activation=None, return_sequences=False)(h)

    # Sampling Node
    z = Lambda(sampling)([z_mean, z_log_sigma])
    # Decoder
    z_repeat_vector = RepeatVector(timesteps)(z)

    # Decoder, train using outputted mean
    h_decoded = LSTM(intermediate_dim, activation='relu', return_sequences=True)(z_repeat_vector)
    x_decoded_mean = LSTM(original_dim, activation='sigmoid', return_sequences=True)(h_decoded)
    
    vae = Model(x, x_decoded_mean)
    encoder = Model(x, z_mean)

    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.summary()
    history = vae.fit(data, data,
                                batch_size=64,
                                epochs=e,
                                verbose=True,
                                validation_split=0.2,
                                callbacks=[tim, history, stopper])



    data = encoder.predict(data)
    l_s = 250
    total = time.time()-start_timer1
    loss = history.history['loss']
    epochs = list(range(1, e+1))
    History_data = [epochs, loss, [total]]
    History_data = np.array(History_data)

    if train:
        np.save(os.path.join("data", "VAE", "train", "encoded", "OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "VAE", "train", "history", "OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)


    else:
        np.save(os.path.join("data", "VAE", "test", "encoded" ,"OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
        np.save(os.path.join("data", "VAE", "test", "history", "OGE{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)









#####################################################################
# Print Channel Dimmeniosns
#####################################################################
chan_ids = [x.split('.')[0] for x in os.listdir('data/train/')]
removals = ['A-5', 'A-6', 'A-8', 'A-9', 'C-2', 'D-12']
channels  = [i for i in chan_ids if i not in removals]


# SMAP
# channels = ['P-1', 'S-1', 'E-1', 'D-12', 'P-7']
# channels = ['P-1']
# # MSL
# channels = ['M-5', 'F-7', 'P-14', 'T-4', 'C-2']
channels = ['P-1', 'S-1', 'E-1', 'D-12', 'P-7', 'E-5', 'M-5', 'F-7', 'P-14', 'T-4', 'C-2', 'T-13']

# for i in channels:
#     train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
#     test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
#     print("Channel ID: {}".format(i))
#     print("Train {}".format(train.shape))
#     print("Test {}".format(test.shape))




#####################################################################
# Experiment 1
######################################################################


#####################################################################
# Create AE data
# #####################################################################
for i in channels:
    print("Channel ID: {}".format(i))
    train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
    train = np.delete(train, 0, 1)
    test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
    test = np.delete(test, 0, 1)
    # AE_generate(train, i, True, 10, 1, 100)
    # AE_generate(test, i, False, 10, 30, 100)

#####################################################################
# Create VAE data
#####################################################################
for i in channels:
    print("Channel ID: {}".format(i))
    train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
    train = np.delete(train, 0, 1)
    test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
    test = np.delete(test, 0, 1)
    # VAE_generate(train, i, True, 10, 1, 100)
    # VAE_generate(test, i, False, 10, 30, 100)

#####################################################################
# Create Merged Data Ex1
#####################################################################
for i in channels:
    print("Channel ID: {}".format(i))
    train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
    test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
    # AE_generateOG(train, i, True, 10, 30, 100)
    # AE_generateOG(test, i, False, 10, 30, 100)-
    # VAE_generateOG(train, i, True, 10, 30, 100)
    # VAE_generateOG(test, i, False, 10, 30, 100)-

#####################################################################
# Experiment 2
######################################################################

#####################################################################
# Create VAE data merged
####################################################################
for i in channels:
    print("Channel ID: {}".format(i))
    train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
    train = np.delete(train, 0, 1)
    test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
    test = np.delete(test, 0, 1)
    AE_generateOG1(train, test, i, True, 10, 40, 100)
    VAE_generateOG1(train, test, i, True, 10, 40, 100)









#####################################################################
# Denoising
#####################################################################

# Extracting Telem
# chan_ids = [x.split('.')[0] for x in os.listdir('data/test/')]
# chan_ids.remove('')
# chan_ids = np.array(chan_ids)
# np.savetxt("ids.txt", chan_ids, delimiter=" ", newline = "\n", fmt="%s")

# removals = ['A-5', 'A-6', 'A-8', 'A-9', 'C-2', 'D-12']
# channels  = [i for i in chan_ids if i not in removals]
channels = ['P-1']

# for i in channels:
#     train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
#     test = np.load(os.path.join("data", "test", "{}.npy".format(i)))

#     j=[]; x=[]
#     for k in range(0, train.shape[0]):
#         j.append(train[k][0])
#     for k in range(0, test.shape[0]):
#         x.append(test[k][0])
#     j = savgol_filter(j, 41, 2)
#     x = savgol_filter(x, 41, 2)

#     d_train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
#     d_train_tel = j

#     d_test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
#     d_test_tel = x

#     for k in range(0, d_train.shape[0]-1):
#         d_train[k][0] = d_train_tel[k]

#     for k in range(0, d_test.shape[0]-1):
#         d_test[k][0] = d_test_tel[k]

#     denoise_AE_generate(train, d_train, i, True, 1, 10)
#     denoise_AE_generate(test, d_test, i, False, 1, 10)


#####################################################################
# Denoising OG
#####################################################################
# for i in channels:
#     train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
#     test = np.load(os.path.join("data", "test", "{}.npy".format(i)))

#     print(train.shape)
#     print(test.shape)

#     j = []
#     x = []
#     for k in range(0, train.shape[0]):
#         j.append(train[k][0])
#     for k in range(0, test.shape[0]):
#         x.append(test[k][0])
#     j = savgol_filter(j, 41, 2)
#     x = savgol_filter(x, 41, 2)

#     d_train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
#     d_train_tel = j

#     d_test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
#     d_test_tel = x

#     for k in range(0, d_train.shape[0]-1):
#         d_train[k][0] = d_train_tel[k]

#     for k in range(0, d_test.shape[0]-1):
#         d_test[k][0] = d_test_tel[k]
    
#     print(d_train.shape)
#     print(d_test.shape)

#     # denoise_AE_generate_OG(d_train, i, True, 5, 10)
#     # denoise_AE_generate_OG(d_test, i, False, 5, 10)

#     np.save(os.path.join("data", "train", "d_{}.npy".format(i)), d_train)
#     np.save(os.path.join("data", "test", "d_{}.npy".format(i)), d_test)


# for i in channels:
#     train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
#     test = np.load(os.path.join("data", "test", "{}.npy".format(i)))

#     d_train = np.load(os.path.join("data", "train", "{}.npy".format(i)))
#     d_train_tel = np.array(pd.read_csv("Train-Denoised-{}.csv".format(i)))
#     print(d_train_tel.shape)
#     print(d_train_tel)

#     d_test = np.load(os.path.join("data", "test", "{}.npy".format(i)))
#     d_test_tel = np.array(pd.read_csv("Test-Denoised-{}.csv".format(i)))

#     for k in range(0, d_train.shape[0]-1):
#         d_train[k][0] = d_train_tel[k][0]

#     for k in range(0, d_test.shape[0]-1):
#         d_test[k][0] = d_test_tel[k][0]

    # denoise_AE_generate(train, d_train, i, True, 5, 10)
    # denoise_AE_generate(test, d_test, i, False, 5, 10)
    # denoise_AE_generate(train, d_train, i, False, 5, 5)
    # denoise_AE_generate(test, d_test, i, False, 5, 5)







# train = np.load(os.path.join("data", "train", "{}.npy".format(i)))





























# def AE_generate(arr, chan_id, train=True, l=5, e=5):

#     l_s = 250
#     n_predictions = 10
#     latent_dim = l
#     history = History()

    # data = []
    # for i in range(len(arr) - l_s - n_predictions):
    #     data.append(arr[i:i + l_s + n_predictions])
    # data = np.array(data)
    # assert len(data.shape) == 3

    # input_img = Input(shape=(data.shape[1], data.shape[2]))
    # encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(input_img)

    # decoded = RepeatVector(data.shape[1])(encoded)
    # decoded = LSTM(data.shape[2], activation='relu', return_sequences=True)(decoded)
    # decoded = TimeDistributed(Dense(data.shape[2]))(decoded)


    # autoencoder = Model(input_img, decoded)
    # encoder = Model(input_img, encoded)
    # autoencoder.summary()
    # autoencoder.compile(loss='mse', optimizer='adam')

    # history = autoencoder.fit(data, data,
    #                             batch_size=64,
    #                             epochs=e,
    #                             verbose=True,
    #                             callback=[history])

    # data = encoder.predict(data)


    # if train:
    #     np.save(os.path.join("data", "AE", "train", "encoded", "d_E{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
    #     np.save(os.path.join("data", "AE", "train", "history", "d_E{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)
    # else:
    #     np.save(os.path.join("data", "AE", "test", "encoded" ,"d_E{}L{}-{}.npy".format(e, latent_dim, chan_id)), data)
    #     np.save(os.path.join("data", "AE", "test", "history", "d_E{}L{}-{}.npy".format(e, latent_dim, chan_id)), History_data)

    # arr = data
    # arr = np.insert(arr, 0, 0, axis=1)
    # for i in range(0, arr.shape[0]):
    #     arr[i][0] = traindata[i][0]
    
    # data = []
    # for i in range(len(arr) - l_s - n_predictions):
    #     data.append(arr[i:i + l_s + n_predictions])
    # data = np.array(data)
    # assert len(data.shape) == 3

    # if train:
    #     np.random.shuffle(data)
    #     X_train = data[:, :-n_predictions, :]
    #     y_train = data[:, -n_predictions:, 0]  # telemetry value is at position 0

    #     # np.save(os.path.join("data", "AE", "train", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), X_train)
    #     # np.save(os.path.join("data", "AE", "train", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), y_train)

    #     np.save(os.path.join("data", "AE", "train", "x_train", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), X_train)
    #     np.save(os.path.join("data", "AE", "train", "y_train", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), y_train)

    # else:
    #     X_test = data[:, :-n_predictions, :]
    #     y_test = data[:, -n_predictions:, 0]  # telemetry value is at position 0

    #     # np.save(os.path.join("data", "AE", "test",  "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), X_test)
    #     # np.save(os.path.join("data", "AE", "test", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), y_test)

    #     np.save(os.path.join("data", "AE", "test",  "x_test", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), X_test)
    #     np.save(os.path.join("data", "AE", "test", "y_test", "E{}L{}-{}.npy".format(e, latent_dim, chan_id)), y_test)







































































# for i in range(0, train.shape[0]-1):
#     train[i][0] =  d_train[i][0]

# print(train.shape)
# np.save(os.path.join("data", "train", "D2-P-1.npy"), train)   

# for i in range(0, test.shape[0]-1):
#     test[i][0] =  d_test[i][0]

# print(test.shape)
# np.save(os.path.join("data", "test", "D2-P-1.npy"), test)   























# ############################################################################################################


# L5E10 = np.load(os.path.join("dimensions", "TestL5E10.npy"))
# # # test = np.load(os.path.join("data", "test", "P-1.npy"))

# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = L5E10[:,2]
# y = L5E10[:,3]
# z = L5E10[:,4]

# ax.scatter(x, y, z, c='r', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()



# ax.scatter(x, y, z, c='r', marker='o')

# print(L2E10.shape)


# x1 = list(range(0,L2E10.shape[0]))
# print(L2E10[:,0])

# plt.scatter(L2E10[:,0], L2E10[:,1], L2E10[:,2])





############################################################################################################

# Extracting Telem
# train = np.load(os.path.join("data", "train", "P-1.npy"))
# test = np.load(os.path.join("data", "test", "P-1.npy"))

# j=[]
# for i in range(0, train.shape[0]):
#     j.append(train[i][0])

# x=[]
# for i in range(0, test.shape[0]):
#     x.append(test[i][0])

# np.savetxt('trainv1.txt', j)
# np.savetxt('testv1.txt', x)


# Putting it back in
# train = np.load(os.path.join("data", "train", "P-1.npy"))
# test = np.load(os.path.join("data", "test", "P-1.npy"))

# d_train = pd.read_csv("denoise_train2.csv") 
# d_test = pd.read_csv("denoise_test2.csv") 
# d_train = np.array(d_train)
# d_test = np.array(d_test)

# print(train.shape)
# print(d_train.shape)
# print(test.shape)
# print(d_test.shape)


# for i in range(0, train.shape[0]-1):
#     train[i][0] =  d_train[i][0]

# print(train.shape)
# np.save(os.path.join("data", "train", "D2-P-1.npy"), train)   

# for i in range(0, test.shape[0]-1):
#     test[i][0] =  d_test[i][0]

# print(test.shape)
# np.save(os.path.join("data", "test", "D2-P-1.npy"), test)   









# train1 = np.load(os.path.join("data", "train", "C-1.npy"))

# chan_df = pd.read_csv("labeled_anomalies.csv")

# # chan_ids = [x.split('.')[0] for x in chan_df]
# chan_id = pd.DataFrame({"chan_id": chan_df})

# train = np.insert(train, 0, 0, axis=1)
# for i in range(0, train1.shape[0]):
#     train[i][0] = train1[i][0]

# print(train.shape)



























# n_features = data.shape[2]
# serie_size = data.shape[1]
# latent_dim=5

# encoder_decoder = Sequential()
# encoder_decoder.add(LSTM(serie_size, activation='relu', input_shape=(serie_size, n_features), return_sequences=True))
# encoder_decoder.add(LSTM(int(data.shape[2]), activation='relu', return_sequences=True))
# encoder_decoder.add(LSTM(latent_dim, activation='relu'))
# encoder_decoder.add(RepeatVector(serie_size))
# encoder_decoder.add(LSTM(serie_size, activation='relu', return_sequences=True))
# encoder_decoder.add(LSTM(int(data.shape[2]), activation='relu', return_sequences=True))
# encoder_decoder.add(TimeDistributed(Dense(latent_dim)))
# encoder_decoder.summary()
# encoder_decoder.compile(loss='mse', optimizer='adam')




# input_img = Input(shape=(train[1], train.shape[0]))
# # encoded = LSTM(data.shape[2], activation='relu', return_sequences=False)(input_img)
# # encoded = RepeatVector(data.shape[1])(encoded)
# encoded = LSTM(int(3), activation='relu', return_sequences=False)(input_img)
# encoded = RepeatVector(int(data.shape[1]))(encoded)
# decoded = LSTM(data.shape[2], activation='relu', return_sequences=True)(encoded)
# autoencoder = Model(input_img, decoded)
# encoder = Model(input_img, encoded)
# encoded_input = Input(shape=(int(data.shape[2]/2),))
# autoencoder.summary()
# autoencoder.compile(loss='mse', optimizer='adam')
# # model.compile(optimizer='adam', loss=vae_loss)
# # model.fit(data, y_train, batch_size=m, epochs=n_epoch)
# autoencoder.fit(data,
#                 data,
#                 batch_size=64,
#                 epochs=1,
#                 verbose=True)

# data = encoder.predict(data)

# print(data.shape)

# data = []
# for i in range(len(arr) - l_s - n_predictions):
#     data.append(arr[i:i + l_s + n_predictions])
# data = np.array(data)

# assert len(data.shape) == 3



