import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Reshape, Conv2D, MaxPooling2D
import keras
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from UtilsDeepLOB import zscore_nomalization, zscore_nomalization_mbo, read_data, read_data_mbo, clean_lob, clean_mbo, deprado
from configs import cols_LOB, col_list_LOB, cols_MBO, col_list_MBO

class TensurfDeepLOB():
    def __init__(self, data_path, lookback_cnn=100) -> None:
        self.prepro_obj = PreprocessingLOB(data_path=data_path)
        self.lookback_cnn = lookback_cnn
        self.checkpoint_filepath = "./weights"
    
    def run_deep_lob(self, K, split_sizes=[0.6, 0.2, 0.2]):
        self.K = K
        self.prepro_obj.run_prepare_data(K)
        self.prepared_data(split_sizes)
        self.create_deeplob(T=self.lookback_cnn, NF=40, number_of_lstm=60)
        self.model_training()
        self.model_testing()

    def prepared_data(self, split_sizes):
        N = self.prepro_obj.dataY.shape[0]
        train_size = int(split_sizes[0] * N)
        stop_val_size = train_size + int(split_sizes[1] * N)
        self.trainX = self.prepro_obj.dataX[:train_size, :, :, :]
        self.trainY = self.prepro_obj.dataY[:train_size, :]
        self.valX = self.prepro_obj.dataX[train_size+self.K:stop_val_size, :, :, :]
        self.valY = self.prepro_obj.dataY[train_size+self.K:stop_val_size, :]
        self.testX = self.prepro_obj.dataX[stop_val_size+self.K:, :, :, :]
        self.testY = self.prepro_obj.dataY[stop_val_size+self.K:, :]

    def model_testing(self):
        self.model.load_weights(self.checkpoint_filepath)
        pred = self.model.predict(self.testX)
        print('accuracy_score:', accuracy_score(np.argmax(self.testY, axis=1), np.argmax(pred, axis=1)))
        print(classification_report(np.argmax(self.testY, axis=1), np.argmax(pred, axis=1), digits=4))


    def model_training(self, epochs=200, batch_size=128, verbose=2):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=self.checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)
        self.model.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), 
                    epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[model_checkpoint_callback])
    
    def create_deeplob(self, T, NF, number_of_lstm):
        input_lmd = Input(shape=(T, NF, 1))
        # build the convolutional block
        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (1, 10))(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        # build the inception module
        convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
        convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
        convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
        convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
        convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
        convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
        convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
        convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
        convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
        convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
        conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)
        conv_reshape = keras.layers.Dropout(0.2, noise_shape=(None, 1, int(conv_reshape.shape[2])))(conv_reshape, training=True)
        # build the last LSTM layer
        conv_lstm = LSTM(number_of_lstm)(conv_reshape)
        # build the output layer
        out = Dense(3, activation='softmax')(conv_lstm)
        model = Model(inputs=input_lmd, outputs=out)
        adam = Adam(lr=0.0001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model


class PreprocessingLOB():
    def __init__(self, data_path, lob=True, data=None) -> None:
        self.lob = lob
        if not data:
          if lob:
              self.data_raw = read_data(data_path, cols_LOB, col_list_LOB)
              self.data_clean = clean_lob(self.data_raw)
          else:
              self.data_raw = read_data_mbo(data_path, cols_MBO, col_list_MBO)
              self.data_clean = clean_mbo(self.data_raw)
        else:
          if lob:
              self.data_raw = data
              self.data_clean = clean_lob(self.data_raw)
          else:
              self.data_raw = data
              self.data_clean = clean_mbo(self.data_raw)

    def run_prepare_data(self, K, labeling_type=2, alpha_type=3, show_print=True, alpha=0.00008):
        self.data = self.data_clean.copy()
        ind_labels, est_dsc = self.labeling_deep_lob(K, labeling_type=labeling_type, alpha_type=alpha_type)
        if self.lob:
          zscore_nomalization(self.data)
        else:
          zscore_nomalization_mbo(self.data)
        self.data = self.data.loc[ind_labels]
        self.data.dropna(inplace=True)
        self.dataX, self.dataY = self.prepar_X_Y()
        self.bin_edges_ = None
        if alpha_type == 1:
            self.bin_edges_ = est_dsc.bin_edges_
        elif alpha_type == 2:
            self.bin_edges_ = [-10, -alpha, alpha, 10]
        if show_print:
            self.print_info()  
        
          
    def print_info(self):
        print("---------------------------------------------------------Preprocessed---------------------------------")
        print(self.bin_edges_)
        print(self.data["AlphaLabel"].value_counts())
        print(self.dataX.shape, self.dataY.shape)
        print("-----------------------------------------------------------------------------------------------------")

    def labeling_deep_lob(self, K, column="Midprice", n_bins=3, labeling_type=2, alpha_type=3, alpha=0.00008):
        self.data["MeanNegativeMid"] = self.data[column].rolling(window=K).mean()
        self.data["MeanPositiveMid"] = self.data["MeanNegativeMid"].shift(-(K-1))
        if labeling_type == 1:
            self.data["SmoothingLabel"] = (self.data["MeanPositiveMid"] - self.data[column]) / self.data[column]
        elif labeling_type == 2:
            self.data["SmoothingLabel"] = (self.data["MeanPositiveMid"] - self.data["MeanNegativeMid"]) / self.data["MeanNegativeMid"] 
        labels_np = self.data["SmoothingLabel"].dropna()
        ind_labels = labels_np.index
        est_dsc = None
        if alpha_type == 1: 
            labels_np = labels_np.values
            labels_np = labels_np.reshape(labels_np.shape + (1,))
            est_dsc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            est_dsc.fit(labels_np)
            label_dsc = est_dsc.transform(labels_np)
            self.data["AlphaLabel"] = np.nan
            self.data.loc[ind_labels, "AlphaLabel"] = label_dsc
        elif alpha_type == 2:
            self.data["AlphaLabel"] = np.nan
            self.data.loc[ind_labels, "AlphaLabel"] = 0
            self.data.loc[self.data["SmoothingLabel"] < -alpha, "AlphaLabel"] = -1
            self.data.loc[self.data["SmoothingLabel"] > alpha, "AlphaLabel"] = 1
        elif alpha_type == 3:
            self.data = deprado(self.data)
            self.data = self.data.set_index(self.data.key)
            self.data["AlphaLabel"] = np.nan
            self.data.loc[ind_labels, "AlphaLabel"] = 0
            #self.data.loc[self.data["DePrado"] == -1, "AlphaLabel"] = -1
            #self.data.loc[self.data["DePrado"] == 1, "AlphaLabel"] = 1
            self.data.loc[self.data["DePrado"] < -0.25, "AlphaLabel"] = -1
            self.data.loc[self.data["DePrado"] > 0.25, "AlphaLabel"] = 1
        return ind_labels, est_dsc

    def prepar_X_Y(self, lookback_cnn=100, level_data_num=40):
        data_np = self.data["ConcatLOB"].to_numpy()    
        N = data_np.shape[0]
        dataX = np.zeros((N - lookback_cnn + 1, lookback_cnn, level_data_num))
        for i in range(dataX.shape[0]):
            tmp_sample = np.zeros((lookback_cnn, level_data_num))
            for j in range(lookback_cnn):
                ind = i + j
                tmp_sample[j, :] = data_np[ind]
            dataX[i, :, :] = tmp_sample
        dataY = self.data.loc[self.data.index[lookback_cnn-1:], "AlphaLabel"]
        dataX = dataX.reshape(dataX.shape + (1,))
        dataY = np_utils.to_categorical(dataY, 3)
        return dataX, dataY
