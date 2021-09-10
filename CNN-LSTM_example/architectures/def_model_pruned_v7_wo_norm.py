import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model


def model_arc():
    # Inputs
    hidden1 = Input(shape=(32,))
    cell1 = Input(shape=(32,))
    f_hidden1 = Flatten()(hidden1)
    f_cell1 = Flatten()(cell1)
    
    cnn_input = Input(shape=(40, 2))
    seq_input = Input(shape=(1, 40, 2))
    # CNN
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(cnn_input)
    x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv)
    x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    
    conv_out = Flatten()(x_conv)
    conv_out = Dense(16, activation='relu')(conv_out)
    modelCNN = Model(inputs=[cnn_input], outputs=[conv_out])
    
    print(modelCNN.summary())
    
    # CNN --> LSTM
    modelJoined = TimeDistributed(modelCNN)(seq_input)
    modelJoined, h_out1, c_out1 = LSTM(32, return_sequences=True, return_state=True)(modelJoined, initial_state=[f_hidden1, f_cell1])
    # Outputs
    states_out1 = [h_out1, c_out1]
    modelJoined = Dense(8, activation='relu')(modelJoined)
    modelJoined = Dropout(.5)(modelJoined)
    modelJoinedStates = Dense(3, activation='softmax')(modelJoined)
    modelJoinedElms = Dense(2, activation='softmax')(modelJoined)
    # CNN-LSTM Model
    model = Model([seq_input, hidden1, cell1],[modelJoinedStates, modelJoinedElms] + states_out1)
    return model
