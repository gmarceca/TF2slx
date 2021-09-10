import numpy as np
import tensorflow as tf
import time
import pandas as pd
from helper_funcs import *

def model_accuracy(label, prediction):

    # Evaluate the trained model
    return np.sum(label == prediction) / len(prediction)

libmodel = np.ctypeslib.load_library('lib_CNNLSTM_LHD_states_03032021.so', './tensorflow/bazel-bin/external/org_tensorflow/')
libmodel.run.argtypes = [
    np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(40,2), flags=('c', 'a')),
    np.ctypeslib.ndpointer(np.float32, ndim=1, shape=(3), flags=('c', 'a', 'w')),
    np.ctypeslib.ctypes.c_bool]

libmodel.CreateNetwork()

def predict(x, reset):
    x = np.require(x, np.float32, ('c', 'a'))
    y = np.require(np.zeros((3)), np.float32, ('c', 'a', 'w'))
    libmodel.run(x, y, reset)
    return y


test_shots = [59073, 61714, 61274, 59065, 61010, 61043, 64770, 64774, 64369, 64060, 64662, 64376, 57093, 57095, 61021, 32911, 30268, 45105, 62744, 60097, 58460, 61057, 31807, 33459, 34309, 53601, 42197 ]
#test_shots = [59073]

for sh in test_shots:

    # Load Data
    
    if sh == 61043 or sh == 64770 or sh == 64774:
        X = np.load('./inputs/input_signals_{}_ch13.npy'.format(sh))
    elif sh == 64662:
        X = np.load('./inputs/input_signals_{}_ch14.npy'.format(sh))
    else:
        X = np.load('./inputs/input_signals_{}_ch01.npy'.format(sh))

    data_dir = './TCV_validation_TestSet'
    fshot = pd.read_csv(data_dir + '/TCV_'  + str(sh) + '_' + 'apau_and_marceca' + '_labeled.csv', encoding='utf-8')
    fshot_signals = pd.read_csv('./TCV_signals_TestSet/TCV_'  + str(sh) + '_signals.csv', encoding='utf-8')

    shot_df = fshot.copy()
    shot_df = remove_current_30kA(shot_df)
    shot_df = remove_no_state(shot_df)
    shot_df = remove_disruption_points(shot_df)
    intersect_times = np.round(shot_df.time.values,5)
    fshot_equalized = fshot_signals.loc[fshot_signals['time'].round(5).isin(intersect_times)]

    intersec = np.nonzero(np.in1d(fshot_signals.time.values.round(5), intersect_times))[0]

    # In the simulink implementation predictions are computed starting from X[1:] instead of X[0:].
    # This is because of how the tool is implemented and it should be solved in the next version.
    # As a result, to be consistent with the simulink evaluation, we start our X array from 2nd index.
    X = X[1:,:]
    # One hack to avoid this issue is to decrease the index of the intersec array by 1. FIXME
    intersec = intersec -1
    X = X[intersec]

    # Reshape input in sliding windows with stride
    stride=10
    conv_window_size=40
    no_input_channels = X.shape[1]
    length = int(np.ceil((X.shape[0]-conv_window_size)/stride))
    X_scalars_single = np.empty((length, conv_window_size, no_input_channels))
    for j in np.arange(length):
        vals_FIR = X[j*stride : conv_window_size + j*stride,0]
        vals_PD = X[j*stride : conv_window_size + j*stride,1]
        scalars = np.asarray([vals_FIR, vals_PD]).swapaxes(0, 1)
        assert scalars.shape == (conv_window_size, no_input_channels)
        X_scalars_single[j] = scalars
    # Apply normalization
    X_scalars_single[:,:,0] = X_scalars_single[:,:,0]*1e-19

    allinputs = []
    pred_states = []
    start = time.time()
    reset = True
    X_scalars_single = np.nan_to_num(X_scalars_single)
    for i in range(X_scalars_single.shape[0]):
        if i >0:
            reset = False
        sample = X_scalars_single[i,:,:]
        allinputs.append(sample)
        y = predict(sample, reset)
        pred_states.append(y)
    end = time.time()
    print('elapsed time: ', end - start)
    np.save('./libso_predictions/libso_pred_states_{}.npy'.format(sh), np.asarray(pred_states))

