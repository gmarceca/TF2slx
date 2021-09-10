import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint #TensorBoard
from keras.utils import plot_model
from helper_funcs import *
pd.options.mode.chained_assignment = None
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import csv
# from scipy import stats
import datetime
import pickle
import itertools
from argparse import ArgumentParser

def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Load a tranined .h5 model, freeze it and store it as .pb model.')
    parser.add_argument("--validation_from_keras", type=int, default=0,
                        help="Perform validation of the .h5 model")
    parser.add_argument("--validation_from_pb", type=int, default=0,
                        help="Perform validation of the frozen .pb model")
    parser.add_argument("--validation_from_libso", type=int, default=0,
                        help="Perform validation from the .so cpp compiled model library.")
    parser.add_argument("--validation_from_slx", type=int, default=0,
                        help="Perform validation from the simulink predictions.")
    return parser

def repelem(arr, num):
    arr = list(itertools.chain.from_iterable(itertools.repeat(x, num) for x in arr.tolist()))
    return np.asarray(arr)

def checkshot(fshot):
    conv_window_size = 40
    stride = 10
    no_input_channels = 2

    length = int(np.ceil((len(fshot)-conv_window_size)/stride))
    X_scalars_single = np.empty((length, conv_window_size, no_input_channels)) # First LSTM predicted value will correspond to index 20 of the full sequence.
    for j in np.arange(length):
        vals = fshot.iloc[j*stride : conv_window_size + j*stride]
        scalars = np.asarray([vals.FIR, vals.PD]).swapaxes(0, 1)
        assert scalars.shape == (conv_window_size, no_input_channels)
        X_scalars_single[j] = scalars
    return X_scalars_single

def run(args=None):
    
    parser = get_argparser()
    args = parser.parse_args(args)

    shots = [59073, 61714, 61274, 59065, 61010, 61043, 64770, 64774, 64369, 64060, 64662, 64376, 57093, 57095, 61021, 32911, 30268, 45105, 62744, 60097, 58460, 61057, 31807, 33459, 34309, 53601, 42197 ]
    #shots = [59073]
    
    data_dir = './TCV_validation_TestSet'
    
    states_pred_concat =[]
    ground_truth_concat = []
    k_indexes = []
    k_indexes_dic = {}
    consensus_concat = []
    
    conv_w_offset = 20
    conv_window_size = 40
    
    for i, shot in zip(range(len(shots)), shots):
        labeler_states = []
        print('Reading shot', shot)
        fshot = pd.read_csv(data_dir + '/TCV_'  + str(shot) + '_' + 'apau_and_marceca' + '_labeled.csv', encoding='utf-8')
        
        fshot_signals = pd.read_csv('./TCV_signals_TestSet/TCV_'  + str(shot) + '_signals.csv', encoding='utf-8')
    
        stride=10 
        shot_df = fshot.copy()
        shot_df = remove_current_30kA(shot_df)
        shot_df = remove_no_state(shot_df)
        shot_df = remove_disruption_points(shot_df)
        shot_df = shot_df.reset_index(drop=True)
        shot_df = normalize_current_MA(shot_df)
        shot_df = normalize_signals_mean(shot_df)
   
        intersect_times = np.round(shot_df.time.values,5)
        intersec = np.nonzero(np.in1d(fshot_signals.time.values.round(5), intersect_times))[0]

        fshot_equalized = fshot.loc[fshot['time'].round(5).isin(intersect_times)]
        fshot_signals_eq = fshot_signals.loc[fshot_signals['time'].round(5).isin(intersect_times)]
    
        intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
        
        fshot_sliced = fshot.loc[fshot['time'].round(5).isin(intersect_times)]
    
        if args.validation_from_keras:
            pred_states = np.load('./h5_predictions/pred_states_{}.npy'.format(shot))
        elif args.validation_from_pb:
            pred_states = np.load('./pb_predictions/pred_states_{}.npy'.format(shot))
        elif args.validation_from_libso:
            pred_states = np.load('./libso_predictions/libso_pred_states_{}.npy'.format(shot))
        elif args.validation_from_slx:
            pred_states = np.load('./simulink/slx_predictions/slxpred_{}.npy'.format(shot))
        else:
            raise ValueError("At least one type of validation should be specified! either h5, pb, libso or slx.")
        
        if args.validation_from_slx:
            pred_states_slx = np.argmax(pred_states[:], axis=1)
            warm_up = 4 # Predictions start to make sense once the first window of 40 is filled
            pred_states_slx = pred_states_slx[warm_up:]
            # Repeat x'stride' the predictions
            pred_states_slx = repelem(pred_states_slx, stride)
            pred_states_slx = pred_states_slx[intersec]
           
            labeler_states += [fshot_sliced['LHD_label']]
            labeler_states = np.asarray(labeler_states)
            labeler_states = labeler_states[0:1, 0::stride]
            # Compute stride in predictions
            length = int(np.ceil((pred_states_slx.shape[0]-conv_window_size)/stride))
            pred_states_disc = np.empty((length))
            for j in np.arange(length):
                pred_states_disc[j] = pred_states_slx[j*stride]
        else:
            labeler_states += [fshot_sliced['LHD_label'].values[0::stride]]
            labeler_states = np.asarray(labeler_states)
            pred_states_disc = np.argmax(pred_states[:], axis=1)
        
        pred_states_disc += 1
        states_pred_concat.extend(pred_states_disc)
        assert (labeler_states.shape[1] == pred_states_disc.shape[0])
    
        ground_truth = calc_mode(labeler_states.swapaxes(0,1))
        ground_truth_concat.extend(ground_truth)
    
        dice_cf = dice_coefficient(pred_states_disc, ground_truth)
        k_st = k_statistic(pred_states_disc, ground_truth)
        k_indexes += [k_st]
        print('kst', k_st)
        k_indexes_dic[shot] = k_st
    
        consensus = calc_consensus(labeler_states.swapaxes(0,1)) #has -1 in locations which are not consensual, ie at least one person disagrees (case 3)
        consensus_concat.extend(consensus)
    
    k_indexes = np.asarray(k_indexes)
    
    ground_truth_concat = np.asarray(ground_truth_concat)
    consensus_concat = np.asarray(consensus_concat)
    states_pred_concat = np.asarray(states_pred_concat)
    
    ground_truth_mask = np.where(ground_truth_concat!=-1)[0]
    
    ground_truth_concat = ground_truth_concat[ground_truth_mask]
    states_pred_concat = states_pred_concat[ground_truth_mask]
    consensus_concat = consensus_concat[ground_truth_mask] #should stay the same, as consensus is subset of ground truth
    
    print('ground_truth_concat', ground_truth_concat.shape)
    print('states_pred_concat', states_pred_concat.shape)
    
    print('Kappa score pred vs GT: ', k_statistic(states_pred_concat, ground_truth_concat))
    print('Kappa score between different labelers: ', k_statistic(consensus_concat, ground_truth_concat))

if __name__ == "__main__":
    run()
