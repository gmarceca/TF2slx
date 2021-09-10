import sys
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils import wrap_frozen_graph
from architectures.def_model_pruned_v7_wo_norm import model_arc
import time
import pandas as pd
from helper_funcs import *
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
    return parser

def validation(model, flag):
    
    # Perform validation of the trained keras model in the official test set shots
    test_shots = [59073, 61714, 61274, 59065, 61010, 61043, 64770, 64774, 64369, 64060, 64662, 64376, 57093, 57095, 61021, 32911, 30268, 45105, 62744, 60097, 58460, 61057, 31807, 33459, 34309, 53601, 42197 ]
    #test_shots = [59073]

    for sh in test_shots:
        
        print('Running shot {}'.format(sh))

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

        #np.save('./inputs_normalized_from_matlab/{}_processed_shot_with_normalization.npy'.format(sh), X_scalars_single[:, :, :])
        h1 = np.zeros([1, 32])
        c1 = np.zeros([1, 32])

        frozen_pred_states = []
        start = time.time()
        for i in range(X_scalars_single.shape[0]):
            sample = X_scalars_single[i:i+1,:,:]
            if flag == 'h5':
                sample = tf.expand_dims(sample, axis=0)
                states, elms, h1, c1 = model([sample, h1, c1])
            elif flag == 'pb':
                states, elms, h1, c1 = model(x=tf.cast(tf.constant(sample), dtype='float32'), x_1=tf.cast(tf.constant(h1), dtype='float32'), x_2=tf.cast(tf.constant(c1), dtype='float32'))
            frozen_pred_states.append(states)
        end = time.time()
        print('elapsed time: ', end - start)
        output_arr = np.squeeze(np.asarray(frozen_pred_states))
        np.save('./{}_predictions/pred_states_{}.npy'.format(flag, sh), output_arr)

def run(args=None):
    
    parser = get_argparser()
    args = parser.parse_args(args)

    model = model_arc()
    model.summary()
    model.compile()
    model.load_weights("keras_models/CNNLSTM_new_dataset_pruned_exp13_ep500.h5")
    #model.reset_states()
   
    if args.validation_from_keras:
        validation(model, 'h5')
    else:
        # Freeze the .h5 graph and store it as .pb file needed by AOT compiler.
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
        x=[tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype), tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype), tf.TensorSpec(model.inputs[2].shape, model.inputs[2].dtype)])
        frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
        frozen_func.graph.as_graph_def()
        print("-" * 50)
        #print("Frozen model layers: ")
        #for layer in layers_ph:
        #    print(layer)
        #print('Number of nodes: ', len(layers_ph))
        
        #print("-" * 50)
        #print("Frozen model inputs: ")
        #print(frozen_func.inputs)
        #print("Frozen model outputs: ")
        #print(frozen_func.outputs)
        #Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir="./frozen_models",
                          name="CNNLSTM_new_dataset_pruned_exp13_wo_norm_ep500.pb",
                          as_text=False)
        
        if args.validation_from_pb:
            # Load frozen graph using TensorFlow 1.x functions
            with tf.io.gfile.GFile("./frozen_models/CNNLSTM_new_dataset_pruned_exp13_wo_norm_ep500.pb", "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())
            # Wrap frozen graph to ConcreteFunctions
            frozen_func = wrap_frozen_graph(graph_def=graph_def,
                    #inputs=["x:0"],
                    inputs=["x:0", "x_1:0", "x_2:0"],
                                            outputs=["Identity:0", "Identity_1:0", "Identity_2:0", "Identity_3:0"],
                                            #outputs=["Identity:0"],
                                            print_graph=True)
            print("-" * 50)
            print("Frozen model inputs: ")
            print(frozen_func.inputs)
            print("Frozen model outputs: ")
            print(frozen_func.outputs)

            validation(frozen_func, 'pb')
            
if __name__ == "__main__":
    run()
