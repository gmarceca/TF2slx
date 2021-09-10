## CNN-LSTM example

Here we describe the steps to compile a trained CNN-LSTM model from keras to simulink and how to perform the validation at each step. Also we remark some subtle changes we had to do in order get the expected results.\
Make sure you are in <em>CNN-LSTM_example/</em>

To run the following scripts we assume you have miniconda installed. We provide a "pkgs.txt" python enviroment with TF 2.2 you can create and activate.\
`conda create --name NEWENV --file pkgs.txt`\
`conda activate NEWENV`

The script <em>CNN_LSTM_h52pb_conversion.py</em> provided loads a ".h5" trained model <em>keras_models/CNNLSTM_new_dataset_pruned_exp13_ep500.h5</em> and creates a frozen ".pb" model required for the cpp compilation. You have also the option to compute the model predictions in the test set provided needed for the validation. For this you should run\
`python CNN_LSTM_h52pb_conversion.py --validation_from_keras=1` (to compute the predictions from the ".h5" model) or\
`python CNN_LSTM_h52pb_conversion.py --validation_from_pb=1` (to compute the predictions from the ".pb" frozen model)\
If you just want to generate the frozen model just run\
`python CNN_LSTM_h52pb_conversion.py`

To compute the final kappa scores and validate the predictions either from the .h5 or .pb models:\
`python CNN_LSTM_validation.py --validation_from_keras=1` or \
`python CNN_LSTM_validation.py --validation_from_pb=1`

### Compiling the frozen .pb CNN-LSTM model to C++

In contrast with the toy_example described above, here we have a much complex architecture and we should be sure all operations are compatible with the XLA version we are using. It results that unfortunately the 'Conv2D' operation is not registered as a compatible OpKernel in the tensorflow version r2.4. But it's registered in version TF r2.0. For this reason, and to also inherit some good features from TF r2.4, we opted to use r2.0 but with a patch of the new release. This patch can be obtained from the following git commit:\
https://github.com/tensorflow/tensorflow/commit/96f4a930dbdd1b5b3c73d262851bb0b867ea0117

The steps are:\
`git clone https://github.com/tensorflow/tensorflow.git`\
`cd tensorflow`\
`git checkout r2.0`\
Modify the relevant functions from the aforementioned commit.

This way you will run r2.0 with all the compatible TF operations and also inheriting the new features from r2.4.\
To use r2.0 you need a lower version of bazel, in this example we used bazel 0.26.1.\
[bazel releases here](https://github.com/bazelbuild/bazel/releases)\
`chmod +x bazel-0.26.1-installer-linux-x86_64.sh`\
`sudo ./bazel-0.26.1-installer-linux-x86_64.sh` or `./bazel-0.26.1-installer-linux-x86_64.sh --user`\
`bazel version`\
`cd tensorflow`\
`./configure` and just click enter to all the questions.\
The frozen model should be within the tensorflow directory where you will launch the bazel commands.\
`mv ../frozen_models/CNNLSTM_new_dataset_pruned_exp13_wo_norm_ep500.pb .`\
Build the TF graph:\
`cp ../BUILD .`\
`cp ../CNN_LSTM.pbtxt .`\
`cp ../CNNLSTM_LHD_states.cc .`\
`bazel build --show_progress_rate_limit=600 @org_tensorflow//:aot_model`\
For the cpp library compilation we added two dependencies by hand:\
<em>//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_conv2d</em> and \
<em>//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_conv2d</em>\
Now we generate the .so compiled model <em>lib_CNNLSTM_LHD_states_03032021.so</em>\
Uncomment lines from <em>cc_library</em> onwards on the BUILD file and run:\
`bazel build --show_progress_rate_limit=60 @org_tensorflow//:lib_CNNLSTM_LHD_states_03032021.so`

An important feature of the CNN-LSTM implementation is the possibility to update the hidden states of the LSTM internally
in the AotModel. This can be seen in <em>CNNLSTM_LHD_states.cc</em>, the boolean <em>reset</em> when set <em>true</em>
set the LSTM states to zero. Furthermore, one in principle could add some pre-processing routines here as well (or directly in the TF graph).However, in this application we performed the data pre-processing in simulink.

### Validation from C++ generated library

The generated .so library can be found here:\
<em>bazel-bin/external/org_tensorflow/lib_CNNLSTM_LHD_states_03032021.so</em>
 We will call it from a python routine to validate the predictions:\
`cp bazel-bin/external/org_tensorflow/lib_CNNLSTM_LHD_states_03032021.so ..` \
`cd ../`\
Generate predictions from the .so library\
`python predict_plasma_states_cnnlstm_libso_test.py`\
Validate the kappa scores:\
`python CNN_LSTM_validation.py --validation_from_libso=1`

### Build the Simulink model

We follow the same instructions as described in the toy model above.\
`cd simulink`\
The compiled library is already provided here:\
`lib/lib_CNNLSTM_LHD_states_03032021.so`\
We also provide the needed header file <em>inc/CNNLSTM_LHD_states.h</em>\
Connect to scd and open matlab960\
`compile_CNNLSTM_LHD_states.m()`\
This will generate the simulink block model. To test an end-to-end example, we provided with the final simulink implementation (ready to run in real-time) including the pre-processing steps. This can be found as <em>CNNLSTM_LHD_states_v1.slx</em>.\
To be able to run this model the matlab <em>buffer_func_10khz.m</em> and <em>buffer_func_1khz.m</em> modules are provided.

Let's now run the model and generate the simulink predictions. For this we need as input a timeseries called "input_name". This is a timeseries 2D vector in this case consisting of two cahnnels, the FIR and PD signals needed by the CNN-LSTM model. All the simulink blocks that you can see apart from the main model, are intended to do the same <em>sliding window</em> computations as present in <em>predict_plasma_states_cnnlstm_libso_test.py</em> or <em>CNN_LSTM_h52pb_conversion.py</em>.

When you run the model, the simulink predictions will be stored in a ".npy" array that will be use later to validate the kappa scores. For this you need to download the [npy-matlab](https://github.com/kwikteam/npy-matlab) package.\
`git clone https://github.com/kwikteam/npy-matlab`\
`addpath('./npy-matlab/npy-matlab')`\
`savepath`

Finally, to compute the simulink predictions:\
`run_slx_model`\
The predictions will be stored in "./slx_predictions"

### Validation from Simulink predictions

As a final step, let's compute the kappa scores from the simulink predictions:\
`cd ../`\
`python CNN_LSTM_validation.py --validation_from_slx=1`\
You will notice that the final kappa scores are:\
Kappa score pred vs GT = **[0.9318846  0.84820858 0.94044878 0.92818138]**\
whereas for the previous validation (from h5, pb and libso) you have got:\
Kappa score pred vs GT = **[0.93472339 0.86199973 0.9431412  0.93185477]**

It is expected to be some differences. There are two reason for this:
1. Simulink predictions are performed from inputs X[1] instead of X[0], and so the associated labelling does not match exact.\
if we managed to account for this, the results would be:\
Kappa score pred vs GT = **[0.93335652 0.85046063 0.94167305 0.92962759]**
2. In simulink the hidden LSTM states are reset once the first window of 40 time steps is filled. On the other hand, in the "offline" model the states are reset after <em>remove_current_30kA</em> and <em>remove_no_state</em> functions are applied, in other words, after IP>20kA cut.
