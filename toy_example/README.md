## Export a tensorflow model to C++

We compile the model using AOT (ahead-of-time) tf compiler\
https://www.tensorflow.org/xla/tfcompile

For this, we build tensorflow from source and so we use the bazel package\
https://docs.bazel.build/versions/master/install-ubuntu.html

Make sure you are in <em>toy_example/</em>

#### Installation of Bazel

For this, you need to have sudo permissions in the machine.

1. The binary installers can be downloaded from Bazel: [GitHub releases page](https://github.com/bazelbuild/bazel/releases)
2. Download a binary installer version compatible with the tf version you will use and copy it in your `my_working_dir/`
3. If you are using linux, do:\
`chmod +x bazel-<version>-installer-linux-x86_64.sh`\
`sudo ./bazel-<version>-installer-linux-x86_64.sh`\
`export PATH="$PATH:$HOME/bin"`\
4. Check if bazel was installed:\
`bazel version`
5. For the working example demonstration we will use TF 2.4 and bazel 3.7.1

#### Setup AOT compiler

Once bazel is installed, clone the tf repo:\
`git clone https://github.com/tensorflow/tensorflow.git`\
`cd tensorflow`\
`git checkout r2.4`\
`./configure` \
you will be prompted with some questions, for this example you can just click enter which answers everything as "No" by default.
This assumes you have already set your python library path in your "~/.bashrc".

Before using AOT compiler, we need to have a model we would like to compile. The TF graph contains mutable nodes that have to be constants,
we do this by "freezing the graph".

#### Build and freeze a tensorflow graph

For this example, we build a very simple TF graph which will perform a simple add operation of two variables.

To build this graph you can run the script <em>generate_tf_toy_model.py</em>. You just need tensorflow and numpy
libraries setup in e.g a python enviroment.\
`cd ../tensorflow/`\
`mkdir frozen_models`\
`python generate_tf_toy_model.py`

This will generate a frozen model in a protobuf <em>.pb</em> format <em>frozen_models/toy_model.pb</em>

#### Compile the tensorflow graph

One we have built a frozen graph we can go ahead with AOT compilation:\
`cp frozen_models/toy_model.pb tensorflow/`

We create a file <em>toy_model.pbtxt</em> and specify the "feeds" and "fetchs" input and output nodes respectively.
An example is already in the repo.

We create then a BUILD file as shown in the provided example.\
The <em>tf_library</em> specifies the frozen model path, the name of the cpp class model "AotModel", the ".pbtxt" config and some flags. This module will automatically generate a header file <em>bazel-bin/aot_model.h</em>, which is created from <em>tensorflow/compiler/aot/codegen.cc</em>.

Before compiling the aot model, let's confirm that all the operations of our TF graph are defined in tfcompiler\
`bazel run -c opt -- tensorflow/compiler/tf2xla:tf2xla_supported_ops --device=XLA_CPU_JIT`\
We can see that indeed the <em>Add</em> operation is listed, and so it's compatible with the XLA requirements.

Now we compile the AotModel graph:\
`bazel build --show_progress_rate_limit=600 @org_tensorflow//:aot_model`\
Note that this will take a while as quite a significant part of Tensorflow may need to be compiled, depending on your model.

Next, we define the AotModel main functions (CreateNetwork, DeleteNetwork and run) in extern "C" as shown in <em>my_policy_export_toy.cc</em>.\
The run() function receives the two input terms "x" and "y" as integer pointers and copy the result via <em>std::copy_n</em> in a variable "z".\
In BUILD we set the AotModel cc_library name as <em>my_policy_export</em> (which depends on aot_model.h) and name the desired shared library as <em>libpolicy_export_toy.so</em> in the <em>cc_binary</em> module. Finally we generate the compiled model: <em>bazel-bin/libpolicy_export_toy.so</em>.\
`bazel build --show_progress_rate_limit=60 @org_tensorflow//:libpolicy_export_toy.so`

Test predictions from the generated <em>bazel-bin/libpolicy_export_toy.so</em>:\
`cd ../tensorflow`\
`python test_cpp_model.py`

## Export from C++ to Simulink

In this section we show how to export your compiled .so cpp library to simulink via <em>legacy_code</em>.\
For this you need your .so library and the corresponded header file, which can be created manually by knowing
the data types of your defined CreateNetwork, DeleteNetwork and run functions.\
`cd simulink/`

The main file we wil use to compile the .so model is provided in <em>policy_compile_toy.m</em>. This script calls
the .h and .so files, the first one is provided in <em>inc/my_policy_export_toy.h</em>.\
`mv ../libpolicy_export_toy.so lib/`

The main functions in the .m file are:\
<em>def.StartFcnSpec</em> which calls <em>CreateNetwork()</em>\
<em>def.OutputFcnSpec</em> which calls the main function of the model <em>run(int* a, int* b, int* c)</em> and\
<em>def.TerminateFcnSpec</em> which calls <em>DeleteNetwork()</em>\
The naming convention for the run() function is:\
<em>u1, u2, ..., uN</em> for inputs\
<em>y1, y2, ..., yN</em> for outputs\
Be aware that the ordering in which the array is depicted is not the same in cpp and in matlab. In this case it's not important since they are just 1D array, but in case you have a 2D array of e.g size_u1 = [12][34], in simulink you should reverse the order: run(double u1[34][12], ...).

Let's now compile the model. We tested this in the scd machine which counts with simulink and matlab960.\
`matlab960`\
`policy_compile_toy()`

After compiling, a window will open containing the model in a simulink block. To execute the mode you should provide it with the inputs and output blocks. An example was provided in <em>test_simulink_model.slx</em>. By opening it you will see the model with the expected inputs (in this case two constant blocks) and an output <em>to workspace</em> block. You can run the model and check that the addition of the two numbers is correct.
