# TF2SLX

This readme documents how to export a trained tf/keras model to simulink.
The first step is to export the model to a cpp library. Next, we will embed
this library in simulink. 

Two working examples are provided, a very basic one <em>toy_example/</em> and a more complicated <em>CNN-LSTM_example/</em> where we show how to export an already trained Convolutional-LSTM neural network. 

A detailed README is provided in both folders <em>toy_example/</em> and <em>CNN-LSTM_example/</em>.

The examples were successfully tested in the spcpc395  and scd machines.

# Compiling tensorflow in Centos 7 with Docker

## Install docker
- `sudo apt-get update`
- `sudo apt-get install docker-ce docker-ce-cli containerd.io`
- `sudo docker run hello-world`

## Create a docker image (Centos7 image provided by CERN)
- `sudo docker pull gitlab-registry.cern.ch/linuxsupport/cc7-base:latest`
- `sudo docker images`
- `sudo docker run -it IMAGE_ID bash`

Open another terminal (terminal 2) and check that you can see the image:
- `sudo docker ps`

Let's install something in the image:
- `yum install wget`

Commit the installation:

From terminal 2 run:
- `sudo docker commit NAME_ID centos7_test`
where NAME_ID is the Name from your image process `sudo docker ps`. Now you will see a new image whose REPOSITORY is `centos7_test` and an auto-generated IMAGE_ID. If you exit now from your docker image and enter to centos7_test, you will still see `wget` there.

## Install git version 2.27 from source (needed to build tensorflow)
CAVEAT: the latest (default) git version for Centos7 is 1.8, you need to install 2.27 to build tensorflow, otherwise you will get the following error: `Unknown option: -C (git)`. For this, follow the instructions here to install git version 2.27.0 from source.
https://computingforgeeks.com/how-to-install-latest-version-of-git-git-2-x-on-centos-7/

## Further installations
install python: https://tecadmin.net/install-python-2-7-on-centos-rhel/
- `curl -O https://bootstrap.pypa.io/pip/2.7/get-pip.py`
- `python2.7 get-pip.py`
- `pip install future`
- `yum install gcc-c++`

## Compile tensorflow:
- `git clone https://github.com/tensorflow/tensorflow.git`
- `cd tensorflow`
- `git checkout r2.0`

### Install Bazel
From terminal 2, copy the following file to your docker image:
`sudo docker cp bazel-0.26.1-installer-linux-x86_64.sh CONTAINER_ID:/root/`. Where CONTAINER_ID is obtained from `sudo docker ps`. Now you will see the file in your docker image
- `chmod +x bazel-0.26.1-installer-linux-x86_64.sh`
- `./bazel-0.26.1-installer-linux-x86_64.sh`
- `bazel --version` should be 0.26.1

### Build tensorflow
From terminal 2 copy the following files:
- `sudo docker cp ./build_docker_inputs/CNNLSTM_exp09_ep400_12042021.pb CONTAINER_ID:/root/tensorflow/`
- `sudo docker cp ./build_docker_inputs/CNN_LSTM.pbtxt CONTAINER_ID:/root/tensorflow/`
- `sudo docker cp ./build_docker_inputs/CNNLSTM_LHD_states.cc CONTAINER_ID:/root/tensorflow/`
- `sudo docker cp ./build_docker_inputs/codegen.cc CONTAINER_ID:/root/tensorflow/tensorflow/compiler/aot/`
- `sudo docker cp ./build_docker_inputs/codegen_test_h.golden CONTAINER_ID:/root/tensorflow/tensorflow/compiler/aot/`
- `sudo docker cp ./build_docker_inputs/BUILD CONTAINER_ID:/root/tensorflow/`

From the docker image:
- `cd tensorflow`
Replace in the `configure` file: `which python` --> `which python2.7`
- `./configure` (press all enter)
- `bazel build --show_progress_rate_limit=600 @org_tensorflow//:aot_model`
- `bazel build --show_progress_rate_limit=60 @org_tensorflow//:lib_CNNLSTM_LHD_states_16042021_centos7_wo_googlestr.so`. 
Done!. Now you can copy the compiled `.so` model to your local dir:

From terminal 2:
- `sudo docker cp CONTAINER_ID:/root/tensorflow/bazel-bin/external/org_tensorflow/lib_CNNLSTM_LHD_states_16042021_centos7_wo_googlestr.so .`. Remember to commit your image before leaving if you want to keep the installation!
