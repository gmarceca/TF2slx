load('@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl', 'tf_library')
  
tf_library(
    name = 'aot_model',
    config = 'CNN_LSTM.pbtxt',
    cpp_class = 'AotModel',
    graph = 'CNNLSTM_exp09_ep400_12042021.pb',
    include_standard_runtime_deps = False,
    tfcompile_flags = [
        "--xla_cpu_multi_thread_eigen=false",
#        #"--target_cpu=native"
        ]
)

cc_library(
    name = "CNNLSTM_LHD_states",
    srcs = ["CNNLSTM_LHD_states.cc"],
    deps = [
        ":aot_model",
    ] + [
            "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_conv2d",
            "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
           ],
    linkstatic=False
)
cc_binary(
    name = "lib_CNNLSTM_LHD_states_16042021_centos7_wo_googlestr.so",
    # When -c opt is specified at the command-line, optimize this binary for
    # size.  This flag is not "transitive" and does not apply to dependencies --
    # still, it creates a large size reduction (cuts it in half).  This often
    # produces a faster executable as well, but not always and has not yet been
    # tested on the realtime host that will execute this target.
    #copts = select({
    #    "//tensorflow:optimized": ["-Os"],
    #    "//conditions:default": [],
    #}),
    ## Set "fully_static_link" and `linkshared` to get a self-contained .so.
    #features = ["fully_static_link"],
    #linkshared = True,
    deps = [
        ":CNNLSTM_LHD_states",
    ],
    #linkopts = ["-lpthread"],
    linkshared = 1,
    copts = ["-fPIC"]
)
