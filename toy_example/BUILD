load('@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl', 'tf_library')
  
tf_library(
    name = 'aot_model',
    config = 'toy_model.pbtxt',
    cpp_class = 'AotModel',
    graph = 'toy_model.pb',
    include_standard_runtime_deps = False,
    tfcompile_flags = [
        "--xla_cpu_multi_thread_eigen=false",
#        #"--target_cpu=native"
        ]
)


cc_library(
    name = "my_policy_export",
    srcs = ["my_policy_export_toy.cc"],
    deps = [
        ":aot_model",
    ],
    linkstatic=False
)
cc_binary(
    name = "libpolicy_export_toy.so",
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
        ":my_policy_export",
    ],
    #linkopts = ["-lpthread"],
    linkshared = 1,
    copts = ["-fPIC"]
)
