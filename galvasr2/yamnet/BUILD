py_binary(
    name = "single_process_yamnet",
    main = "inference.py",
    srcs = [
           "inference.py",
           "features.py",
           "params.py",
           "yamnet.py"
           ],
    deps = [
    ],
    data = [
           "yamnet_class_map.csv",
           ":config/default.yaml"
    ],
    visibility = ["//visibility:public"],
)
