py_binary(
    name = "spark_forced_aligner",
    main = "align.py",
    srcs = [
           "align.py",
           "event_listener.py",
           "schemas.py",
           ],
    deps = [
        "//lingvo/tools:audio_lib", # audio_lib.DecodeToWav
        # Implicit pyspark dependency
         "//galvasr2/align:dsalign_import",
    ],
    data = [
         # trainer will be called as a subprocess
         "//lingvo:trainer"
    ],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "cuda_decoder_forced_aligner",
    main = "align_cuda_decoder.py",
    srcs = [
           "align_cuda_decoder.py",
           "align_lib.py",
           "dsalign_lib.py",
           "event_listener.py",
           "schemas.py",
           "timeout.py",
           ],
    deps = [
         "//galvasr2/align:dsalign_import",
    ],
    data = [
    ],
    visibility = ["//visibility:public"],
)


py_binary(
    name = "spark_hours_per_language",
    main = "hours_per_language.py",
    srcs = [
           "hours_per_language.py",
           "align_lib.py",
           "event_listener.py",
           "schemas.py",
           "timeout.py",
           ],
    deps = [
         "//galvasr2/align:dsalign_import",
    ],
    data = [
    ],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "spark_hours_per_license",
    main = "hours_per_license.py",
    srcs = [
           "hours_per_license.py",
           "align_lib.py",
           "event_listener.py",
           "schemas.py",
           ],
    deps = [
         "//galvasr2/align:dsalign_import",
    ],
    data = [
    ],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "spark_sampling_rate",
    main = "sampling_rate.py",
    srcs = [
           "sampling_rate.py",
           "align_lib.py",
           "schemas.py",
           ],
    deps = [
         "//galvasr2/align:dsalign_import",
    ],
    data = [
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "align_lib_test",
    main = "align_lib_test.py",
    imports = ["align"],
    srcs = [
           "align_lib_test.py",
           "align_lib.py",
           "align.py",
           "event_listener.py",
           "schemas.py",
           "timeout.py",
           ], # +
           # glob(["../align/*.py"]),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "download_missing_files",
    main = "download_missing_files.py",
    srcs = ["download_missing_files.py",],
    visibility = ["//visibility:public"],
)

py_library(
    name = "juan_lib",
    srcs = [
         "juan_lib.py",
         "align_lib.py",
         "schemas.py",
         "timeout.py",
    ]
)

py_test(
    name = "dsalign_lib_test",
    main = "dsalign_lib_test.py",
    # imports = ["align"],
    srcs = [
           "align_lib.py",
           "dsalign_lib_test.py",
           "dsalign_lib.py",
           "event_listener.py",
           "schemas.py",
           "timeout.py",
           ],
    deps = [
         "//galvasr2:utils",
         "//galvasr2/align:dsalign_import",
    ],
    data = [
         "alphabet2.txt",
         "test_data/BOS2015April7/BOS2015April7.asr.srt",
         "test_data/BOS2015April7/BOS2015April7-BOS_2015April7.mp3.ctm",
    ],
    visibility = ["//visibility:public"],
)

# bazel run galvasr2:dump_cc_by_licenses
# bazel build galvasr2:dump_cc_by_licenses
# python dump_cc_by_licenses.py
py_binary(
    name = "dump_cc_by_licenses",
    main = "dump_cc_by_licenses.py",
    srcs = ["dump_cc_by_licenses.py",
            "schemas.py",],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)