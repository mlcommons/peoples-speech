py_binary(
    name = "spark_forced_aligner",
    main = "align/spark/align.py",
    imports = ["align"],
    srcs = [
           "align/spark/align.py",
           "align/spark/event_listener.py",
           "align/spark/schemas.py",
           ] +
           glob(["align/*.py"]),
    deps = [
        "//lingvo/tools:audio_lib", # audio_lib.DecodeToWav
        # Implicit pyspark dependency
    ],
    data = [
         # trainer will be called as a subprocess
         "//lingvo:trainer"
    ],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "cuda_decoder_forced_aligner",
    main = "align/spark/align_cuda_decoder.py",
    imports = ["align"],
    srcs = [
           "align/spark/align_cuda_decoder.py",
           "align/spark/align_lib.py",
           "align/spark/dsalign_lib.py",
           "align/spark/event_listener.py",
           "align/spark/schemas.py",
           ] +
           glob(["align/*.py"]),
    deps = [
    ],
    data = [
    ],
    visibility = ["//visibility:public"],
)


py_binary(
    name = "spark_hours_per_language",
    main = "align/spark/hours_per_language.py",
    imports = ["align"],
    srcs = [
           "align/spark/hours_per_language.py",
           "align/spark/align_lib.py",
           "align/spark/event_listener.py",
           "align/spark/schemas.py",
           ] +
           glob(["align/*.py"]),
    deps = [
    ],
    data = [
    ],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "spark_sampling_rate",
    main = "align/spark/sampling_rate.py",
    imports = ["align"],
    srcs = [
           "align/spark/sampling_rate.py",
           "align/spark/align_lib.py",
           "align/spark/schemas.py",
           ] +
           glob(["align/*.py"]),
    deps = [
    ],
    data = [
    ],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "galvasr_tokenize_words",
    srcs = ["galvasr_tokenize_words.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    "//lingvo/core/ops:ops",
    "//lingvo/core:py_utils"
])

# bazel run galvasr2:dump_cc_by_licenses
# bazel build galvasr2:dump_cc_by_licenses
# python dump_cc_by_licenses.py
py_binary(
    name = "dump_cc_by_licenses",
    main = "dump_cc_by_licenses.py",
    srcs = ["dump_cc_by_licenses.py",
            "align/spark/schemas.py",],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)