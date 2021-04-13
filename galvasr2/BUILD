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
    name = "spark_hours_per_language",
    main = "align/spark/hours_per_language.py",
    imports = ["align"],
    srcs = [
           "align/spark/align.py",
           "align/spark/hours_per_language.py",
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
    name = "galvasr_tokenize_words",
    srcs = ["galvasr_tokenize_words.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    "//lingvo/core/ops:ops",
    "//lingvo/core:py_utils"
])