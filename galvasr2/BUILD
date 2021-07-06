py_binary(
    name = "galvasr_tokenize_words",
    srcs = ["galvasr_tokenize_words.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    "//lingvo/core/ops:ops",
    "//lingvo/core:py_utils"
])

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "ner_pipeline_transcript",
    main = "ner_pipeline_transcript.py",
    srcs = ["ner_pipeline_transcript.py",
            "align/spark/align_lib.py",
            "align/spark/schemas.py",],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_binary(
    name = "zero_shot_classification_pipeline",
    main = "zero_shot_classification_pipeline.py",
    srcs = ["zero_shot_classification_pipeline.py",
            "align/spark/align_lib.py",
            "align/spark/schemas.py",],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)