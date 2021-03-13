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
