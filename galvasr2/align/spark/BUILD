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
