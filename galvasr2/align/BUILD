load("//bazel:cython_rules.bzl", "pyx_library")

pyx_library(name = 'smith_waterman',
            srcs = ['smith_waterman.pyx'], # , 'a.pxd', 'b.py', 'pkg/__init__.py', 'pkg/c.pyx'],
            deps = ["@python//:python-lib"],
)


py_library(
    name = "dsalign_import",
    imports =  ["."],
    srcs = glob(["*.py"]),
    deps = [":smith_waterman"],
    visibility = ["//visibility:public"],
)