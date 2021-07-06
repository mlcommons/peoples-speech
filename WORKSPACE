"""Workspace file for lingvo."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load(
    "//lingvo:repo.bzl",
    "cc_tf_configure",
    "icu",
    "lingvo_protoc_deps",
    "lingvo_testonly_deps",
)

git_repository(
    name = "subpar",
    remote = "https://github.com/google/subpar",
    tag = "2.0.0",
)

git_repository(
    name = "cython",
    remote = "https://github.com/cython/cython",
    tag = "3.0a7",
)

# This is not robust. However, I don't know of a way to configure
# bazel to find the correct python. It appears that tensorflow and
# grpc have ways to do it, though.

new_local_repository(
    name = "python",
    path = "/install/miniconda3/envs/100k-hours-lingvo-3",
    build_file_content = """
cc_library(
    name = "python-lib",
    hdrs = glob(["include/python3.7m/*.h"]) +
           glob(["lib/python3.7/site-packages/numpy/core/include/numpy/*.h",
                 "lib/python3.7/site-packages/numpy/core/include/numpy/random/*.h"]),
    includes = ["include/python3.7m",
                "lib/python3.7/site-packages/numpy/core/include"],
    visibility = ["//visibility:public"]
)
    """
)

cc_tf_configure()

lingvo_testonly_deps()

lingvo_protoc_deps()

icu()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

skylib_version = "1.0.3"
http_archive(
  name = "bazel_skylib",
  sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
  type = "tar.gz",
  url = "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel-skylib-{}.tar.gz".format(skylib_version, skylib_version),
)

rules_scala_version = "5df8033f752be64fbe2cedfd1bdbad56e2033b15"

git_repository(
    name = "io_bazel_rules_scala",
    remote = "https://github.com/bazelbuild/rules_scala/",
    commit = "2b7edf77c153f3fbb882005e0f199f95bd322880"
)

# http_archive(
#   name = "io_bazel_rules_scala",
#   sha256 = "b7fa29db72408a972e6b6685d1bc17465b3108b620cb56d9b1700cf6f70f624a",
#   strip_prefix = "rules_scala-%s" % rules_scala_version,
#   type = "zip",
#   url = "https://github.com/bazelbuild/rules_scala/archive/%s.zip" % rules_scala_version,
# )

# Stores Scala version and other configuration
# 2.12 is a default version, other versions can be use by passing them explicitly:
# scala_config(scala_version = "2.11.12")
load("@io_bazel_rules_scala//:scala_config.bzl", "scala_config")
scala_config()

load("@io_bazel_rules_scala//scala:scala.bzl", "scala_repositories")
scala_repositories()

load("@io_bazel_rules_scala//scala:toolchains.bzl", "scala_register_toolchains")
scala_register_toolchains()

# optional: setup ScalaTest toolchain and dependencies
load("@io_bazel_rules_scala//testing:scalatest.bzl", "scalatest_repositories", "scalatest_toolchain")
scalatest_repositories()
scalatest_toolchain()

load("@io_bazel_rules_scala//scala:scala.bzl",
  "scala_library", "scala_macro_library", "scala_binary", "scala_test", "scala_repl")

new_local_repository(
    name = "apache_spark",
    path = "/install/spark/",
    build_file = "third_party/spark.BUILD",
)
