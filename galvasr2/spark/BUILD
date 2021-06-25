load("@io_bazel_rules_scala//scala:scala.bzl",
  "scala_library", "scala_macro_library", "scala_binary", "scala_test", "scala_repl")
scala_library(
    name = "tar_spark_datasource",
    srcs = [
         "TarDataSource.scala",
         # "TarOutputWriter.scala"
    ],
    deps = [
         "@apache_spark//:spark_jars",
    ],
    resources = [
          "META-INF/services/org.apache.spark.sql.sources.DataSourceRegister",
    ],
    resource_strip_prefix = "galvasr2/spark",
    visibility = ["//visibility:public"],
    # deps,
    # runtime_deps,
    # exports,
    # data,
    # main_class,
    # resources,
    # resource_strip_prefix,
    # scalacopts,
    # jvm_flags,
    # scalac_jvm_flags,
    # javac_jvm_flags,
    # unused_dependency_checker_mode
)
