package org.mlcommons.peoplesspeech.galvasr2.datasources.tar

import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.hadoop.io.SequenceFile.CompressionType
import org.apache.hadoop.mapreduce.{Job, JobContext, TaskAttemptContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.UnsafeRowWriter
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.util.SerializableConfiguration
import org.slf4j.LoggerFactory

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, 
  TarArchiveInputStream, TarArchiveOutputStream}

class DefaultSource extends FileFormat with DataSourceRegister {
  override val shortName: String = "tar"

  override def inferSchema(
      sparkSession: SparkSession,
      options: Map[String, String],
    files: Seq[FileStatus]): Option[StructType] = {
    Some(StructType(List(StructField("key", StringType), StructField("value", BinaryType))))
  }

  override def prepareWrite(
      sparkSession: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType): OutputWriterFactory = {
    val conf = job.getConfiguration
    val codec = options.getOrElse("codec", "")
    if (!codec.isEmpty) {
      conf.set("mapreduce.output.fileoutputformat.compress", "true")
      conf.set("mapreduce.output.fileoutputformat.compress.type", CompressionType.BLOCK.toString)
      conf.set("mapreduce.output.fileoutputformat.compress.codec", codec)
      conf.set("mapreduce.map.output.compress", "true")
      conf.set("mapreduce.map.output.compress.codec", codec)
    }

    new OutputWriterFactory {
      override def newInstance(
        path: String,
        dataSchema: StructType,
        context: TaskAttemptContext): OutputWriter = {
        new TarOutputWriter(path, context)
      }

      override def getFileExtension(context: TaskAttemptContext): String = {
        ".tar" + CodecStreams.getCompressionExtension(context)
      }
    }
  }

  override def buildReader(
      sparkSession: SparkSession,
      dataSchema: StructType,
      partitionSchema: StructType,
      requiredSchema: StructType,
      filters: Seq[Filter],
      options: Map[String, String],
      hadoopConf: Configuration): (PartitionedFile) => Iterator[InternalRow] = {
    // why broadcast this, but not other things like dataSchema?
    val broadcastedHadoopConf =
      sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))

    System.out.println("GALVEZ")
    System.out.println("data schema", dataSchema)
    System.out.println("required schema", requiredSchema)

    (file: PartitionedFile) => {
      val conf = broadcastedHadoopConf.value.value
      val path = new Path(new URI(file.filePath))
      val fs = path.getFileSystem(conf)
      val status = fs.getFileStatus(path)

      new Iterator[InternalRow] {
        // private val writer = new UnsafeRowWriter(requiredSchema.length)
        private val istream = new TarArchiveInputStream(fs.open(status.getPath))
        private val READ_BUFFER_SIZE = 4096
        // private val readBuffer = new Array[Byte](READ_BUFFER_SIZE)
        // becomes None if this tar file is empty
        private var nextEntry: Option[TarArchiveEntry] = Option(istream.getNextTarEntry())

        override def hasNext: Boolean = {
          nextEntry.nonEmpty
        }

        override def next(): InternalRow = {
          if (!hasNext) {
            throw new java.util.NoSuchElementException("End of stream")
          }
          val writer = new UnsafeRowWriter(requiredSchema.length)
          writer.resetRowWriter()
          // System.out.println("READ", nextEntry.get.getName())
          // System.out.println("READ UTF8", UTF8String.fromString(nextEntry.get.getName()))
          writer.write(0, UTF8String.fromString(nextEntry.get.getName()))

          // does this read only up to the next tar entry?
          val readBuffer = new Array[Byte](nextEntry.get.getSize().toInt)
          val count: Int = istream.read(readBuffer)
          assert(count == nextEntry.get.getSize())
          if (requiredSchema.length == 2) {
            writer.write(1, readBuffer)
          }
          // var count: Int = 0
          // while ( { count = istream.read(readBuffer) ; count != -1 }) {
          //   writer.write(1, readBuffer, 0, count);
          // }

          nextEntry = Option(istream.getNextTarEntry())

          writer.getRow
        }
      }
    }
  }

  override def supportDataType(dataType: DataType): Boolean = dataType match {
    case _: StringType => true
    case _: BinaryType => true
    case _ => false
  }
}

class TarOutputWriter(val pathString: String,
    context: JobContext) extends OutputWriter {

  private val path = new Path(new URI(pathString))
  private val outputStream = new TarArchiveOutputStream(CodecStreams.createOutputStream(context, path))
  outputStream.setLongFileMode(TarArchiveOutputStream.LONGFILE_POSIX);

  override def write(row: InternalRow): Unit = {
    val key = row.getString(0)
    val value = row.getBinary(1)
    // System.out.println("WRITE", key)
    val entry = new TarArchiveEntry(key)
    entry.setSize(value.length)
    outputStream.putArchiveEntry(entry)
    outputStream.write(value)
    outputStream.closeArchiveEntry()
  }

  override def close(): Unit = {
    outputStream.close()
  }
}
