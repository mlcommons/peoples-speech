package org.mlcommons.peoplesspeech.galvasr2.datasources.tar

import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream
import org.apache.commons.compress.archivers.tar.TarArchiveEntry

class TarOutputWriter(val path: String,
    context: JobContext) extends OutputWriter {

  private val outputStream = new TarArchiveOutputStream(Codecs.createOutputStream(context, path))

  override def write(row: InternalRow): Unit = {
    val key = row.getString(0)
    val value = row.getBinary(1)
    val entry = new TarArchiveEntry(key)
    outputStream.putArchiveEntry(entry)
    outputStream.write(value)
    outputStream.closeArchiveEntry()
  }

  override def close(): Unit = {
    outputStream.close()
  }
}
