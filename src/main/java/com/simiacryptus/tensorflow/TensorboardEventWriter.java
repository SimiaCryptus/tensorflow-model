/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.tensorflow;

import com.google.common.hash.Hashing;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.RunMetadata;
import org.tensorflow.framework.Summary;
import org.tensorflow.util.Event;
import org.tensorflow.util.LogMessage;
import org.tensorflow.util.SessionLog;
import org.tensorflow.util.TaggedRunMetadata;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.*;
import java.net.InetAddress;

public class TensorboardEventWriter implements AutoCloseable {
  private static final long kMaskDelta = 0xa282ead8L;
  private static final long intMask = 0xFFFFFFFFL;
  @Nullable
  private static volatile String hostName = null;
  @Nonnull
  public final File location;
  @Nullable
  private volatile FileOutputStream fileOutputStream = null;
  private long step = 0;

  public TensorboardEventWriter(@Nonnull File location, @Nonnull GraphDef graphDef) throws IOException {
    this.location = location;
    location.getAbsoluteFile().getParentFile().mkdirs();
    write(graphDef);
  }

  @Nullable
  public static String getHostName() {
    if (null == hostName) {
      synchronized (TensorboardEventWriter.class) {
        if (null == hostName) {
          try {
            hostName = InetAddress.getLocalHost().getHostName();
            if (null == hostName) hostName = InetAddress.getLocalHost().getHostAddress();
            if (null == hostName) hostName = "local";
          } catch (Throwable e) {
            throw new RuntimeException(e);
          }
        }
      }
    }
    return hostName;
  }

  @Nullable
  public OutputStream getOutput() throws IOException {
    if (null == fileOutputStream) {
      synchronized (this) {
        if (null == fileOutputStream) {
          String[] split = location.getName().split("\\.", 2);
          fileOutputStream = new FileOutputStream(new File(location.getParentFile(), String.format("%s.out.tfevents.%d.%s",
              split[0],
              System.currentTimeMillis() / 1000,
              getHostName(),
              split.length == 2 ? split[1] : ""
          )));
          write(fileOutputStream, Event.newBuilder()
              .setWallTime(System.currentTimeMillis() / 1000)
              .setFileVersion("brain.Event.2")
              .build()
              .toByteArray());
        }
      }
    }
    return fileOutputStream;
  }

  public long getStep() {
    return step;
  }

  @Nonnull
  public TensorboardEventWriter setStep(long step) {
    this.step = step;
    return this;
  }

  public static void write(@Nonnull OutputStream dataInput, @Nonnull byte[] data) throws IOException {
    byte[] header = new byte[12];
    setInt(header, 0, 8, data.length);
    setInt(header, 8, 4, mask(longHash(header, 0, 8)));
    byte[] footer = new byte[4];
    setInt(footer, 0, 4, mask(longHash(data, 0, data.length)));
    dataInput.write(header);
    dataInput.write(data);
    dataInput.write(footer);
  }

  @Nonnull
  public static byte[] read(@Nonnull DataInputStream dataInput) throws IOException {
    byte[] header = new byte[12];
    dataInput.readFully(header);
    long length = getInt(header, 0, 8);
    long masked_crc = getInt(header, 8, 4);
    long len_hash_calc = longHash(header, 0, 8);
    if (unmask(masked_crc) != len_hash_calc) throw new RuntimeException(String.format("%s != %s",
        Long.toHexString(unmask(masked_crc)),
        Long.toHexString(len_hash_calc)));
    if (0 >= length) throw new RuntimeException("length=" + length);
    byte[] data = new byte[(int) length];
    dataInput.readFully(data);
    byte[] footer = new byte[4];
    dataInput.readFully(footer);
    if (unmask(getInt(footer, 0, 4)) != longHash(data, 0, data.length)) throw new RuntimeException(String.format("%s != %s",
        Long.toHexString(unmask(getInt(footer, 0, 4))),
        Long.toHexString(longHash(data, 0, data.length))));
    return data;
  }

  public static long longHash(@Nonnull byte[] bytes, int start, int length) {
    return getInt(Hashing.crc32c().newHasher().putBytes(bytes, start, length).hash().asBytes(), 0, 4);
  }

  public static long unmask(long masked_crc) {
    long rot = (masked_crc > kMaskDelta) ? (masked_crc - kMaskDelta) : (masked_crc + ((intMask + 1) - kMaskDelta));
    return (((rot >>> 17) | (rot << 15)) & intMask) & intMask;
  }

  public static long mask(long crc) {
    return ((((crc >>> 15) | (crc << 17)) & intMask) + kMaskDelta) & intMask;
  }

  public static long getInt(byte[] bytes, int start, int length) {
    long value = 0;
    for (int offset = 0; offset < length; offset++) {
      value += (bytes[start + offset] & 0xFFL) << (offset * 8);
    }
    return value;
  }

  private static long setInt(byte[] bytes, int start, int length, long value) {
    for (int offset = 0; offset < length; offset++) {
      bytes[start + offset] = (byte) ((value >> (offset * 8)) & 0xFF);
    }
    return value;
  }

  public void writeSummary(byte[] summaryBytes) {
    try {
      write(Summary.parseFrom(summaryBytes));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void write(Summary summary) throws IOException {
    write(Event.newBuilder()
        .setSummary(summary)
        .build());
  }

  public void write(LogMessage message) throws IOException {
    write(Event.newBuilder()
        .setLogMessage(message)
        .build());
  }

  public void write(SessionLog sessionLog) throws IOException {
    write(Event.newBuilder()
        .setSessionLog(sessionLog)
        .build());
  }

  public void write(@Nonnull GraphDef graphDef) throws IOException {
    write(Event.newBuilder()
        .setGraphDef(graphDef.toByteString())
        .build());
  }

  public void write(@Nonnull MetaGraphDef metaGraphDef) throws IOException {
    write(Event.newBuilder()
        .setMetaGraphDef(metaGraphDef.toByteString())
        .build());
  }

  public void write(TaggedRunMetadata taggedRunMetadata) throws IOException {
    write(Event.newBuilder()
        .setTaggedRunMetadata(taggedRunMetadata)
        .build());
  }

  public void write(@Nonnull RunMetadata runMetadata, @Nonnull String tag) throws IOException {
    write(Event.newBuilder()
        .setTaggedRunMetadata(TaggedRunMetadata.newBuilder().setRunMetadata(runMetadata.toByteString()).setTag(tag).build())
        .build());
  }

  public void write(@Nonnull Event event) throws IOException {
    OutputStream output = getOutput();
    assert output != null;
    write(output, event.toBuilder()
        .setWallTime(System.currentTimeMillis() / 1000)
        .setStep(getStep())
        .build()
        .toByteArray());
    output.flush();
  }

  @Override
  public void close() throws IOException {
    synchronized (this) {
      fileOutputStream.close();
      fileOutputStream = null;
    }
  }

  @Nonnull
  public TensorboardEventWriter incStep(long step) {
    this.step += step;
    return this;
  }
}
