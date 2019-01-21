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

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: test_log.proto

package org.tensorflow.util.testlog;

public interface MemoryInfoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.MemoryInfo)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Total virtual memory in bytes
   * </pre>
   *
   * <code>int64 total = 1;</code>
   */
  long getTotal();

  /**
   * <pre>
   * Immediately available memory in bytes
   * </pre>
   *
   * <code>int64 available = 2;</code>
   */
  long getAvailable();
}
