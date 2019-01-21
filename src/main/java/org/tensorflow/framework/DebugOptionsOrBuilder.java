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
// source: debug.proto

package org.tensorflow.framework;

public interface DebugOptionsOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.DebugOptions)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Debugging options
   * </pre>
   *
   * <code>repeated .tensorflow.DebugTensorWatch debug_tensor_watch_opts = 4;</code>
   */
  java.util.List<org.tensorflow.framework.DebugTensorWatch> 
      getDebugTensorWatchOptsList();
  /**
   * <pre>
   * Debugging options
   * </pre>
   *
   * <code>repeated .tensorflow.DebugTensorWatch debug_tensor_watch_opts = 4;</code>
   */
  org.tensorflow.framework.DebugTensorWatch getDebugTensorWatchOpts(int index);
  /**
   * <pre>
   * Debugging options
   * </pre>
   *
   * <code>repeated .tensorflow.DebugTensorWatch debug_tensor_watch_opts = 4;</code>
   */
  int getDebugTensorWatchOptsCount();
  /**
   * <pre>
   * Debugging options
   * </pre>
   *
   * <code>repeated .tensorflow.DebugTensorWatch debug_tensor_watch_opts = 4;</code>
   */
  java.util.List<? extends org.tensorflow.framework.DebugTensorWatchOrBuilder> 
      getDebugTensorWatchOptsOrBuilderList();
  /**
   * <pre>
   * Debugging options
   * </pre>
   *
   * <code>repeated .tensorflow.DebugTensorWatch debug_tensor_watch_opts = 4;</code>
   */
  org.tensorflow.framework.DebugTensorWatchOrBuilder getDebugTensorWatchOptsOrBuilder(
      int index);

  /**
   * <pre>
   * Caller-specified global step count.
   * Note that this is distinct from the session run count and the executor
   * step count.
   * </pre>
   *
   * <code>int64 global_step = 10;</code>
   */
  long getGlobalStep();

  /**
   * <pre>
   * Whether the total disk usage of tfdbg is to be reset to zero
   * in this Session.run call. This is used by wrappers and hooks
   * such as the local CLI ones to indicate that the dumped tensors
   * are cleaned up from the disk after each Session.run.
   * </pre>
   *
   * <code>bool reset_disk_byte_usage = 11;</code>
   */
  boolean getResetDiskByteUsage();
}
