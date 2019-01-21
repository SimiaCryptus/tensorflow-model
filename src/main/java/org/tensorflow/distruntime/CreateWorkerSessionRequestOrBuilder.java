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
// source: worker.proto

package org.tensorflow.distruntime;

public interface CreateWorkerSessionRequestOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.CreateWorkerSessionRequest)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Sessions are identified by a given handle.
   * </pre>
   *
   * <code>string session_handle = 1;</code>
   */
  java.lang.String getSessionHandle();
  /**
   * <pre>
   * Sessions are identified by a given handle.
   * </pre>
   *
   * <code>string session_handle = 1;</code>
   */
  com.google.protobuf.ByteString
      getSessionHandleBytes();

  /**
   * <pre>
   * Defines the configuration of a TensorFlow worker.
   * </pre>
   *
   * <code>.tensorflow.ServerDef server_def = 2;</code>
   */
  boolean hasServerDef();
  /**
   * <pre>
   * Defines the configuration of a TensorFlow worker.
   * </pre>
   *
   * <code>.tensorflow.ServerDef server_def = 2;</code>
   */
  org.tensorflow.distruntime.ServerDef getServerDef();
  /**
   * <pre>
   * Defines the configuration of a TensorFlow worker.
   * </pre>
   *
   * <code>.tensorflow.ServerDef server_def = 2;</code>
   */
  org.tensorflow.distruntime.ServerDefOrBuilder getServerDefOrBuilder();

  /**
   * <pre>
   * If true, any resources such as Variables used in the session will not be
   * shared with other sessions.
   * </pre>
   *
   * <code>bool isolate_session_state = 3;</code>
   */
  boolean getIsolateSessionState();
}
