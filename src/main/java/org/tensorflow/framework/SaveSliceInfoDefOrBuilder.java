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
// source: variable.proto

package org.tensorflow.framework;

public interface SaveSliceInfoDefOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.SaveSliceInfoDef)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Name of the full variable of which this is a slice.
   * </pre>
   *
   * <code>string full_name = 1;</code>
   */
  java.lang.String getFullName();
  /**
   * <pre>
   * Name of the full variable of which this is a slice.
   * </pre>
   *
   * <code>string full_name = 1;</code>
   */
  com.google.protobuf.ByteString
      getFullNameBytes();

  /**
   * <pre>
   * Shape of the full variable.
   * </pre>
   *
   * <code>repeated int64 full_shape = 2;</code>
   */
  java.util.List<java.lang.Long> getFullShapeList();
  /**
   * <pre>
   * Shape of the full variable.
   * </pre>
   *
   * <code>repeated int64 full_shape = 2;</code>
   */
  int getFullShapeCount();
  /**
   * <pre>
   * Shape of the full variable.
   * </pre>
   *
   * <code>repeated int64 full_shape = 2;</code>
   */
  long getFullShape(int index);

  /**
   * <pre>
   * Offset of this variable into the full variable.
   * </pre>
   *
   * <code>repeated int64 var_offset = 3;</code>
   */
  java.util.List<java.lang.Long> getVarOffsetList();
  /**
   * <pre>
   * Offset of this variable into the full variable.
   * </pre>
   *
   * <code>repeated int64 var_offset = 3;</code>
   */
  int getVarOffsetCount();
  /**
   * <pre>
   * Offset of this variable into the full variable.
   * </pre>
   *
   * <code>repeated int64 var_offset = 3;</code>
   */
  long getVarOffset(int index);

  /**
   * <pre>
   * Shape of this variable.
   * </pre>
   *
   * <code>repeated int64 var_shape = 4;</code>
   */
  java.util.List<java.lang.Long> getVarShapeList();
  /**
   * <pre>
   * Shape of this variable.
   * </pre>
   *
   * <code>repeated int64 var_shape = 4;</code>
   */
  int getVarShapeCount();
  /**
   * <pre>
   * Shape of this variable.
   * </pre>
   *
   * <code>repeated int64 var_shape = 4;</code>
   */
  long getVarShape(int index);
}
