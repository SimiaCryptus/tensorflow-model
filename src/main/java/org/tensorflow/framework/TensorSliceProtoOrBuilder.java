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
// source: tensor_slice.proto

package org.tensorflow.framework;

public interface TensorSliceProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.TensorSliceProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Extent of the slice in all tensor dimensions.
   * Must have one entry for each of the dimension of the tensor that this
   * slice belongs to.  The order of sizes is the same as the order of
   * dimensions in the TensorShape.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto.Extent extent = 1;</code>
   */
  java.util.List<org.tensorflow.framework.TensorSliceProto.Extent> 
      getExtentList();
  /**
   * <pre>
   * Extent of the slice in all tensor dimensions.
   * Must have one entry for each of the dimension of the tensor that this
   * slice belongs to.  The order of sizes is the same as the order of
   * dimensions in the TensorShape.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto.Extent extent = 1;</code>
   */
  org.tensorflow.framework.TensorSliceProto.Extent getExtent(int index);
  /**
   * <pre>
   * Extent of the slice in all tensor dimensions.
   * Must have one entry for each of the dimension of the tensor that this
   * slice belongs to.  The order of sizes is the same as the order of
   * dimensions in the TensorShape.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto.Extent extent = 1;</code>
   */
  int getExtentCount();
  /**
   * <pre>
   * Extent of the slice in all tensor dimensions.
   * Must have one entry for each of the dimension of the tensor that this
   * slice belongs to.  The order of sizes is the same as the order of
   * dimensions in the TensorShape.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto.Extent extent = 1;</code>
   */
  java.util.List<? extends org.tensorflow.framework.TensorSliceProto.ExtentOrBuilder> 
      getExtentOrBuilderList();
  /**
   * <pre>
   * Extent of the slice in all tensor dimensions.
   * Must have one entry for each of the dimension of the tensor that this
   * slice belongs to.  The order of sizes is the same as the order of
   * dimensions in the TensorShape.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto.Extent extent = 1;</code>
   */
  org.tensorflow.framework.TensorSliceProto.ExtentOrBuilder getExtentOrBuilder(
      int index);
}
