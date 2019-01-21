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
// source: tf2xla.proto

package org.tensorflow.tf2xla;

public final class Tf2XlaProtos {
  private Tf2XlaProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_tf2xla_TensorId_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_tf2xla_TensorId_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_tf2xla_Feed_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_tf2xla_Feed_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_tf2xla_Fetch_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_tf2xla_Fetch_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_tf2xla_Config_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_tf2xla_Config_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\014tf2xla.proto\022\021tensorflow.tf2xla\032,tenso" +
      "rflow/core/framework/tensor_shape.proto\032" +
      "%tensorflow/core/framework/types.proto\"3" +
      "\n\010TensorId\022\021\n\tnode_name\030\001 \001(\t\022\024\n\014output_" +
      "index\030\002 \001(\003\"\216\001\n\004Feed\022\'\n\002id\030\001 \001(\0132\033.tenso" +
      "rflow.tf2xla.TensorId\022+\n\005shape\030\002 \001(\0132\034.t" +
      "ensorflow.TensorShapeProto\022\014\n\004name\030\003 \001(\t" +
      "\022\"\n\004type\030\004 \001(\0162\024.tensorflow.DataType\">\n\005" +
      "Fetch\022\'\n\002id\030\001 \001(\0132\033.tensorflow.tf2xla.Te" +
      "nsorId\022\014\n\004name\030\002 \001(\t\"X\n\006Config\022%\n\004feed\030\001" +
      " \003(\0132\027.tensorflow.tf2xla.Feed\022\'\n\005fetch\030\002" +
      " \003(\0132\030.tensorflow.tf2xla.FetchB*\n\025org.te" +
      "nsorflow.tf2xlaB\014Tf2XlaProtosP\001\370\001\001b\006prot" +
      "o3"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tensorflow.framework.TensorShapeProtos.getDescriptor(),
          org.tensorflow.framework.TypesProtos.getDescriptor(),
        }, assigner);
    internal_static_tensorflow_tf2xla_TensorId_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_tf2xla_TensorId_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_tf2xla_TensorId_descriptor,
        new java.lang.String[] { "NodeName", "OutputIndex", });
    internal_static_tensorflow_tf2xla_Feed_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tensorflow_tf2xla_Feed_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_tf2xla_Feed_descriptor,
        new java.lang.String[] { "Id", "Shape", "Name", "Type", });
    internal_static_tensorflow_tf2xla_Fetch_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_tensorflow_tf2xla_Fetch_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_tf2xla_Fetch_descriptor,
        new java.lang.String[] { "Id", "Name", });
    internal_static_tensorflow_tf2xla_Config_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_tensorflow_tf2xla_Config_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_tf2xla_Config_descriptor,
        new java.lang.String[] { "Feed", "Fetch", });
    org.tensorflow.framework.TensorShapeProtos.getDescriptor();
    org.tensorflow.framework.TypesProtos.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
