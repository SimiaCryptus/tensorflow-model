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
// source: queue_runner.proto

package org.tensorflow.framework;

public final class QueueRunnerProtos {
  private QueueRunnerProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_QueueRunnerDef_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_QueueRunnerDef_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\022queue_runner.proto\022\ntensorflow\032*tensor" +
      "flow/core/lib/core/error_codes.proto\"\252\001\n" +
      "\016QueueRunnerDef\022\022\n\nqueue_name\030\001 \001(\t\022\027\n\017e" +
      "nqueue_op_name\030\002 \003(\t\022\025\n\rclose_op_name\030\003 " +
      "\001(\t\022\026\n\016cancel_op_name\030\004 \001(\t\022<\n\034queue_clo" +
      "sed_exception_types\030\005 \003(\0162\026.tensorflow.e" +
      "rror.CodeBp\n\030org.tensorflow.frameworkB\021Q" +
      "ueueRunnerProtosP\001Z<github.com/tensorflo" +
      "w/tensorflow/tensorflow/go/core/protobuf" +
      "\370\001\001b\006proto3"
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
          org.tensorflow.framework.ErrorCodesProtos.getDescriptor(),
        }, assigner);
    internal_static_tensorflow_QueueRunnerDef_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_QueueRunnerDef_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_QueueRunnerDef_descriptor,
        new java.lang.String[] { "QueueName", "EnqueueOpName", "CloseOpName", "CancelOpName", "QueueClosedExceptionTypes", });
    org.tensorflow.framework.ErrorCodesProtos.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
