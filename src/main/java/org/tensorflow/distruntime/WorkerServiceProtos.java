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
// source: worker_service.proto

package org.tensorflow.distruntime;

public final class WorkerServiceProtos {
  private WorkerServiceProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\024worker_service.proto\022\017tensorflow.grpc\032" +
      "%tensorflow/core/protobuf/worker.proto2\360" +
      "\t\n\rWorkerService\022H\n\tGetStatus\022\034.tensorfl" +
      "ow.GetStatusRequest\032\035.tensorflow.GetStat" +
      "usResponse\022f\n\023CreateWorkerSession\022&.tens" +
      "orflow.CreateWorkerSessionRequest\032\'.tens" +
      "orflow.CreateWorkerSessionResponse\022f\n\023De" +
      "leteWorkerSession\022&.tensorflow.DeleteWor" +
      "kerSessionRequest\032\'.tensorflow.DeleteWor" +
      "kerSessionResponse\022T\n\rRegisterGraph\022 .te" +
      "nsorflow.RegisterGraphRequest\032!.tensorfl" +
      "ow.RegisterGraphResponse\022Z\n\017DeregisterGr" +
      "aph\022\".tensorflow.DeregisterGraphRequest\032" +
      "#.tensorflow.DeregisterGraphResponse\022E\n\010" +
      "RunGraph\022\033.tensorflow.RunGraphRequest\032\034." +
      "tensorflow.RunGraphResponse\022Q\n\014CleanupGr" +
      "aph\022\037.tensorflow.CleanupGraphRequest\032 .t" +
      "ensorflow.CleanupGraphResponse\022K\n\nCleanu" +
      "pAll\022\035.tensorflow.CleanupAllRequest\032\036.te" +
      "nsorflow.CleanupAllResponse\022M\n\nRecvTenso" +
      "r\022\035.tensorflow.RecvTensorRequest\032\036.tenso" +
      "rflow.RecvTensorResponse\"\000\022B\n\007Logging\022\032." +
      "tensorflow.LoggingRequest\032\033.tensorflow.L" +
      "oggingResponse\022B\n\007Tracing\022\032.tensorflow.T" +
      "racingRequest\032\033.tensorflow.TracingRespon" +
      "se\022D\n\007RecvBuf\022\032.tensorflow.RecvBufReques" +
      "t\032\033.tensorflow.RecvBufResponse\"\000\022Z\n\017GetS" +
      "tepSequence\022\".tensorflow.GetStepSequence" +
      "Request\032#.tensorflow.GetStepSequenceResp" +
      "onse\022T\n\rCompleteGroup\022 .tensorflow.Compl" +
      "eteGroupRequest\032!.tensorflow.CompleteGro" +
      "upResponse\022]\n\020CompleteInstance\022#.tensorf" +
      "low.CompleteInstanceRequest\032$.tensorflow" +
      ".CompleteInstanceResponseBq\n\032org.tensorf" +
      "low.distruntimeB\023WorkerServiceProtosP\001Z<" +
      "github.com/tensorflow/tensorflow/tensorf" +
      "low/go/core/protobufb\006proto3"
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
          org.tensorflow.distruntime.WorkerProtos.getDescriptor(),
        }, assigner);
    org.tensorflow.distruntime.WorkerProtos.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
