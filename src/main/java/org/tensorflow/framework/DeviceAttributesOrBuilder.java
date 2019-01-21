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
// source: device_attributes.proto

package org.tensorflow.framework;

public interface DeviceAttributesOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.DeviceAttributes)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Fully specified name of the device within a cluster.
   * </pre>
   *
   * <code>string name = 1;</code>
   */
  java.lang.String getName();
  /**
   * <pre>
   * Fully specified name of the device within a cluster.
   * </pre>
   *
   * <code>string name = 1;</code>
   */
  com.google.protobuf.ByteString
      getNameBytes();

  /**
   * <pre>
   * String representation of device_type.
   * </pre>
   *
   * <code>string device_type = 2;</code>
   */
  java.lang.String getDeviceType();
  /**
   * <pre>
   * String representation of device_type.
   * </pre>
   *
   * <code>string device_type = 2;</code>
   */
  com.google.protobuf.ByteString
      getDeviceTypeBytes();

  /**
   * <pre>
   * Memory capacity of device in bytes.
   * </pre>
   *
   * <code>int64 memory_limit = 4;</code>
   */
  long getMemoryLimit();

  /**
   * <pre>
   * Platform-specific data about device that may be useful
   * for supporting efficient data transfers.
   * </pre>
   *
   * <code>.tensorflow.DeviceLocality locality = 5;</code>
   */
  boolean hasLocality();
  /**
   * <pre>
   * Platform-specific data about device that may be useful
   * for supporting efficient data transfers.
   * </pre>
   *
   * <code>.tensorflow.DeviceLocality locality = 5;</code>
   */
  org.tensorflow.framework.DeviceLocality getLocality();
  /**
   * <pre>
   * Platform-specific data about device that may be useful
   * for supporting efficient data transfers.
   * </pre>
   *
   * <code>.tensorflow.DeviceLocality locality = 5;</code>
   */
  org.tensorflow.framework.DeviceLocalityOrBuilder getLocalityOrBuilder();

  /**
   * <pre>
   * A device is assigned a global unique number each time it is
   * initialized. "incarnation" should never be 0.
   * </pre>
   *
   * <code>fixed64 incarnation = 6;</code>
   */
  long getIncarnation();

  /**
   * <pre>
   * String representation of the physical device that this device maps to.
   * </pre>
   *
   * <code>string physical_device_desc = 7;</code>
   */
  java.lang.String getPhysicalDeviceDesc();
  /**
   * <pre>
   * String representation of the physical device that this device maps to.
   * </pre>
   *
   * <code>string physical_device_desc = 7;</code>
   */
  com.google.protobuf.ByteString
      getPhysicalDeviceDescBytes();
}
