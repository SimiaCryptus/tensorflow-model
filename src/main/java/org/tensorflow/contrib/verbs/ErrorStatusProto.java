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
// source: verbs_service.proto

package org.tensorflow.contrib.verbs;

/**
 * Protobuf type {@code tensorflow.ErrorStatusProto}
 */
public  final class ErrorStatusProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.ErrorStatusProto)
    ErrorStatusProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use ErrorStatusProto.newBuilder() to construct.
  private ErrorStatusProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ErrorStatusProto() {
    errorCode_ = 0;
    errorMessage_ = "";
    errorDetails_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ErrorStatusProto(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          case 8: {

            errorCode_ = input.readInt32();
            break;
          }
          case 18: {
            java.lang.String s = input.readStringRequireUtf8();

            errorMessage_ = s;
            break;
          }
          case 26: {
            java.lang.String s = input.readStringRequireUtf8();

            errorDetails_ = s;
            break;
          }
          default: {
            if (!parseUnknownFieldProto3(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tensorflow.contrib.verbs.VerbsServiceProtos.internal_static_tensorflow_ErrorStatusProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.contrib.verbs.VerbsServiceProtos.internal_static_tensorflow_ErrorStatusProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.contrib.verbs.ErrorStatusProto.class, org.tensorflow.contrib.verbs.ErrorStatusProto.Builder.class);
  }

  public static final int ERROR_CODE_FIELD_NUMBER = 1;
  private int errorCode_;
  /**
   * <code>int32 error_code = 1;</code>
   */
  public int getErrorCode() {
    return errorCode_;
  }

  public static final int ERROR_MESSAGE_FIELD_NUMBER = 2;
  private volatile java.lang.Object errorMessage_;
  /**
   * <code>string error_message = 2;</code>
   */
  public java.lang.String getErrorMessage() {
    java.lang.Object ref = errorMessage_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      errorMessage_ = s;
      return s;
    }
  }
  /**
   * <code>string error_message = 2;</code>
   */
  public com.google.protobuf.ByteString
      getErrorMessageBytes() {
    java.lang.Object ref = errorMessage_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      errorMessage_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int ERROR_DETAILS_FIELD_NUMBER = 3;
  private volatile java.lang.Object errorDetails_;
  /**
   * <code>string error_details = 3;</code>
   */
  public java.lang.String getErrorDetails() {
    java.lang.Object ref = errorDetails_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      errorDetails_ = s;
      return s;
    }
  }
  /**
   * <code>string error_details = 3;</code>
   */
  public com.google.protobuf.ByteString
      getErrorDetailsBytes() {
    java.lang.Object ref = errorDetails_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      errorDetails_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  private byte memoizedIsInitialized = -1;
  @java.lang.Override
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  @java.lang.Override
  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (errorCode_ != 0) {
      output.writeInt32(1, errorCode_);
    }
    if (!getErrorMessageBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, errorMessage_);
    }
    if (!getErrorDetailsBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 3, errorDetails_);
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (errorCode_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(1, errorCode_);
    }
    if (!getErrorMessageBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, errorMessage_);
    }
    if (!getErrorDetailsBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(3, errorDetails_);
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.tensorflow.contrib.verbs.ErrorStatusProto)) {
      return super.equals(obj);
    }
    org.tensorflow.contrib.verbs.ErrorStatusProto other = (org.tensorflow.contrib.verbs.ErrorStatusProto) obj;

    boolean result = true;
    result = result && (getErrorCode()
        == other.getErrorCode());
    result = result && getErrorMessage()
        .equals(other.getErrorMessage());
    result = result && getErrorDetails()
        .equals(other.getErrorDetails());
    result = result && unknownFields.equals(other.unknownFields);
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    hash = (37 * hash) + ERROR_CODE_FIELD_NUMBER;
    hash = (53 * hash) + getErrorCode();
    hash = (37 * hash) + ERROR_MESSAGE_FIELD_NUMBER;
    hash = (53 * hash) + getErrorMessage().hashCode();
    hash = (37 * hash) + ERROR_DETAILS_FIELD_NUMBER;
    hash = (53 * hash) + getErrorDetails().hashCode();
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.contrib.verbs.ErrorStatusProto parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  @java.lang.Override
  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.tensorflow.contrib.verbs.ErrorStatusProto prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  @java.lang.Override
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * Protobuf type {@code tensorflow.ErrorStatusProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.ErrorStatusProto)
      org.tensorflow.contrib.verbs.ErrorStatusProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.contrib.verbs.VerbsServiceProtos.internal_static_tensorflow_ErrorStatusProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.contrib.verbs.VerbsServiceProtos.internal_static_tensorflow_ErrorStatusProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.contrib.verbs.ErrorStatusProto.class, org.tensorflow.contrib.verbs.ErrorStatusProto.Builder.class);
    }

    // Construct using org.tensorflow.contrib.verbs.ErrorStatusProto.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      errorCode_ = 0;

      errorMessage_ = "";

      errorDetails_ = "";

      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.contrib.verbs.VerbsServiceProtos.internal_static_tensorflow_ErrorStatusProto_descriptor;
    }

    @java.lang.Override
    public org.tensorflow.contrib.verbs.ErrorStatusProto getDefaultInstanceForType() {
      return org.tensorflow.contrib.verbs.ErrorStatusProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tensorflow.contrib.verbs.ErrorStatusProto build() {
      org.tensorflow.contrib.verbs.ErrorStatusProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tensorflow.contrib.verbs.ErrorStatusProto buildPartial() {
      org.tensorflow.contrib.verbs.ErrorStatusProto result = new org.tensorflow.contrib.verbs.ErrorStatusProto(this);
      result.errorCode_ = errorCode_;
      result.errorMessage_ = errorMessage_;
      result.errorDetails_ = errorDetails_;
      onBuilt();
      return result;
    }

    @java.lang.Override
    public Builder clone() {
      return super.clone();
    }
    @java.lang.Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.setField(field, value);
    }
    @java.lang.Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return super.clearField(field);
    }
    @java.lang.Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return super.clearOneof(oneof);
    }
    @java.lang.Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return super.setRepeatedField(field, index, value);
    }
    @java.lang.Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.addRepeatedField(field, value);
    }
    @java.lang.Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.tensorflow.contrib.verbs.ErrorStatusProto) {
        return mergeFrom((org.tensorflow.contrib.verbs.ErrorStatusProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.contrib.verbs.ErrorStatusProto other) {
      if (other == org.tensorflow.contrib.verbs.ErrorStatusProto.getDefaultInstance()) return this;
      if (other.getErrorCode() != 0) {
        setErrorCode(other.getErrorCode());
      }
      if (!other.getErrorMessage().isEmpty()) {
        errorMessage_ = other.errorMessage_;
        onChanged();
      }
      if (!other.getErrorDetails().isEmpty()) {
        errorDetails_ = other.errorDetails_;
        onChanged();
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    @java.lang.Override
    public final boolean isInitialized() {
      return true;
    }

    @java.lang.Override
    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.tensorflow.contrib.verbs.ErrorStatusProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.contrib.verbs.ErrorStatusProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }

    private int errorCode_ ;
    /**
     * <code>int32 error_code = 1;</code>
     */
    public int getErrorCode() {
      return errorCode_;
    }
    /**
     * <code>int32 error_code = 1;</code>
     */
    public Builder setErrorCode(int value) {
      
      errorCode_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 error_code = 1;</code>
     */
    public Builder clearErrorCode() {
      
      errorCode_ = 0;
      onChanged();
      return this;
    }

    private java.lang.Object errorMessage_ = "";
    /**
     * <code>string error_message = 2;</code>
     */
    public java.lang.String getErrorMessage() {
      java.lang.Object ref = errorMessage_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        errorMessage_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string error_message = 2;</code>
     */
    public com.google.protobuf.ByteString
        getErrorMessageBytes() {
      java.lang.Object ref = errorMessage_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        errorMessage_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string error_message = 2;</code>
     */
    public Builder setErrorMessage(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      errorMessage_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string error_message = 2;</code>
     */
    public Builder clearErrorMessage() {
      
      errorMessage_ = getDefaultInstance().getErrorMessage();
      onChanged();
      return this;
    }
    /**
     * <code>string error_message = 2;</code>
     */
    public Builder setErrorMessageBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      errorMessage_ = value;
      onChanged();
      return this;
    }

    private java.lang.Object errorDetails_ = "";
    /**
     * <code>string error_details = 3;</code>
     */
    public java.lang.String getErrorDetails() {
      java.lang.Object ref = errorDetails_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        errorDetails_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string error_details = 3;</code>
     */
    public com.google.protobuf.ByteString
        getErrorDetailsBytes() {
      java.lang.Object ref = errorDetails_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        errorDetails_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string error_details = 3;</code>
     */
    public Builder setErrorDetails(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      errorDetails_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string error_details = 3;</code>
     */
    public Builder clearErrorDetails() {
      
      errorDetails_ = getDefaultInstance().getErrorDetails();
      onChanged();
      return this;
    }
    /**
     * <code>string error_details = 3;</code>
     */
    public Builder setErrorDetailsBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      errorDetails_ = value;
      onChanged();
      return this;
    }
    @java.lang.Override
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFieldsProto3(unknownFields);
    }

    @java.lang.Override
    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:tensorflow.ErrorStatusProto)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.ErrorStatusProto)
  private static final org.tensorflow.contrib.verbs.ErrorStatusProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.contrib.verbs.ErrorStatusProto();
  }

  public static org.tensorflow.contrib.verbs.ErrorStatusProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<ErrorStatusProto>
      PARSER = new com.google.protobuf.AbstractParser<ErrorStatusProto>() {
    @java.lang.Override
    public ErrorStatusProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new ErrorStatusProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ErrorStatusProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ErrorStatusProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tensorflow.contrib.verbs.ErrorStatusProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

