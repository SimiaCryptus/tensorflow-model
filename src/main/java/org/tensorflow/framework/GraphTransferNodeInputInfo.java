// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: graph_transfer_info.proto

package org.tensorflow.framework;

/**
 * Protobuf type {@code tensorflow.GraphTransferNodeInputInfo}
 */
public  final class GraphTransferNodeInputInfo extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.GraphTransferNodeInputInfo)
    GraphTransferNodeInputInfoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use GraphTransferNodeInputInfo.newBuilder() to construct.
  private GraphTransferNodeInputInfo(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private GraphTransferNodeInputInfo() {
    nodeId_ = 0;
    nodeInput_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private GraphTransferNodeInputInfo(
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

            nodeId_ = input.readInt32();
            break;
          }
          case 18: {
            if (!((mutable_bitField0_ & 0x00000002) == 0x00000002)) {
              nodeInput_ = new java.util.ArrayList<org.tensorflow.framework.GraphTransferNodeInput>();
              mutable_bitField0_ |= 0x00000002;
            }
            nodeInput_.add(
                input.readMessage(org.tensorflow.framework.GraphTransferNodeInput.parser(), extensionRegistry));
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
      if (((mutable_bitField0_ & 0x00000002) == 0x00000002)) {
        nodeInput_ = java.util.Collections.unmodifiableList(nodeInput_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tensorflow.framework.GraphTransferInfoProto.internal_static_tensorflow_GraphTransferNodeInputInfo_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.framework.GraphTransferInfoProto.internal_static_tensorflow_GraphTransferNodeInputInfo_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.framework.GraphTransferNodeInputInfo.class, org.tensorflow.framework.GraphTransferNodeInputInfo.Builder.class);
  }

  private int bitField0_;
  public static final int NODE_ID_FIELD_NUMBER = 1;
  private int nodeId_;
  /**
   * <code>int32 node_id = 1;</code>
   */
  public int getNodeId() {
    return nodeId_;
  }

  public static final int NODE_INPUT_FIELD_NUMBER = 2;
  private java.util.List<org.tensorflow.framework.GraphTransferNodeInput> nodeInput_;
  /**
   * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
   */
  public java.util.List<org.tensorflow.framework.GraphTransferNodeInput> getNodeInputList() {
    return nodeInput_;
  }
  /**
   * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
   */
  public java.util.List<? extends org.tensorflow.framework.GraphTransferNodeInputOrBuilder> 
      getNodeInputOrBuilderList() {
    return nodeInput_;
  }
  /**
   * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
   */
  public int getNodeInputCount() {
    return nodeInput_.size();
  }
  /**
   * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
   */
  public org.tensorflow.framework.GraphTransferNodeInput getNodeInput(int index) {
    return nodeInput_.get(index);
  }
  /**
   * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
   */
  public org.tensorflow.framework.GraphTransferNodeInputOrBuilder getNodeInputOrBuilder(
      int index) {
    return nodeInput_.get(index);
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
    if (nodeId_ != 0) {
      output.writeInt32(1, nodeId_);
    }
    for (int i = 0; i < nodeInput_.size(); i++) {
      output.writeMessage(2, nodeInput_.get(i));
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (nodeId_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(1, nodeId_);
    }
    for (int i = 0; i < nodeInput_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, nodeInput_.get(i));
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
    if (!(obj instanceof org.tensorflow.framework.GraphTransferNodeInputInfo)) {
      return super.equals(obj);
    }
    org.tensorflow.framework.GraphTransferNodeInputInfo other = (org.tensorflow.framework.GraphTransferNodeInputInfo) obj;

    boolean result = true;
    result = result && (getNodeId()
        == other.getNodeId());
    result = result && getNodeInputList()
        .equals(other.getNodeInputList());
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
    hash = (37 * hash) + NODE_ID_FIELD_NUMBER;
    hash = (53 * hash) + getNodeId();
    if (getNodeInputCount() > 0) {
      hash = (37 * hash) + NODE_INPUT_FIELD_NUMBER;
      hash = (53 * hash) + getNodeInputList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.GraphTransferNodeInputInfo parseFrom(
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
  public static Builder newBuilder(org.tensorflow.framework.GraphTransferNodeInputInfo prototype) {
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
   * Protobuf type {@code tensorflow.GraphTransferNodeInputInfo}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.GraphTransferNodeInputInfo)
      org.tensorflow.framework.GraphTransferNodeInputInfoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.framework.GraphTransferInfoProto.internal_static_tensorflow_GraphTransferNodeInputInfo_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.framework.GraphTransferInfoProto.internal_static_tensorflow_GraphTransferNodeInputInfo_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.framework.GraphTransferNodeInputInfo.class, org.tensorflow.framework.GraphTransferNodeInputInfo.Builder.class);
    }

    // Construct using org.tensorflow.framework.GraphTransferNodeInputInfo.newBuilder()
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
        getNodeInputFieldBuilder();
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      nodeId_ = 0;

      if (nodeInputBuilder_ == null) {
        nodeInput_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000002);
      } else {
        nodeInputBuilder_.clear();
      }
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.framework.GraphTransferInfoProto.internal_static_tensorflow_GraphTransferNodeInputInfo_descriptor;
    }

    @java.lang.Override
    public org.tensorflow.framework.GraphTransferNodeInputInfo getDefaultInstanceForType() {
      return org.tensorflow.framework.GraphTransferNodeInputInfo.getDefaultInstance();
    }

    @java.lang.Override
    public org.tensorflow.framework.GraphTransferNodeInputInfo build() {
      org.tensorflow.framework.GraphTransferNodeInputInfo result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tensorflow.framework.GraphTransferNodeInputInfo buildPartial() {
      org.tensorflow.framework.GraphTransferNodeInputInfo result = new org.tensorflow.framework.GraphTransferNodeInputInfo(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      result.nodeId_ = nodeId_;
      if (nodeInputBuilder_ == null) {
        if (((bitField0_ & 0x00000002) == 0x00000002)) {
          nodeInput_ = java.util.Collections.unmodifiableList(nodeInput_);
          bitField0_ = (bitField0_ & ~0x00000002);
        }
        result.nodeInput_ = nodeInput_;
      } else {
        result.nodeInput_ = nodeInputBuilder_.build();
      }
      result.bitField0_ = to_bitField0_;
      onBuilt();
      return result;
    }

    @java.lang.Override
    public Builder clone() {
      return (Builder) super.clone();
    }
    @java.lang.Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.setField(field, value);
    }
    @java.lang.Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return (Builder) super.clearField(field);
    }
    @java.lang.Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return (Builder) super.clearOneof(oneof);
    }
    @java.lang.Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return (Builder) super.setRepeatedField(field, index, value);
    }
    @java.lang.Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.addRepeatedField(field, value);
    }
    @java.lang.Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.tensorflow.framework.GraphTransferNodeInputInfo) {
        return mergeFrom((org.tensorflow.framework.GraphTransferNodeInputInfo)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.framework.GraphTransferNodeInputInfo other) {
      if (other == org.tensorflow.framework.GraphTransferNodeInputInfo.getDefaultInstance()) return this;
      if (other.getNodeId() != 0) {
        setNodeId(other.getNodeId());
      }
      if (nodeInputBuilder_ == null) {
        if (!other.nodeInput_.isEmpty()) {
          if (nodeInput_.isEmpty()) {
            nodeInput_ = other.nodeInput_;
            bitField0_ = (bitField0_ & ~0x00000002);
          } else {
            ensureNodeInputIsMutable();
            nodeInput_.addAll(other.nodeInput_);
          }
          onChanged();
        }
      } else {
        if (!other.nodeInput_.isEmpty()) {
          if (nodeInputBuilder_.isEmpty()) {
            nodeInputBuilder_.dispose();
            nodeInputBuilder_ = null;
            nodeInput_ = other.nodeInput_;
            bitField0_ = (bitField0_ & ~0x00000002);
            nodeInputBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getNodeInputFieldBuilder() : null;
          } else {
            nodeInputBuilder_.addAllMessages(other.nodeInput_);
          }
        }
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
      org.tensorflow.framework.GraphTransferNodeInputInfo parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.framework.GraphTransferNodeInputInfo) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private int nodeId_ ;
    /**
     * <code>int32 node_id = 1;</code>
     */
    public int getNodeId() {
      return nodeId_;
    }
    /**
     * <code>int32 node_id = 1;</code>
     */
    public Builder setNodeId(int value) {
      
      nodeId_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 node_id = 1;</code>
     */
    public Builder clearNodeId() {
      
      nodeId_ = 0;
      onChanged();
      return this;
    }

    private java.util.List<org.tensorflow.framework.GraphTransferNodeInput> nodeInput_ =
      java.util.Collections.emptyList();
    private void ensureNodeInputIsMutable() {
      if (!((bitField0_ & 0x00000002) == 0x00000002)) {
        nodeInput_ = new java.util.ArrayList<org.tensorflow.framework.GraphTransferNodeInput>(nodeInput_);
        bitField0_ |= 0x00000002;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.tensorflow.framework.GraphTransferNodeInput, org.tensorflow.framework.GraphTransferNodeInput.Builder, org.tensorflow.framework.GraphTransferNodeInputOrBuilder> nodeInputBuilder_;

    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public java.util.List<org.tensorflow.framework.GraphTransferNodeInput> getNodeInputList() {
      if (nodeInputBuilder_ == null) {
        return java.util.Collections.unmodifiableList(nodeInput_);
      } else {
        return nodeInputBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public int getNodeInputCount() {
      if (nodeInputBuilder_ == null) {
        return nodeInput_.size();
      } else {
        return nodeInputBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public org.tensorflow.framework.GraphTransferNodeInput getNodeInput(int index) {
      if (nodeInputBuilder_ == null) {
        return nodeInput_.get(index);
      } else {
        return nodeInputBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder setNodeInput(
        int index, org.tensorflow.framework.GraphTransferNodeInput value) {
      if (nodeInputBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureNodeInputIsMutable();
        nodeInput_.set(index, value);
        onChanged();
      } else {
        nodeInputBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder setNodeInput(
        int index, org.tensorflow.framework.GraphTransferNodeInput.Builder builderForValue) {
      if (nodeInputBuilder_ == null) {
        ensureNodeInputIsMutable();
        nodeInput_.set(index, builderForValue.build());
        onChanged();
      } else {
        nodeInputBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder addNodeInput(org.tensorflow.framework.GraphTransferNodeInput value) {
      if (nodeInputBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureNodeInputIsMutable();
        nodeInput_.add(value);
        onChanged();
      } else {
        nodeInputBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder addNodeInput(
        int index, org.tensorflow.framework.GraphTransferNodeInput value) {
      if (nodeInputBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureNodeInputIsMutable();
        nodeInput_.add(index, value);
        onChanged();
      } else {
        nodeInputBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder addNodeInput(
        org.tensorflow.framework.GraphTransferNodeInput.Builder builderForValue) {
      if (nodeInputBuilder_ == null) {
        ensureNodeInputIsMutable();
        nodeInput_.add(builderForValue.build());
        onChanged();
      } else {
        nodeInputBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder addNodeInput(
        int index, org.tensorflow.framework.GraphTransferNodeInput.Builder builderForValue) {
      if (nodeInputBuilder_ == null) {
        ensureNodeInputIsMutable();
        nodeInput_.add(index, builderForValue.build());
        onChanged();
      } else {
        nodeInputBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder addAllNodeInput(
        java.lang.Iterable<? extends org.tensorflow.framework.GraphTransferNodeInput> values) {
      if (nodeInputBuilder_ == null) {
        ensureNodeInputIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, nodeInput_);
        onChanged();
      } else {
        nodeInputBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder clearNodeInput() {
      if (nodeInputBuilder_ == null) {
        nodeInput_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000002);
        onChanged();
      } else {
        nodeInputBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public Builder removeNodeInput(int index) {
      if (nodeInputBuilder_ == null) {
        ensureNodeInputIsMutable();
        nodeInput_.remove(index);
        onChanged();
      } else {
        nodeInputBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public org.tensorflow.framework.GraphTransferNodeInput.Builder getNodeInputBuilder(
        int index) {
      return getNodeInputFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public org.tensorflow.framework.GraphTransferNodeInputOrBuilder getNodeInputOrBuilder(
        int index) {
      if (nodeInputBuilder_ == null) {
        return nodeInput_.get(index);  } else {
        return nodeInputBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public java.util.List<? extends org.tensorflow.framework.GraphTransferNodeInputOrBuilder> 
         getNodeInputOrBuilderList() {
      if (nodeInputBuilder_ != null) {
        return nodeInputBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(nodeInput_);
      }
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public org.tensorflow.framework.GraphTransferNodeInput.Builder addNodeInputBuilder() {
      return getNodeInputFieldBuilder().addBuilder(
          org.tensorflow.framework.GraphTransferNodeInput.getDefaultInstance());
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public org.tensorflow.framework.GraphTransferNodeInput.Builder addNodeInputBuilder(
        int index) {
      return getNodeInputFieldBuilder().addBuilder(
          index, org.tensorflow.framework.GraphTransferNodeInput.getDefaultInstance());
    }
    /**
     * <code>repeated .tensorflow.GraphTransferNodeInput node_input = 2;</code>
     */
    public java.util.List<org.tensorflow.framework.GraphTransferNodeInput.Builder> 
         getNodeInputBuilderList() {
      return getNodeInputFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.tensorflow.framework.GraphTransferNodeInput, org.tensorflow.framework.GraphTransferNodeInput.Builder, org.tensorflow.framework.GraphTransferNodeInputOrBuilder> 
        getNodeInputFieldBuilder() {
      if (nodeInputBuilder_ == null) {
        nodeInputBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.tensorflow.framework.GraphTransferNodeInput, org.tensorflow.framework.GraphTransferNodeInput.Builder, org.tensorflow.framework.GraphTransferNodeInputOrBuilder>(
                nodeInput_,
                ((bitField0_ & 0x00000002) == 0x00000002),
                getParentForChildren(),
                isClean());
        nodeInput_ = null;
      }
      return nodeInputBuilder_;
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


    // @@protoc_insertion_point(builder_scope:tensorflow.GraphTransferNodeInputInfo)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.GraphTransferNodeInputInfo)
  private static final org.tensorflow.framework.GraphTransferNodeInputInfo DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.framework.GraphTransferNodeInputInfo();
  }

  public static org.tensorflow.framework.GraphTransferNodeInputInfo getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<GraphTransferNodeInputInfo>
      PARSER = new com.google.protobuf.AbstractParser<GraphTransferNodeInputInfo>() {
    @java.lang.Override
    public GraphTransferNodeInputInfo parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new GraphTransferNodeInputInfo(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<GraphTransferNodeInputInfo> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<GraphTransferNodeInputInfo> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tensorflow.framework.GraphTransferNodeInputInfo getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
