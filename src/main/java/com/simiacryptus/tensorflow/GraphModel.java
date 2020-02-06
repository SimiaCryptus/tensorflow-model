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

package com.simiacryptus.tensorflow;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.google.common.primitives.Floats;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.framework.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GraphModel {
  @JsonIgnore
  public final GraphDef graphDef;
  @JsonIgnore
  public final Map<String, NodeDef> nodeMap;
  private final ConcurrentHashMap<String, GraphNode> nodeCache = new ConcurrentHashMap<>();

  public GraphModel(byte[] bytes) {
    this.graphDef = parseGraph(bytes);
    this.nodeMap = graphDef.getNodeList().stream().collect(Collectors.toMap(x -> x.getName(), x -> x));
  }

  public Map<String, GraphNode> getNodes() {
    return this.nodeMap.entrySet().stream().collect(Collectors.toMap(x -> x.getKey(), x -> getChild(x.getKey())));
  }

  public List<String> getRootNodes() {
    return getNodes().entrySet().stream()
        .filter(potentialRoot -> !getNodes().values().stream().filter(node -> node.getInputKeys().contains(potentialRoot.getKey())).findAny().isPresent())
        .map(x -> x.getKey())
        .collect(Collectors.toList());
  }

  public static GraphDef parseGraph(byte[] bytes) {
    final GraphDef graphDef;
    try {
      graphDef = GraphDef.parseFrom(bytes);
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
    return graphDef;
  }

  @Nullable
  public static double[] toDoubles(@Nullable float[] values) {
    return null == values ? null : Floats.asList(values).stream().mapToDouble(x -> x).toArray();
  }

  @Nullable
  public static double[] toDoubles(@Nullable int[] values) {
    return null == values ? null : Arrays.stream(values).mapToDouble(x -> x).toArray();
  }

  @Nullable
  public static double[] toDoubles(@Nullable long[] values) {
    return null == values ? null : Arrays.stream(values).mapToDouble(x -> x).toArray();
  }

  @Nullable
  public static HashMap<String, Double> summary(@Nullable double[] values) {
    if (null == values) return null;
    DoubleSummaryStatistics finiteStatistics = Arrays.stream(values).filter(x -> Double.isFinite(x)).summaryStatistics();
    long nanCount = Arrays.stream(values).filter(x -> Double.isNaN(x)).count();
    long infCount = Arrays.stream(values).filter(x -> Double.isInfinite(x)).count();
    HashMap<String, Double> data = new HashMap<>();
    data.put("min", finiteStatistics.getMin());
    data.put("max", finiteStatistics.getMax());
    data.put("sum", finiteStatistics.getSum());
    data.put("avg", finiteStatistics.getAverage());
    data.put("finiteCount", (double) finiteStatistics.getCount());
    data.put("nanCount", (double) nanCount);
    data.put("infCount", (double) infCount);
    return data;
  }

  @Nonnull
  public static int[] getInts(@Nonnull ByteBuffer byteBuffer) {
    int[] values = new int[byteBuffer.limit() / 4];
    byte in[] = {0, 0, 0, 0};
    for (int i = 0; i < values.length; i++) {
      byteBuffer.get(in);
      int value = 0;
      for (int b = 0; b < 4; b++) {
        value |= (in[b] & 0xFFL) << 8 * b;
      }
      values[i] = value;
    }
    return values;
  }

  @Nonnull
  public static long[] getLongs(@Nonnull ByteBuffer byteBuffer) {
    long[] values = new long[byteBuffer.limit() / 8];
    byte in[] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < values.length; i++) {
      byteBuffer.get(in);
      long value = 0;
      for (int b = 0; b < 4; b++) {
        value |= (in[b] & 0xFFL) << 8 * b;
      }
      values[i] = value;
    }
    return values;
//    long[] values = new long[byteBuffer.limit() / 8];
//    byteBuffer.asLongBuffer().get(values);
//    return values;
  }

  @Nonnull
  public static ByteBuffer putDoubles(@Nonnull double[] values) {
    return (ByteBuffer) putDoubles(ByteBuffer.allocate(values.length * 8), values).flip();
  }

  @Nonnull
  public static ByteBuffer putDoubles(@Nonnull ByteBuffer byteBuffer, @Nonnull double[] values) {
    byte in[] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < values.length; i++) {
      long value = Double.doubleToLongBits(values[i]);
      for (int b = 0; b < 8; b++) {
        in[b] = (byte) (value >> 8 * b & 0xFFL);
      }
      byteBuffer.put(in);
    }
    return byteBuffer;
  }

  @Nonnull
  public static double[] getDoubles(@Nonnull ByteBuffer byteBuffer) {
    double[] values = new double[byteBuffer.limit() / 8];
    byte in[] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < values.length; i++) {
      byteBuffer.get(in);
      long value = 0;
      for (int b = 0; b < 8; b++) {
        value |= (in[b] & 0xFFL) << 8 * b;
      }
      values[i] = Double.longBitsToDouble(value);
    }
    return values;
  }

  @Nonnull
  public static ByteBuffer putFloats(@Nonnull float[] values) {
    return (ByteBuffer) putFloats(ByteBuffer.allocate(values.length * 8), values).flip();
  }

  @Nonnull
  public static ByteBuffer putFloats(@Nonnull ByteBuffer byteBuffer, @Nonnull float[] values) {
    byte in[] = {0, 0, 0, 0};
    for (int i = 0; i < values.length; i++) {
      int value = Float.floatToIntBits(values[i]);
      for (int b = 0; b < 4; b++) {
        in[b] = (byte) (value >> 8 * b & 0xFFL);
      }
      byteBuffer.put(in);
    }
    return byteBuffer;
  }

  @Nonnull
  public static float[] getFloats(@Nonnull ByteBuffer byteBuffer) {
    float[] values = new float[byteBuffer.limit() / 4];
    byte in[] = {0, 0, 0, 0};
    for (int i = 0; i < values.length; i++) {
      byteBuffer.get(in);
      int value = 0;
      for (int b = 0; b < 4; b++) {
        value |= (in[b] & 0xFF) << 8 * b;
      }
      values[i] = Float.intBitsToFloat(value);
    }
    return values;
  }

  public Map<String, DeltaRecord> compare(@Nonnull GraphModel other) {
    final Map<String, GraphNode> myNodes = getNodes();
    final Map<String, GraphNode> yourNodes = other.getNodes();
    return Stream.concat(
        myNodes.keySet().stream(),
        yourNodes.keySet().stream()
    ).distinct().flatMap(name -> {
      final GraphNode l = myNodes.get(name);
      final GraphNode r = yourNodes.get(name);
      boolean equals = true;
      if (null == l || null == r) return Stream.of(new DeltaRecord(name, l, r));
      equals = l.getProperties().equals(r.getProperties());
      equals &= Arrays.equals(l.getShape(), r.getShape());
      equals &= Arrays.equals(l.getData(), r.getData());
      equals &= l.getInputKeys().equals(r.getInputKeys());
      equals &= l.getInputs().stream().map(x -> x.name).collect(Collectors.toSet()).equals(
          r.getInputs().stream().map(x -> x.name).collect(Collectors.toSet())
      );
      return equals ? Stream.empty() : Stream.of(new DeltaRecord(name, l, r));
    }).collect(Collectors.toMap(x -> x.name, x -> x));
  }

  public GraphNode getChild(@Nonnull String x) {
    return nodeCache.computeIfAbsent(x, y -> new GraphNode(y));
  }

  public static class DeltaRecord {
    public final String name;
    public final GraphNode left;
    public final GraphNode right;

    public DeltaRecord(String name, GraphNode left, GraphNode right) {
      this.name = name;
      this.left = left;
      this.right = right;
    }
  }

  public class GraphNode {
    public final String name;
    @Nullable
    @JsonIgnore
    private transient NodeDef nodeDef = null;
    @Nullable
    private transient String op = null;
    @Nullable
    private volatile Integer order = null;
    @Nullable
    private volatile List<GraphNode> rootInputs = null;

    protected GraphNode(String name) {
      this.name = name;
    }

    @Nullable
    @JsonIgnore
    public double[] getData() {
      DataType dataType = getDataType();
      if (dataType == DataType.DT_DOUBLE) {
        return getDoubles();
      } else if (dataType == DataType.DT_FLOAT) {
        return toDoubles(getFloats());
      } else if (dataType == DataType.DT_INT32) {
        return toDoubles(getInts());
      } else if (dataType == DataType.DT_INT64) {
        return toDoubles(getLongs());
      } else if (dataType == DataType.UNRECOGNIZED) {
        return null;
      } else {
        throw new RuntimeException("Unsupported Data Type: " + dataType);
      }
    }

    @Nullable
    public Object getDataSummary() {
      if (getDataType() == DataType.DT_STRING) {
        return "";
      } else {
        double[] data = getData();
        if (null == data || 0 == data.length) return "";
        if (32 > data.length) return data;
        return summary(data);
      }
    }

    @JsonIgnore
    @Nonnull
    public final DataType getDataType() {
      Map<String, AttrValue> attrMap = getNodeDef().getAttrMap();
      DataType type = null;
      if (attrMap.containsKey("dtype")) {
        type = attrMap.get("dtype").getType();
      }
      if (null == type && attrMap.containsKey("value")) {
        type = attrMap.get("value").getTensor().getDtype();
      }
      if (null == type) {
        type = DataType.UNRECOGNIZED;
      }
      return type;
    }

    @Nullable
    public String getDataTypeString() {
      DataType dataType = getDataType();
      if (dataType == DataType.UNRECOGNIZED) return null;
      return dataType.toString();
    }

    @Nullable
    @JsonIgnore
    public double[] getDoubles() {
      TensorProto tensor = getTensor();
      return null == tensor ? null : GraphModel.getDoubles(tensor.getTensorContent().asReadOnlyByteBuffer());
    }

    @Nullable
    @JsonIgnore
    public float[] getFloats() {
      TensorProto tensor = getTensor();
      return null == tensor ? null : GraphModel.getFloats(tensor.getTensorContent().asReadOnlyByteBuffer());
    }

    public List<String> getInputKeys() {
      return getInputs().stream().map(x -> x.name).collect(Collectors.toList());
    }

    @JsonIgnore
    public List<GraphNode> getInputs() {
      final NodeDef nodeDef = this.getNodeDef();
      if (null == nodeDef) return Collections.emptyList();
      return nodeDef.getInputList().stream().map(x -> getChild(x)).collect(Collectors.toList());
    }

    @Nullable
    @JsonIgnore
    public int[] getInts() {
      TensorProto tensor = getTensor();
      return null == tensor ? null : GraphModel.getInts(tensor.getTensorContent().asReadOnlyByteBuffer());
    }

    @Nullable
    @JsonIgnore
    public long[] getLongs() {
      TensorProto tensor = getTensor();
      return null == tensor ? null : GraphModel.getLongs(tensor.getTensorContent().asReadOnlyByteBuffer());
    }

    @Nullable
    @JsonIgnore
    public NodeDef getNodeDef() {
      if (null == this.nodeDef) {
        synchronized (this) {
          if (null == this.nodeDef) {
            String key = normalizeKey();
            this.nodeDef = nodeMap.get(key);
            if (null == this.nodeDef) {
              return null;
            }
          }
        }
      }
      return nodeDef;
    }

    @Nullable
    public String getOp() {
      if (null == this.op) {
        synchronized (this) {
          if (null == this.op) {
            final NodeDef nodeDef = getNodeDef();
            if (null == nodeDef) return "null";
            this.op = nodeDef.getOp();
          }
        }
      }
      return op;
    }

    public int getOrder() {
      if (null == order) {
        synchronized (this) {
          if (null == order) {
            order = getInputs().stream().mapToInt(x -> x.getOrder()).max().orElseGet(() -> -1) + 1;
          }
        }
      }
      return order;
    }

    public Map<String, String> getProperties() {
      Map<String, AttrValue> attrMap = getNodeDef().getAttrMap();
      return attrMap.entrySet().stream()
          .collect(Collectors.toMap(stringAttrValueEntry1 -> stringAttrValueEntry1.getKey(), stringAttrValueEntry -> {
            if (stringAttrValueEntry.getKey().equals("value")) {
              AttrValue value = stringAttrValueEntry.getValue();
              final String s = value.toBuilder().build().toString();
              return s.substring(0, Math.min(s.length(), 1024));
            } else {
              AttrValue value = stringAttrValueEntry.getValue();
              return value.toString();
            }
          }));
    }

    @Nullable
    @JsonIgnore
    public List<GraphNode> getRootInputs() {
      if (null == rootInputs) {
        synchronized (this) {
          if (null == rootInputs) {
            if (getInputs().isEmpty()) {
              if (getOp().equals("Placeholder")) {
                rootInputs = Arrays.asList(this);
              } else {
                rootInputs = Arrays.asList();
              }
            } else {
              rootInputs = getInputs().stream().flatMap(input -> input.getRootInputs().stream()).distinct().collect(Collectors.toList());
            }
          }
        }
      }
      return rootInputs;
    }

    public List<String> getRootKeys() {
      return getRootInputs().stream().map(x -> x.name).collect(Collectors.toList());
    }

    @Nullable
    public long[] getShape() {
      Map<String, AttrValue> attrMap = getNodeDef().getAttrMap();
      TensorShapeProto shape;
      if (attrMap.containsKey("shape")) {
        shape = attrMap.get("shape").getShape();
      } else if (attrMap.containsKey("value")) {
        shape = attrMap.get("value").getTensor().getTensorShape();
      } else {
        return null;
      }
      return shape.getDimList().stream().mapToLong(x -> x.getSize()).toArray();
    }

    @Nullable
    @JsonIgnore
    public TensorProto getTensor() {
      Map<String, AttrValue> attrMap = getNodeDef().getAttrMap();
      return !attrMap.containsKey("value") ? null : attrMap.get("value").getTensor();
    }

    @Nonnull
    @Override
    public String toString() {
      return "GraphNode{" +
          "name='" + name + '\'' +
          ", op=" + getOp() +
          ", dataType=" + getDataType() +
          ", shape=" + getShape() +
          ", inputs=" + getInputs() +
          '}';
    }

    @Nonnull
    public GraphDef subgraph(@Nonnull Set<String> inputs) {
      GraphDef.Builder builder = GraphDef.newBuilder();
      for (String input : inputs) {
        builder.addNode(NodeDef.newBuilder()
            .setName(input)
            .setOp("Placeholder")
            .putAttr("dtype", AttrValue.newBuilder().setType(DataType.DT_FLOAT).build())
            .build());
      }
      subgraphNodes(inputs).stream().map(graphNode -> graphNode.getNodeDef()).forEach(value -> builder.addNode(value));
      return builder.build();
    }

    @JsonIgnore
    public List<GraphNode> subgraphNodes(@Nonnull Set<String> inputs) {
      List<GraphNode> subgraph;
      if (inputs.contains(name)) {
        subgraph = Arrays.asList();
      } else if (getInputs().isEmpty()) {
        subgraph = Arrays.asList(this);
      } else {
        subgraph = Stream.concat(
            Stream.of(this),
            getInputs().stream().flatMap(input -> input.subgraphNodes(inputs).stream()).distinct()
        ).distinct().collect(Collectors.toList());
      }
      return subgraph;
    }

    @Override
    public boolean equals(@Nullable Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      GraphNode graphNode = (GraphNode) o;
      return Objects.equals(name, graphNode.name);
    }

    @Override
    public int hashCode() {

      return Objects.hash(name);
    }

    private String normalizeKey() {
      String name = this.name;
      name = name.split(":")[0];
      while (name.startsWith("^")) name = name.substring(1);
      return name;
    }
  }
}
