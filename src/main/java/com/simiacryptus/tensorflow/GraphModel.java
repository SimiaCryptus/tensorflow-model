package com.simiacryptus.tensorflow;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.google.common.primitives.Floats;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.framework.*;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class GraphModel {
  @JsonIgnore
  public final GraphDef graphDef;
  @JsonIgnore
  public final Map<String, NodeDef> nodeMap;

  public GraphModel(byte[] bytes) {
    this.graphDef = parseGraph(bytes);
    this.nodeMap = graphDef.getNodeList().stream().collect(Collectors.toMap(x -> x.getName(), x -> x));
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

  public class GraphNode {
    public final String name;
    @JsonIgnore
    public final NodeDef nodeDef;
    public final String op;

    protected GraphNode(String name) {
      this.name = name;
      this.nodeDef = nodeMap.get(this.name);
      assert name.equals(nodeDef.getName());
      this.op = nodeDef.getOp();
      Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
    }

    @Override
    public String toString() {
      return "GraphNode{" +
          "name='" + name + '\'' +
          ", op=" + op +
          ", dataType=" + getDataType() +
          ", shape=" + getShape() +
          ", inputs=" + getInputs() +
          '}';
    }

    public DataType getDataType() {
      Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
      if (attrMap.containsKey("dtype")) {
        return attrMap.get("dtype").getType();
      } else if (attrMap.containsKey("value")) {
        return attrMap.get("value").getTensor().getDtype();
      } else {
        return null;
      }
    }

    public String getDataSummary() {
      return summary(getData());
    }

    @JsonIgnore
    public double[] getData() {
      if (getDataType() == DataType.DT_FLOAT) {
        return toDoubles(getFloats());
      } else if (getDataType() == DataType.DT_INT32) {
        return toDoubles(getInts());
      } else if (getDataType() == null) {
        return null;
      } else {
        throw new RuntimeException("Unsupported Data Type: " + getDataType());
      }
    }

    @JsonIgnore
    public float[] getFloats() {
      TensorProto tensor = getTensor();
      return null==tensor?null:GraphModel.getFloats(tensor.getTensorContent().asReadOnlyByteBuffer());
    }

    @JsonIgnore
    public int[] getInts() {
      return GraphModel.getInts(getTensor().getTensorContent().asReadOnlyByteBuffer());
    }

    @JsonIgnore
    public TensorProto getTensor() {
      Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
      return !attrMap.containsKey("value") ? null : attrMap.get("value").getTensor();
    }

    @JsonIgnore
    public TensorShapeProto getShape() {
      Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
      if (attrMap.containsKey("shape")) {
        return attrMap.get("shape").getShape();
      } else if (attrMap.containsKey("value")) {
        return attrMap.get("value").getTensor().getTensorShape();
      } else {
        return null;
      }
    }

    public List<GraphNode> getInputs() {
      return this.nodeDef.getInputList().stream().map(x->getChild(x)).collect(Collectors.toList());
    }
  }

  public static double[] toDoubles(float[] values) {
    return null==values?null:Floats.asList(values).stream().mapToDouble(x -> x).toArray();
  }
  public static double[] toDoubles(int[] values) {
    return null==values?null:Arrays.stream(values).mapToDouble(x -> x).toArray();
  }

  public static String summary(double[] values) {
    if(null == values) return "";
    DoubleSummaryStatistics finiteStatistics = Arrays.stream(values).filter(x -> Double.isFinite(x)).summaryStatistics();
    long nanCount = Arrays.stream(values).filter(x -> Double.isNaN(x)).count();
    long infCount = Arrays.stream(values).filter(x -> Double.isInfinite(x)).count();
    return String.format("Finite=%s; NaN=%s; Inf=%s", finiteStatistics, nanCount, infCount);
  }

  private final ConcurrentHashMap<String, GraphNode> nodeCache = new ConcurrentHashMap<>();

  public GraphNode getChild(String x) {
    return nodeCache.computeIfAbsent(x, y -> new GraphNode(y));
  }

  private static int[] getInts(ByteBuffer byteBuffer) {
    int[] values = new int[byteBuffer.limit() / 4];
    byteBuffer.asIntBuffer().get(values);
    return values;
  }
  private static float[] getFloats(ByteBuffer byteBuffer) {
    float[] values = new float[byteBuffer.limit() / 4];
    byte in[] = {0, 0, 0, 0};
    for (int i = 0; i < values.length; i++) {
      byteBuffer.get(in);
      values[i] = Float.intBitsToFloat(in[0] | in[1] << 8 | in[2] << 16 | in[3] << 24);
    }
    return values;
  }
}
