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

import java.nio.ByteBuffer;
import java.util.*;
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

  public Map<String, GraphNode> getNodes() {
    return this.nodeMap.entrySet().stream().collect(Collectors.toMap(x->x.getKey(), x->getChild(x.getKey())));
  }

  public List<String> getRootNodes() {
    return getNodes().entrySet().stream()
        .filter(potentialRoot->!getNodes().values().stream().filter(node->node.getInputKeys().contains(potentialRoot.getKey())).findAny().isPresent())
        .map(x->x.getKey())
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

    public Object getDataSummary() {
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
      TensorProto tensor = getTensor();
      return null==tensor?null:GraphModel.getInts(tensor.getTensorContent().asReadOnlyByteBuffer());
    }

    @JsonIgnore
    public TensorProto getTensor() {
      Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
      return !attrMap.containsKey("value") ? null : attrMap.get("value").getTensor();
    }

    public long[] getShape() {
      Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
      TensorShapeProto shape;
      if (attrMap.containsKey("shape")) {

        shape = attrMap.get("shape").getShape();
      } else if (attrMap.containsKey("value")) {
        shape = attrMap.get("value").getTensor().getTensorShape();
      } else {
        return null;
      }
      return shape.getDimList().stream().mapToLong(x->x.getSize()).toArray();
    }

    public List<String> getInputKeys() {
      return getInputs().stream().map(x->x.name).collect(Collectors.toList());
    }

    @JsonIgnore
    public List<GraphNode> getInputs() {
      return this.nodeDef.getInputList().stream().map(x->getChild(x)).collect(Collectors.toList());
    }

    private volatile Integer order = null;
    public int getOrder() {
      if(null == order) {
        synchronized (this) {
          if(null == order) {
            order = getInputs().stream().mapToInt(x -> x.getOrder()).max().orElseGet(() -> -1) + 1;
          }
        }
      }
      return order;
    }

    public List<String> getRootKeys() {
      return getRootInputs().stream().map(x->x.name).collect(Collectors.toList());
    }

    private volatile List<GraphNode> rootInputs = null;
    @JsonIgnore
    public List<GraphNode> getRootInputs() {
      if(null == rootInputs) {
        synchronized(this) {
          if(null == rootInputs) {
            if(getInputs().isEmpty()) {
              if(op.equals("Placeholder")) {
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

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      GraphNode graphNode = (GraphNode) o;
      return Objects.equals(name, graphNode.name);
    }

    @Override
    public int hashCode() {

      return Objects.hash(name);
    }
  }

  public static double[] toDoubles(float[] values) {
    return null==values?null:Floats.asList(values).stream().mapToDouble(x -> x).toArray();
  }
  public static double[] toDoubles(int[] values) {
    return null==values?null:Arrays.stream(values).mapToDouble(x -> x).toArray();
  }

  public static HashMap<String, Double> summary(double[] values) {
    if(null == values) return null;
    DoubleSummaryStatistics finiteStatistics = Arrays.stream(values).filter(x -> Double.isFinite(x)).summaryStatistics();
    long nanCount = Arrays.stream(values).filter(x -> Double.isNaN(x)).count();
    long infCount = Arrays.stream(values).filter(x -> Double.isInfinite(x)).count();
    HashMap<String, Double> data = new HashMap<>();
    data.put("min",finiteStatistics.getMin());
    data.put("max",finiteStatistics.getMax());
    data.put("sum",finiteStatistics.getSum());
    data.put("avg",finiteStatistics.getAverage());
    data.put("finiteCount", (double) finiteStatistics.getCount());
    data.put("nanCount", (double) nanCount);
    data.put("infCount", (double) infCount);
    return data;
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
