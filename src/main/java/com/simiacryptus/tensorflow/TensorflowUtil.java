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

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import org.tensorflow.*;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.*;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TensorflowUtil {
  private static final SumFn doubleSum = new SumFn(Double.class);
  private static final SumFn floatSum = new SumFn(Float.class);

  @Nullable
  public static Operation find(@Nonnull Graph graph, String name) {
    Iterator<Operation> operations = graph.operations();
    while (operations.hasNext()) {
      Operation operation = operations.next();
      if (operation.name().equals(name)) {
        return operation;
      }
    }
    return null;
  }

  public static byte[] makeGraph(@Nonnull Consumer<Ops> builder) {
    try (Graph graph = new Graph()) {
      builder.accept(Ops.create(graph));
      byte[] bytes = graph.toGraphDef();
      try {
        validate(GraphDef.parseFrom(bytes));
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
      return bytes;
    }
  }

  public static void validate(@Nonnull GraphDef graphDef) {
    List<String> names = graphDef.getNodeList().stream().map(x -> x.getName()).collect(Collectors.toList());
    graphDef.getNodeList().stream().map(x -> x.getName()).distinct().forEach(names::remove);
    if (!names.isEmpty()) {
      throw new IllegalStateException("Duplicate names: " + RefUtil.get(names.stream().reduce((a, b) -> a + ", " + b)));
    }
  }

  @Nonnull
  public static String addConst(@Nonnull GraphDef.Builder graphBuilder, @Nonnull String name, TensorShapeProto shape, String... label) {
    graphBuilder.addNode(newConst(name, shape, label));
    return name;
  }

  @Nonnull
  public static NodeDef newConst(@Nonnull String name, TensorShapeProto shape, String[] label) {
    return newConst(name, getTensorAttr(shape, label), DataType.DT_STRING);
  }

  @Nonnull
  public static AttrValue getTensorAttr(TensorShapeProto shape, String... label) {
    return AttrValue.newBuilder().setTensor(buildTensor(shape, label)).build();
  }

  @Nonnull
  public static NodeDef newConst(@Nonnull String name, @Nonnull AttrValue attrValue, @Nonnull DataType dtString) {
    return NodeDef.newBuilder()
        .setName(name)
        .setOp("Const")
        .putAttr("dtype", AttrValue.newBuilder().setType(dtString).build())
        .putAttr("value", attrValue)
        .build();
  }

  @Nonnull
  public static GraphDef editGraph(@Nonnull GraphDef graph, @Nonnull @RefAware Function<GraphDef.Builder, GraphDef.Builder> edit) {
    GraphDef build = edit.apply(graph.toBuilder()).build();
    RefUtil.freeRef(build);
    return build;
  }

  public static void editNode(@Nonnull GraphDef.Builder graphBuilder, String name, @Nonnull @RefAware Function<NodeDef.Builder, NodeDef.Builder> edit) {
    List<NodeDef> nodeList = graphBuilder.getNodeList();
    NodeDef nodeDef = nodeList.stream().filter(x -> x.getName().equals(name)).findAny()
        .orElseGet(() -> {
          throw new NoSuchElementException(String.format(
              "%s not found in %s",
              name,
              RefUtil.get(nodeList.stream().map(NodeDef::getName).reduce((a, b) -> a + "," + b))
          ));
        });
    int index = nodeList.indexOf(nodeDef);
    graphBuilder.removeNode(index);
    graphBuilder.addNode(edit.apply(nodeDef.toBuilder()).build());
    RefUtil.freeRef(edit);
  }

  public static void editNodes(@Nonnull GraphDef.Builder graphBuilder, @Nonnull Function<NodeDef, NodeDef> edit) {
    new ArrayList<>(graphBuilder.getNodeList()).stream().forEach(previousValue -> {
      NodeDef newValue = edit.apply(previousValue);
      if (newValue != previousValue) {
        graphBuilder.removeNode(graphBuilder.getNodeList().indexOf(previousValue));
        graphBuilder.addNode(newValue);
      }
    });
  }

  @Nonnull
  public static <T extends Number> Tensor<T> add(@Nonnull Tensor<T>... tensors) {
    return add(Arrays.stream(tensors));
  }

  @Nonnull
  public static TensorProto buildTensor(TensorShapeProto tensorShapeProto, @Nonnull int... vs) {
    TensorProto.Builder tensor = TensorProto.newBuilder()
        .setDtype(DataType.DT_INT32)
        .setTensorShape(tensorShapeProto);
    for (int l : vs) {
      tensor.addIntVal(l);
    }
    return tensor.build();
  }

  @Nonnull
  public static TensorProto buildTensor(TensorShapeProto tensorShapeProto, @Nonnull String... vs) {
    TensorProto.Builder tensor = TensorProto.newBuilder()
        .setDtype(DataType.DT_STRING)
        .setTensorShape(tensorShapeProto);
    for (String l : vs) {
      tensor.addStringVal(ByteString.copyFromUtf8(l));
    }
    return tensor.build();
  }

  @Nonnull
  public static TensorShapeProto buildTensorShape(@Nonnull long... dims) {
    TensorShapeProto.Builder builder = TensorShapeProto.newBuilder();
    for (long l : dims) {
      builder.addDim(TensorShapeProto.Dim.newBuilder().setSize(l).build());
    }
    return builder.build();
  }

  @Nonnull
  public static <T extends Number> Tensor<T> add(@Nonnull Stream<Tensor<T>> stream) {
    return RefUtil.get(stream.reduce((a, b) -> {
      if (a.dataType() == org.tensorflow.DataType.DOUBLE) {
        Tensor<T> tensor = doubleSum.add(a.expect(Double.class), b.expect(Double.class));
        a.close();
        b.close();
        return tensor;
      } else {
        Tensor<T> tensor = floatSum.add(a.expect(Float.class), b.expect(Float.class));
        a.close();
        b.close();
        return tensor;
      }
    }));
  }

  @Nonnull
  public static List<NodeDef> rankNode(@Nonnull NodeDef node, @Nonnull DataType type, @Nonnull String rankNode) {
    String endNode = rankNode + "/end";
    String startNode = rankNode + "/start";
    String stepNode = rankNode + "/step";
    return Arrays.asList(
        newConst(startNode, AttrValue.newBuilder().setTensor(
            buildTensor(TensorflowUtil.buildTensorShape(), 0)
        ).build(), DataType.DT_INT32),
        newConst(stepNode, AttrValue.newBuilder().setTensor(
            buildTensor(TensorflowUtil.buildTensorShape(), 1)
        ).build(), DataType.DT_INT32),
        NodeDef.newBuilder()
            .setName(endNode)
            .addInput(node.getName())
            .setOp("Rank")
            .putAttr("T", AttrValue.newBuilder().setType(type).build())
            .build(),
        NodeDef.newBuilder()
            .addInput(startNode)
            .addInput(endNode)
            .addInput(stepNode)
            .setName(rankNode)
            .setOp("Range")
            .build()
    );
  }

  public static class SumFn<T extends Number> {
    @Nonnull
    private final Graph sumGraph;
    @Nonnull
    private final Session sumSession;
    private final Output<T> in1;
    private final Output<T> in2;
    private final Output<T> out;

    public SumFn(Class<T> dtype) {
      sumGraph = new Graph();
      Ops ops = Ops.create(sumGraph);
      in1 = ops.placeholder(dtype).asOutput();
      in2 = ops.placeholder(dtype).asOutput();
      out = ops.add(
          in1,
          in2
      ).asOutput();
      sumSession = new Session(sumGraph);
    }

    @Nonnull
    public Tensor<Double> add(Tensor<T> a, Tensor<T> b) {
      return sumSession.runner().feed(in1, a).feed(in2, b).fetch(out).run().get(0).expect(Double.class);
    }

    public void close() {
      sumSession.close();
      sumGraph.close();
    }
  }
}
