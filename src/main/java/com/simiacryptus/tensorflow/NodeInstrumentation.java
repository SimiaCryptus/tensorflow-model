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

import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.simiacryptus.tensorflow.TensorflowUtil.buildTensorShape;

public class NodeInstrumentation {
  private DataType type;
  private boolean scalar = true;
  private int[] image = null;

  public NodeInstrumentation(DataType type) {
    this.setType(type);
  }

  @NotNull
  public static GraphDef instrument(GraphDef graphDef, String summaryOutput, Function<NodeDef, NodeInstrumentation> config) {
    TensorflowUtil.validate(graphDef);
    graphDef = TensorflowUtil.editGraph(graphDef, graphBuilder -> {
      ArrayList<NodeDef> nodeDefs = new ArrayList<>();
      TensorflowUtil.editNodes(graphBuilder, node -> {
        NodeInstrumentation nodeInstrumentation = config.apply(node);
        if (null != nodeInstrumentation) {
          nodeDefs.addAll(nodeInstrumentation.instrument(graphBuilder, node));
        }
        return node;
      });
      if (!nodeDefs.isEmpty()) {
        graphBuilder.addAllNode(nodeDefs);
        graphBuilder.addNode(NodeDef.newBuilder()
            .setName(summaryOutput)
            .setOp("MergeSummary")
            .addAllInput(nodeDefs.stream().map(NodeDef::getName).collect(Collectors.toList()))
            .putAttr("N", AttrValue.newBuilder().setI(nodeDefs.size()).build())
            .build());
      }
      ;
      return graphBuilder;
    });
    TensorflowUtil.validate(graphDef);
    return graphDef;
  }

  public static DataType getDataType(NodeDef node, DataType dataType) {
    if (node.getAttrMap().containsKey("value")) {
      TensorProto tensor = node.getAttrOrThrow("value").getTensor();
      dataType = tensor.getDtype();
    }
    if (node.getAttrMap().containsKey("dtype")) {
      dataType = node.getAttrOrThrow("dtype").getType();
    }
    return dataType;
  }

  public ArrayList<NodeDef> instrument(GraphDef.Builder graphBuilder, NodeDef node) {
    ArrayList<NodeDef> nodeDefs = new ArrayList<>();
    String label = node.getName();

    String asFloatNode;
    if (getType() == DataType.DT_FLOAT) {
      asFloatNode = label;
    } else {
      asFloatNode = label + "/cast";
      graphBuilder.addNode(NodeDef.newBuilder()
          .addInput(label)
          .setName(asFloatNode)
          .putAttr("SrcT", AttrValue.newBuilder().setType(getType()).build())
          .putAttr("DstT", AttrValue.newBuilder().setType(DataType.DT_FLOAT).build())
          .setOp("Cast").build());
    }

    if (isScalar()) {

      String rankNode = label + "/summary/rank";
      graphBuilder.addAllNode(TensorflowUtil.rankNode(node, type, rankNode));
      nodeDefs.add(NodeDef.newBuilder()
          .addInput(asFloatNode)
          .setName(label + "/summary/TensorSummary")
          .putAttr("T", AttrValue.newBuilder().setType(DataType.DT_FLOAT).build())
          .setOp("TensorSummary").build());

      for (String summaryOp : Arrays.asList("Mean", "Min", "Max")) {
        String summaryName = label + "/summary/" + summaryOp;
        graphBuilder.addNode(NodeDef.newBuilder()
            .addInput(asFloatNode)
            .addInput(rankNode)
            .setName(summaryName)
            .putAttr("keep_dims", AttrValue.newBuilder().setB(false).build())
            .putAttr("T", AttrValue.newBuilder().setType(DataType.DT_FLOAT).build())
            .putAttr("Tidx", AttrValue.newBuilder().setType(DataType.DT_INT32).build())
            .setOp(summaryOp)
            .build());

        nodeDefs.add(NodeDef.newBuilder()
            .addInput(TensorflowUtil.addConst(graphBuilder,
                label + "/summary/" + summaryOp + "/Label",
                buildTensorShape(),
                summaryName))
            .addInput(summaryName)
            .setName(summaryName + "/ScalarSummary")
            .putAttr("T", AttrValue.newBuilder().setType(DataType.DT_FLOAT).build())
            .setOp("ScalarSummary")
            .build());
      }

      nodeDefs.add(NodeDef.newBuilder()
          .addInput(TensorflowUtil.addConst(graphBuilder,
              label + "/summary/HistogramSummary/Label",
              buildTensorShape(),
              label))
          .addInput(asFloatNode)
          .setName(label + "/summary/HistogramSummary")
          .setOp("HistogramSummary").build());
    }

    if (isImage() != null) {
      String shapeName = label + "/summary/ImageSummary/Shape";
      graphBuilder.addNode(TensorflowUtil.newConst(
          shapeName,
          AttrValue.newBuilder().setTensor(TensorflowUtil.buildTensor(buildTensorShape(4), -1, this.image[0], this.image[1], this.image[2])).build(),
          DataType.DT_INT32));
      graphBuilder.addNode(NodeDef.newBuilder()
          .addInput(asFloatNode)
          .addInput(shapeName)
          .setName(label + "/summary/ImageSummary/Reshape")
          .putAttr("T", AttrValue.newBuilder().setType(DataType.DT_FLOAT).build())
          .setOp("Reshape").build());
      nodeDefs.add(NodeDef.newBuilder()
          .addInput(TensorflowUtil.addConst(graphBuilder,
              label + "/summary/ImageSummary/Label",
              buildTensorShape(),
              label))
          .addInput(label + "/summary/ImageSummary/Reshape")
          .setName(label + "/summary/ImageSummary")
          .setOp("ImageSummary").build());
    }

    return nodeDefs;
  }

  public DataType getType() {
    return type;
  }

  public NodeInstrumentation setType(DataType type) {
    this.type = type;
    return this;
  }

  public boolean isScalar() {
    return scalar;
  }

  public NodeInstrumentation setScalar(boolean scalar) {
    this.scalar = scalar;
    return this;
  }

  @Override
  public String toString() {
    return "NodeInstrumentation{" +
        ", type=" + type +
        ", scalar=" + scalar +
        '}';
  }

  public int[] isImage() {
    return image;
  }

  public NodeInstrumentation setImage(int... image) {
    assert image.length == 3 : image.length;
    assert Arrays.asList(1,3,4).contains(image[2]) : image[2];
    this.image = image;
    return this;
  }
}
