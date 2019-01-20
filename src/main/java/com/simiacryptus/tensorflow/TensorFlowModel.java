/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.simiacryptus.tensorflow;

import com.google.common.primitives.Floats;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.framework.*;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.DoubleSummaryStatistics;
import java.util.Map;

public class TensorFlowModel {

  public static void describeGraph(byte[] bytes) {
    System.out.println("Decoding GraphDef from " + bytes.length + " bytes");
    final GraphDef graphDef = parseGraph(bytes);
    for (int i = 0; i < graphDef.getNodeCount(); i++) {
      NodeDef nodeDef = graphDef.getNode(i);
      String name = nodeDef.getName();
      String op = nodeDef.getOp();
      Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
      int inputCount = nodeDef.getInputCount();
      System.out.println(takeN(String.format("Node %s/%s: %s (%s; properties %s) with %s inputs", i, graphDef.getNodeCount(), name, op, attrMap.keySet(), inputCount), 1024 * 16));
      if(attrMap.containsKey("value")) {
        final TensorProto tensor = attrMap.get("value").getTensor();
        System.out.println(takeN(String.format("  Value: Type=%s; Shape=%s; Size=%s", tensor.getDtype(), tensor.getTensorShape(), tensor.getTensorContent().size()), 1024));
        if(tensor.getDtype() == DataType.DT_FLOAT) {
          float[] floats = cvt(tensor.getTensorContent().asReadOnlyByteBuffer());
          DoubleSummaryStatistics finiteStatistics = Floats.asList(floats).stream().mapToDouble(x -> x).filter(x->Double.isFinite(x)).summaryStatistics();
          long nanCount = Floats.asList(floats).stream().mapToDouble(x -> x).filter(x->Double.isNaN(x)).count();
          long infCount = Floats.asList(floats).stream().mapToDouble(x -> x).filter(x->Double.isInfinite(x)).count();
          System.out.println(takeN(String.format("  Float Statistics=%s; NaN=%s; Inf=%s", finiteStatistics, nanCount, infCount), 1024));
        }
      }
      if(attrMap.containsKey("dtype")) {
        System.out.println(takeN(String.format("  Data Type: %s", attrMap.get("dtype").getType()), 1024));
      }
      if(attrMap.containsKey("shape")) {
        //System.out.println(takeN(String.format("  Shape: %s", attrMap.get("shape")), 1024));
        final TensorShapeProto shape;
        try {
          shape = TensorShapeProto.parseFrom(attrMap.get("shape").toByteString());
        } catch (InvalidProtocolBufferException e) {
          throw new RuntimeException(e);
        }
        System.out.println(takeN(String.format("  Shape: %s", shape), 1024));
      }
      for (int j = 0; j < inputCount; j++) {
        String input = nodeDef.getInput(j);
        System.out.println(takeN(String.format("  Input %s: %s", i, input), 1024*16));
      }
    }
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

  private static float[] cvt(ByteBuffer byteBuffer) {
    FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
    float[] floats = new float[floatBuffer.limit()];
    floatBuffer.get(floats);
    return floats;
  }

  private static float[] cvt2(ByteBuffer byteBuffer) {
    float[] floats = new float[byteBuffer.limit() / 4];
    byte in[] = {0,0,0,0};
    for (int i = 0; i < floats.length; i++) {
      byteBuffer.get(in);
    }
    FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
    floatBuffer.get(floats);
    return floats;
  }

  private static String takeN(String format, int n) {
    return format.substring(0,Math.min(n, format.length()));
  }


}


