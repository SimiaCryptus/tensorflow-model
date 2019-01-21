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

import org.junit.Test;
import org.tensorflow.Output;

import static com.simiacryptus.tensorflow.TestUtil.find;

public class InceptionTest {
  @Test
  public void testModelJson() throws Exception {
    byte[] protobufBinaryData = loadGraphDef();
    GraphModel model = new GraphModel(protobufBinaryData);
    System.out.println("Model: " + TestUtil.toJson(model));
  }

  @Test
  public void testGradient() throws Exception {
    byte[] originalGraphDef = loadGraphDef();
    byte[] newGraphDef = TestUtil.editGraph(originalGraphDef, graph -> {
      graph.addGradients("gradient_", new Output[]{
          find(graph, "mixed4b_1x1_pre_relu/conv").output(0)
      }, new Output[]{
          find(graph, "mixed4a").output(0),
          find(graph, "mixed4b_1x1_w").output(0)
      }, null);
    });
    GraphModel model = new GraphModel(newGraphDef);
    System.out.println("Model: " + TestUtil.toJson(model));
  }

  @Test
  public void testFullGradient() throws Exception {
    try {
      byte[] originalGraphDef = loadGraphDef();
      byte[] newGraphDef = TestUtil.editGraph(originalGraphDef, graph -> {
        graph.addGradients("gradient", new Output[]{
            find(graph, "output").output(0)
        }, new Output[]{
            find(graph, "input").output(0)
        }, null);
      });
      GraphModel model = new GraphModel(newGraphDef);
      System.out.println("Model: " + TestUtil.toJson(model));
    } catch (org.tensorflow.TensorFlowException e) {
      e.printStackTrace(System.err);
    }
  }

  protected byte[] loadGraphDef() throws Exception {
    return TestUtil.loadZipUrl(
        "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
        "tensorflow_inception_graph.pb"
    );
  }


}


