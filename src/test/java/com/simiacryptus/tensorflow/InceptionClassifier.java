
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

import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;
import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.DecodeJpeg;

import java.net.URI;
import java.util.Arrays;
import java.util.List;
import java.util.zip.ZipFile;

/**
 * Sample use of the TensorFlow Java API to label images using a pre-trained model.
 */
public class InceptionClassifier {

  private final byte[] graphDef;
  private final List<String> labels;

  public byte[] getGraphDef() {
    return graphDef;
  }

  public List<String> getLabels() {
    return labels;
  }

  public InceptionClassifier() {
    try (ZipFile zipFile = new ZipFile(Util.cacheFile(new URI("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip")))) {
      graphDef = IOUtils.toByteArray(zipFile.getInputStream(zipFile.getEntry("tensorflow_inception_graph.pb")));
      labels = IOUtils.readLines(zipFile.getInputStream(zipFile.getEntry("imagenet_comp_graph_label_strings.txt")), "UTF-8");
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }

  public double[] predictImgBytes(byte[] imageBytes) {
    try (Tensor<Float> imageInput = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
      try (Tensor<Float> classificationResult = executeInceptionGraph(graphDef, imageInput)) {
        return TestUtil.getFloatValues(classificationResult);
      }
    }
  }

  private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      final Output<Float> normalizedImage =
          ops.div(
              ops.sub(
                  ops.resizeBilinear(
                      ops.expandDims(
                          ops.cast(
                              ops.decodeJpeg(
                                  ops.constant(imageBytes),
                                  DecodeJpeg.channels(3l)),
                              Float.class),
                          ops.constant(0)),
                      ops.constant(new int[]{224, 224})),
                  ops.constant(117f)),
              ops.constant(1f)).asOutput();
      try (Session session = new Session(graph)) {
        return getFloatTensor(session.runner()
            .fetch(normalizedImage)
            .runAndFetchMetadata());
      }
    }
  }

  public String describeGraph() {
    try (Graph graph = new Graph()) {
      graph.importGraphDef(graphDef);
      return TestUtil.describeGraph(graph);
    }
  }

  private static Tensor<Float> executeInceptionGraph(byte[] graphData, Tensor<Float> image) {
    try (Graph graph = new Graph()) {
      graph.importGraphDef(graphData);
      Ops ops = Ops.create(graph);
      Output<String> summaryOutput = ops.mergeSummary(Arrays.asList(
          ops.histogramSummary(ops.constant("test"), graph.operation("output").output(0))
      )).summary();
      try (Session session = new Session(graph)) {
        Session.Runner runner = session.runner()
            .feed("input", image)
            .fetch("output")
            .fetch(summaryOutput);
        return getFloatTensor(runner.runAndFetchMetadata());
      }
    }
  }

  public static Tensor<Float> getFloatTensor(Session.Run run) {
    return run.outputs.get(0).expect(Float.class);
  }


}


