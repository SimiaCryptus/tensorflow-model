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

import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import com.simiacryptus.util.test.TestCategories;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.TensorFlowException;
import org.tensorflow.framework.GraphDef;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static com.simiacryptus.tensorflow.TFUtil.find;

public class InceptionTest {

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  @Test
  @Category(TestCategories.ResearchCode.class)
  public void testReferenceData() throws IOException {
    System.out.println(GraphDef.parseFrom(FileUtils.readFileToByteArray(new File("H:\\SimiaCryptus\\tensorflow\\tensorflow\\examples\\tutorials\\mnist\\model\\train.pb"))));
    TFUtil.streamEvents("H:\\SimiaCryptus\\tensorflow\\tensorflow\\examples\\tutorials\\mnist\\tmp\\test\\events.out.tfevents.1549408929.DESKTOP-L7C95P7")
        .map(event -> event.toString())
        .forEach(x -> System.out.println(x));
  }

  @Test
  @Category(TestCategories.ResearchCode.class)
  public void testClassification() throws Exception {
    File reportFile = new File("target/out/" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
    MarkdownNotebookOutput log = new MarkdownNotebookOutput(reportFile, true);
    InceptionClassifier classifier = new InceptionClassifier();
    String logDir = classifier.eventWriterLocation.getAbsolutePath();
    try {
      CloseableHttpClient client = HttpClientBuilder.create().build();
      for (String keyword : Arrays.asList("dog", "cat", "ship", "city")) {
        log.h1("Image Category: " + keyword);
        byte[] bytes = IOUtils.toByteArray(client.execute(new HttpGet("https://loremflickr.com/320/240/" + keyword)).getEntity().getContent());
        log.p(log.jpg(ImageIO.read(new ByteArrayInputStream(bytes)), "Random Image"));
        log.run(() -> {
          double[] predictions = classifier.predictImgBytes(bytes);
          int[] topValues = IntStream.range(0, predictions.length).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictions[i])).limit(5).mapToInt(i -> i).toArray();
          for (int index : topValues) {
            System.out.println(String.format("%s = %.3f%%", classifier.labels.get(index), predictions[index] * 100.0));
          }
        });
      }
    } finally {
      log.close();
      classifier.close();
    }

    for (File file : TFUtil.allFiles(new File(logDir))) {
      TFUtil.dumpEvents(file.getAbsolutePath());
    }

    TFUtil.launchTensorboard(logDir, tensorboard -> {
      try {
        tensorboard.waitFor(1, TimeUnit.MINUTES);
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    });
  }

  @Test
  public void dumpModelJson() throws Exception {
    byte[] protobufBinaryData = loadGraphDef();
    GraphModel model = new GraphModel(protobufBinaryData);
    System.out.println("Model: " + TFUtil.toJson(model));
    try (Graph graph = new Graph()) {
      graph.importGraphDef(model.graphDef.toByteArray());
      System.out.println(TFUtil.describeGraph(graph));
    }
  }

  @Test
  public void testGradient() throws Exception {
    byte[] originalGraphDef = loadGraphDef();
    byte[] newGraphDef = TFUtil.editGraph(originalGraphDef, graph -> {
      graph.addGradients("gradient_", new Output[]{
          find(graph, "mixed4b_1x1_pre_relu/conv").output(0)
      }, new Output[]{
          find(graph, "mixed4a").output(0),
          find(graph, "mixed4b_1x1_w").output(0)
      }, null);
    });
    GraphModel model = new GraphModel(newGraphDef);
    System.out.println("Model: " + TFUtil.toJson(model));
  }

  @Test
  public void testFullGradient() throws Exception {
    try {
      byte[] originalGraphDef = loadGraphDef();
      byte[] newGraphDef = TFUtil.editGraph(originalGraphDef, graph -> {
        graph.addGradients("gradient", new Output[]{
            find(graph, "output").output(0)
        }, new Output[]{
            find(graph, "input").output(0)
        }, null);
      });
      GraphModel model = new GraphModel(newGraphDef);
      System.out.println("Model: " + TFUtil.toJson(model));
    } catch (TensorFlowException e) {
      e.printStackTrace(System.err);
    }
  }

  protected byte[] loadGraphDef() throws Exception {
    return TFUtil.loadZipUrl(
        "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
        "tensorflow_inception_graph.pb"
    );
  }


}


