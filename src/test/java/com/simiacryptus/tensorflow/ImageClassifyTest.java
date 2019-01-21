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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.apache.commons.io.IOUtils;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.stream.IntStream;

public class ImageClassifyTest {
  static {
    SysOutInterceptor.INSTANCE.init();
  }
  @Test
  public void test() throws IOException {
    CloseableHttpClient client = HttpClientBuilder.create().build();
    InceptionClassifier classifier = new InceptionClassifier();
    File reportFile = new File("target/out/" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
    MarkdownNotebookOutput log = new MarkdownNotebookOutput(reportFile, true);

    try {
      log.h1("Network Info");
      log.p("We can read the operations and available outputs from the loaded, executable graph object:");
      log.eval(()->{
        return classifier.describeGraph();
      });
      log.p("Further information about weights, inputs, and graph connectivity can be read by parsing the protobuf definition:");
      log.run(()->{
        try {
          GraphModel model = new GraphModel(classifier.getGraphDef());
          System.out.println("Model: " + TestUtil.toJson(model));
        } catch (JsonProcessingException e) {
          throw new RuntimeException(e);
        }
      });
      for(String keyword : Arrays.asList("dog","cat","ship","city")) {
        log.h1("Image Category: " + keyword);
        byte[] bytes = IOUtils.toByteArray(client.execute(new HttpGet("https://loremflickr.com/320/240/" + keyword)).getEntity().getContent());
        log.p(log.jpg(ImageIO.read(new ByteArrayInputStream(bytes)), "Random Image"));
        log.run(()->{
          double[] predictions = classifier.predictImgBytes(bytes);
          int[] topValues = IntStream.range(0, predictions.length).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictions[i])).limit(5).mapToInt(i -> i).toArray();
          for (int index : topValues) {
            System.out.println(String.format("%s = %.3f%%", classifier.getLabels().get(index), predictions[index] * 100.0));
          }
        });
      }
    } finally {
      log.close();
    }
  }
}
