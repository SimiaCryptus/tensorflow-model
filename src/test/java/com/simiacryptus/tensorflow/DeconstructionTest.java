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
import org.tensorflow.framework.GraphDef;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.UUID;

public class DeconstructionTest {


  @Test
  public void inceptionTest() {
    ImageNetworkPipeline imageNetworkPipeline = ImageNetworkPipeline.inception5h();
    String now = new SimpleDateFormat("yyyyMMddHHmm").format(new Date());
    imageNetworkPipeline.graphDefs.forEach(graphDef -> {
      try {
        launchTensorboard(writeGraph(graphDef, new File("target/" + now + "/tensorboard/" + UUID.randomUUID().toString()), UUID.randomUUID().toString()));
      } catch (IOException e) {
        throw new RuntimeException(e);
      } catch (URISyntaxException e) {
        throw new RuntimeException(e);
      }
    });
  }

  public void launchTensorboard(File tensorboardDir) throws IOException, URISyntaxException {
    TFUtil.launchTensorboard(tensorboardDir.getAbsolutePath(), tensorboard -> {
      try {
        JOptionPane.showConfirmDialog(null, "OK to continue");
        tensorboard.destroyForcibly();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
  }

  public File writeGraph(GraphDef graphDef, File location, String name) throws IOException {
    TensorboardEventWriter eventWriter = new TensorboardEventWriter(
        new File(location, name),
        graphDef);
    eventWriter.write(graphDef);
    eventWriter.close();
    return location;
  }
}
