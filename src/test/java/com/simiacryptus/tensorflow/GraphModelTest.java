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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;
import org.junit.Test;

import java.net.URI;
import java.util.zip.ZipFile;

public class GraphModelTest {

  @Test
  public void test() throws Exception {
    byte[] protobufBinaryData = loadZipUrl(
        "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
        "tensorflow_inception_graph.pb"
    );
    GraphModel model = new GraphModel(protobufBinaryData);
    GraphModel.GraphNode output = model.getChild("output");
    System.out.println("Model: " + toJson(output));
  }

  public byte[] loadZipUrl(String uri, String file) throws Exception {
    try (ZipFile zipFile = new ZipFile(Util.cacheFile(new URI(uri)))) {
      return IOUtils.toByteArray(zipFile.getInputStream(zipFile.getEntry(file)));
    }
  }

  public String toJson(Object output) throws JsonProcessingException {
    return new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT).writeValueAsString(output);
  }

}


