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
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import java.net.URI;
import java.util.*;
import java.util.zip.ZipFile;

public abstract class ImageNetworkPipeline {

  @Nonnull
  public final List<GraphDef> graphDefs;
  @Nonnull
  public final GraphModel graphModel;

  public ImageNetworkPipeline(@Nonnull GraphDef graphDef) {
    graphModel = new GraphModel(graphDef.toByteArray());
    this.graphDefs = Collections.unmodifiableList(getNodes(graphModel, nodeIds()));
  }

  @Nonnull
  public static ImageNetworkPipeline inception5h() {
    return new ImageNetworkPipeline(ImageNetworkPipeline.loadGraphZip(
        "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
        "tensorflow_inception_graph.pb"
    )) {
      @Override
      public @Nonnull
      List<String> nodeIds() {
        return Arrays.asList(
            "conv2d0",
            "localresponsenorm1",
            "mixed3a",
            "mixed3b",
            "mixed4a",
            "mixed4b",
            "mixed4c",
            "mixed4d",
            "mixed4e",
            "mixed5a",
            "mixed5b"
        );
      }
    };
  }

  public static GraphDef loadGraphZip(@Nonnull String zipUrl, @Nonnull String zipPath) {
    GraphDef graphDef;
    try (ZipFile zipFile = new ZipFile(Util.cacheFile(new URI(zipUrl)))) {
      byte[] graphDefBytes = IOUtils.toByteArray(zipFile.getInputStream(zipFile.getEntry(zipPath)));
      graphDef = GraphDef.parseFrom(graphDefBytes);
    } catch (Throwable e) {
      throw Util.throwException(e);
    }
    return graphDef;
  }

  @Nonnull
  public static ArrayList<GraphDef> getNodes(@Nonnull GraphModel graphModel, @Nonnull List<String> nodes) {
    ArrayList<GraphDef> graphs = new ArrayList<>();
    graphs.add(graphModel.getChild(nodes.get(0)).subgraph(new HashSet<>(Arrays.asList())));
    for (int i = 1; i < nodes.size(); i++) {
      graphs.add(graphModel.getChild(nodes.get(i)).subgraph(new HashSet<>(Arrays.asList(nodes.get(i - 1)))));
    }
    return graphs;
  }

  @Nonnull
  public abstract List<String> nodeIds();
}
