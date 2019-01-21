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

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: config.proto

package org.tensorflow.framework;

public interface GraphOptionsOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.GraphOptions)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * If true, use control flow to schedule the activation of Recv nodes.
   * (Currently ignored.)
   * </pre>
   *
   * <code>bool enable_recv_scheduling = 2;</code>
   */
  boolean getEnableRecvScheduling();

  /**
   * <pre>
   * Options controlling how graph is optimized.
   * </pre>
   *
   * <code>.tensorflow.OptimizerOptions optimizer_options = 3;</code>
   */
  boolean hasOptimizerOptions();
  /**
   * <pre>
   * Options controlling how graph is optimized.
   * </pre>
   *
   * <code>.tensorflow.OptimizerOptions optimizer_options = 3;</code>
   */
  org.tensorflow.framework.OptimizerOptions getOptimizerOptions();
  /**
   * <pre>
   * Options controlling how graph is optimized.
   * </pre>
   *
   * <code>.tensorflow.OptimizerOptions optimizer_options = 3;</code>
   */
  org.tensorflow.framework.OptimizerOptionsOrBuilder getOptimizerOptionsOrBuilder();

  /**
   * <pre>
   * The number of steps to run before returning a cost model detailing
   * the memory usage and performance of each node of the graph. 0 means
   * no cost model.
   * </pre>
   *
   * <code>int64 build_cost_model = 4;</code>
   */
  long getBuildCostModel();

  /**
   * <pre>
   * The number of steps to skip before collecting statistics for the
   * cost model.
   * </pre>
   *
   * <code>int64 build_cost_model_after = 9;</code>
   */
  long getBuildCostModelAfter();

  /**
   * <pre>
   * Annotate each Node with Op output shape data, to the extent it can
   * be statically inferred.
   * </pre>
   *
   * <code>bool infer_shapes = 5;</code>
   */
  boolean getInferShapes();

  /**
   * <pre>
   * Only place the subgraphs that are run, rather than the entire graph.
   * This is useful for interactive graph building, where one might
   * produce graphs that cannot be placed during the debugging
   * process.  In particular, it allows the client to continue work in
   * a session after adding a node to a graph whose placement
   * constraints are unsatisfiable.
   * </pre>
   *
   * <code>bool place_pruned_graph = 6;</code>
   */
  boolean getPlacePrunedGraph();

  /**
   * <pre>
   * If true, transfer float values between processes as bfloat16.
   * </pre>
   *
   * <code>bool enable_bfloat16_sendrecv = 7;</code>
   */
  boolean getEnableBfloat16Sendrecv();

  /**
   * <pre>
   * If &gt; 0, record a timeline every this many steps.
   * EXPERIMENTAL: This currently has no effect in MasterSession.
   * </pre>
   *
   * <code>int32 timeline_step = 8;</code>
   */
  int getTimelineStep();

  /**
   * <pre>
   * Options that control the type and amount of graph rewriting.
   * Not currently configurable via the public Python API (i.e. there is no API
   * stability guarantee if you import RewriterConfig explicitly).
   * </pre>
   *
   * <code>.tensorflow.RewriterConfig rewrite_options = 10;</code>
   */
  boolean hasRewriteOptions();
  /**
   * <pre>
   * Options that control the type and amount of graph rewriting.
   * Not currently configurable via the public Python API (i.e. there is no API
   * stability guarantee if you import RewriterConfig explicitly).
   * </pre>
   *
   * <code>.tensorflow.RewriterConfig rewrite_options = 10;</code>
   */
  org.tensorflow.framework.RewriterConfig getRewriteOptions();
  /**
   * <pre>
   * Options that control the type and amount of graph rewriting.
   * Not currently configurable via the public Python API (i.e. there is no API
   * stability guarantee if you import RewriterConfig explicitly).
   * </pre>
   *
   * <code>.tensorflow.RewriterConfig rewrite_options = 10;</code>
   */
  org.tensorflow.framework.RewriterConfigOrBuilder getRewriteOptionsOrBuilder();
}
