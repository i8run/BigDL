/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.mkldnn.models

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{ConstInitMethod, Xavier, Zeros}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.utils.{T, Table}

object Inception_v1_NoAuxClassifier {
  def apply(batchSize: Int, classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val model = Sequential()
    model.add(Input(Array(batchSize, 3, 224, 224), Memory.Format.nchw))
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv1/7x7_s2"))
    model.add(ReLU().setName("conv1/relu_7x7"))
    model.add(MaxPooling(3, 3, 2, 2).setName("pool1/3x3_s2")) // TODO check the ceil
    model.add(LRN(5, 0.0001, 0.75).setName("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1).
      setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv2/3x3_reduce"))
    model.add(ReLU().setName("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName("conv2/3x3"))
    model.add(ReLU().setName("conv2/relu_3x3"))
    model.add(LRN(5, 0.0001, 0.75).setName("conv2/norm2"))
    model.add(MaxPooling(3, 3, 2, 2).setName("pool2/3x3_s2")) // TODO check the ceil
    model.add(Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    model.add(Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    model.add(MaxPooling(3, 3, 2, 2).setName("pool3/3x3_s2")) // TODO check the ceil
    model.add(Inception_Layer_v1(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))
    model.add(Inception_Layer_v1(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    model.add(Inception_Layer_v1(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    model.add(Inception_Layer_v1(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))
    model.add(Inception_Layer_v1(528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    model.add(MaxPooling(3, 3, 2, 2).setName("pool4/3x3_s2")) // TODO check the ceil
    model.add(Inception_Layer_v1(832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    model.add(Inception_Layer_v1(832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    model.add(AvgPooling(7, 7, 1, 1).setName("pool5/7x7_s1"))
    if (hasDropout) model.add(Dropout(0.4).setName("pool5/drop_7x7_s1"))
    model.add(Linear(1024, classNum)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName("loss3/classifier"))
//    model.add(LogSoftMax().setName("loss3/loss3"))
    model.add(ReorderMemory(HeapData(Array(batchSize, classNum), Memory.Format.nc)))
    model
  }

}

object Inception_Layer_v1 {
  def apply(inputSize: Int, config: Table, namePrefix : String = "") : Module[Float] = {
    val seq = Sequential()
    val concat = ConcatTable()
    val conv1 = Sequential()
    conv1.add(SpatialConvolution(inputSize,
      config[Table](1)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName(namePrefix + "1x1"))
    conv1.add(ReLU().setName(namePrefix + "relu_1x1"))
    concat.add(conv1)
    val conv3 = Sequential()
    conv3.add(SpatialConvolution(inputSize,
      config[Table](2)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "3x3_reduce"))
    conv3.add(ReLU().setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName(namePrefix + "3x3"))
    conv3.add(ReLU().setName(namePrefix + "relu_3x3"))
    concat.add(conv3)
    val conv5 = Sequential()
    conv5.add(SpatialConvolution(inputSize,
      config[Table](3)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "5x5_reduce"))
    conv5.add(ReLU().setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[Table](3)(1),
      config[Table](3)(2), 5, 5, 1, 1, 2, 2)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName(namePrefix + "5x5"))
    conv5.add(ReLU().setName(namePrefix + "relu_5x5"))
    concat.add(conv5)
    val pool = Sequential()
    pool.add(MaxPooling(3, 3, 1, 1, 1, 1).setName(namePrefix + "pool")) // TODO ceil
    pool.add(SpatialConvolution(inputSize,
      config[Table](4)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "pool_proj"))
    pool.add(ReLU().setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")

    seq.add(concat).add(JoinTable(2))
  }
}
