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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import com.intel.analytics.bigdl.nn.{Module, Xavier, Zeros}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

class Int8Spec extends FlatSpec with Matchers {
  private object Conv {
    def apply(
      nInputPlane: Int,
      nOutputPlane: Int,
      kernelW: Int,
      kernelH: Int,
      strideW: Int = 1,
      strideH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      nGroup: Int = 1,
      propagateBack: Boolean = true): SpatialConvolution = {
      val conv = SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack)
      conv.setInitMethod(Xavier.setVarianceNormAverage(false), Zeros)
      conv
    }
  }

//  "lenet int8" should "work correctly" in {
//    val batchSize = 4
//    val inputShape = Array(batchSize, 1, 28, 28)
//    val outputShape = Array(batchSize, 10)
//
//    val model = Sequential()
//      .add(Input(inputShape, Memory.Format.nchw))
//      .add(SpatialConvolution(1, 20, 5, 5).setName("conv1"))
//      .add(MaxPooling(2, 2, 2, 2).setName("pool1"))
//      .add(SpatialConvolution(20, 50, 5, 5).setName("conv2"))
//      .add(MaxPooling(2, 2, 2, 2).setName("pool2"))
//      .add(Linear(50 * 4 * 4, 500).setName("ip1"))
//      .add(Linear(500, 10).setName("ip2"))
//      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nc)))
//
//    val inputs = new Array[Tensor[Float]](3)
//    inputs.indices.foreach { i =>
//      inputs(i) = Tensor[Float](inputShape).rand(-1, 1)
//    }
//
//    val quantized = model.quantize(inputs)
//    // stuck at the mkldnn v0.17 upgrading, because of padding tensors
//    quantized.forward(inputs(0))
//
//    println(quantized.output)
//  }

  "vgg16 int8" should "work correctly" in {
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
    RNG.setSeed(1)

    val batchSize = 4
    val classNum = 1000
    val model = Vgg_16(batchSize, classNum)
    model.asInstanceOf[Sequential].modules.foreach{ x =>
      if (x.isInstanceOf[SpatialConvolution]) {
        val conv = x.asInstanceOf[SpatialConvolution]
        val weight = Tensor.load[Float](s"/tmp/${conv.getName()}_1.bin")
        val bias = Tensor.load[Float](s"/tmp/${conv.getName()}_2.bin")
        conv.parameters()._1(0).copy(weight)
        conv.parameters()._1(1).copy(bias)
      }
    }

    val inputs = new Array[Tensor[Float]](1)

    inputs.indices.foreach { i =>
      inputs(i) = Tensor[Float](Array(4, 3, 224, 224)).rand(-124, 151)
      println(s"forward ${i}")
    }

    model.compile(InferencePhase)
    model.forward(inputs(0))

//    val quantized = model.quantize(inputs)
//    quantized.forward(inputs(0))
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "false")
  }

  "weight scale" should "work correctly" in {
    def scalesOf(tensor: Tensor[Float], indexes: Array[Int]): Tensor[Float] = {
      if (indexes.length == 1) {
        tensor.max(indexes.head)._1
      } else {
        scalesOf(tensor.max(indexes.head)._1, indexes.drop(1))
      }
    }
    val t = Tensor[Float](Array(4, 8, 3, 3)).rand(-1, 1)

    println(scalesOf(t, Array(2, 3, 4)))
  }

//  "2 convs" should "work correctly" in {
//    val batchSize = 4
//    val model = Sequential()
//    model.add(Input(Array(batchSize, 3, 224, 224), Memory.Format.nchw))
//    model.add(Conv(3, 64, 3, 3, 1, 1, 1, 1).setName("conv1_1"))
//    model.add(Conv(64, 64, 3, 3, 1, 1, 1, 1).setName("conv1_2"))
//
//    model.quantize(Array(Tensor[Float](Array(batchSize, 3, 224, 224)).rand(-1, 1)))
//  }

  "load vgg model" should "work correctly" in {
    val model = Vgg_16(32, 1000)
    val weights = Tensor.load[Float]("/tmp/vgg16.weights")

    model.getParameters()._1.set(weights)
  }
}
