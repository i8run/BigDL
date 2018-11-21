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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.abstractnn.Int8ScalesAndMask
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{Module, Xavier, Zeros}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{DnnStorage, DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class SpatialConvolutionSpec extends FlatSpec with Matchers {
  "ConvolutionDnn with format=nchw and ngroup=1" should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    RNG.setSeed(100)
    val layer = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

    val seq = Sequential()
      .add(Input(input.size(), Memory.Format.nchw))
      .add(conv)
      .add(ReorderMemory(HeapData(gradOutput.size(), Memory.Format.nchw)))

    seq.compile(TrainingPhase)

    val output = seq.forward(input)
    val grad1 = seq.backward(input, gradOutput)

    val weight1 = conv.weight.dense
    val gradweight1 = conv.gradWeight.dense
    val bias1 = conv.bias.dense
    val gradbias1 = conv.gradBias.dense

    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = layer.bias
    val gradbias2 = layer.gradBias

    Equivalent.nearequals(weight1, weight2.resizeAs(weight1)) should be(true)
    Equivalent.nearequals(gradweight1, gradweight2.resizeAs(gradweight1)) should be(true)
    Equivalent.nearequals(bias1, bias2) should be(true)
    Equivalent.nearequals(gradbias1, gradbias2) should be(true)
    Equivalent.nearequals(output.toTensor, output2) should be(true)
    Equivalent.nearequals(grad1.toTensor, grad2) should be(true)
  }

  "ConvolutionDnn with same padding" should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = -1
    val padH = -1

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    RNG.setSeed(100)
    val layer = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

    val seq = Sequential()
        .add(Input(input.size(), Memory.Format.nchw))
        .add(conv)
        .add(ReorderMemory(HeapData(gradOutput.size(), Memory.Format.nchw)))

    seq.compile(TrainingPhase)

    val output = seq.forward(input)
    val grad1 = seq.backward(input, gradOutput)

    val weight1 = conv.weight.dense
    val gradweight1 = conv.gradWeight.dense
    val bias1 = Tools.dense(conv.bias.native[Float]).toTensor[Float]
    val gradbias1 = Tools.dense(conv.gradBias.dense).toTensor

    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = layer.bias
    val gradbias2 = layer.gradBias

    Equivalent.nearequals(weight1, weight2.resizeAs(weight1)) should be(true)
    Equivalent.nearequals(gradweight1, gradweight2.resizeAs(gradweight1)) should be(true)
    Equivalent.nearequals(bias1, bias2) should be(true)
    Equivalent.nearequals(gradbias1, gradbias2) should be(true)
    Equivalent.nearequals(output.toTensor, output2) should be(true)
    Equivalent.nearequals(grad1.toTensor, grad2) should be(true)
  }

  "ConvolutionDnn with format=nchw and ngroup=2" should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0
    val ngroup = 2

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, ngroup)
    RNG.setSeed(100)
    val layer = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH,
      dW, dH, padW, padH, ngroup)

    val seq = Sequential()
        .add(Input(input.size(), Memory.Format.nchw))
        .add(conv)
        .add(ReorderMemory(HeapData(gradOutput.size(), Memory.Format.nchw)))

    seq.compile(TrainingPhase)

    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)
    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = layer.bias
    val gradbias2 = layer.gradBias

    val output = seq.forward(input).toTensor[Float]
    val grad1 = seq.backward(input, gradOutput).toTensor[Float]
    val weight1 = conv.weight.dense
    val gradweight1 = conv.gradWeight.dense
    val bias1 = Tools.dense(conv.bias.native).toTensor[Float]
    val gradbias1 = Tools.dense(conv.gradBias.native).toTensor[Float]

    Equivalent.nearequals(weight1, weight2) should be(true)
    Equivalent.nearequals(gradweight1, gradweight2) should be(true)
    Equivalent.nearequals(bias1, bias2) should be(true)
    Equivalent.nearequals(gradbias1, gradbias2) should be(true)
    Equivalent.nearequals(output, output2) should be(true)
    Equivalent.nearequals(grad1, grad2) should be(true)
  }

  "ConvolutionDnn with relu " should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0
    val ngroup = 2

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, ngroup)
    RNG.setSeed(100)
    val conv1 = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,
      ngroup)

    val relu = ReLU()
    val relu1 = nn.ReLU[Float](ip = false)

    val model = Sequential()
      .add(Input(input.size(), Memory.Format.nchw))
      .add(conv)
      .add(relu)
      .add(ReorderMemory(HeapData(Array(2, 4, 6, 6), Memory.Format.nchw)))
    model.compile(TrainingPhase)

    val model1 = nn.Sequential().add(conv1).add(relu1)

    model.forward(input)
    model.backward(input, gradOutput)

    model1.forward(input)
    model1.backward(input, gradOutput)

    val output = Tools.toNCHW(conv.output.toTensor, conv.outputFormats()(0))
    val gradInput = Tools.toNCHW(conv.gradInput.toTensor, conv.gradInputFormats()(0))

    val weight = conv.weight.dense
    val gradweight = conv.gradWeight.dense
    val bias = Tools.dense(conv.bias.native).toTensor
    val gradbias = Tools.dense(conv.gradBias.native).toTensor

    val output1 = conv1.output.toTensor
    val gradInput1 = conv1.gradInput

    val weight1 = conv1.weight
    val gradweight1 = conv1.gradWeight
    val bias1 = conv1.bias
    val gradbias1 = conv1.gradBias

    Equivalent.nearequals(weight, weight1) should be(true)
    Equivalent.nearequals(gradweight, gradweight1) should be(true)
    Equivalent.nearequals(bias, bias1) should be(true)
    Equivalent.nearequals(gradbias, gradbias1) should be(true)
    Equivalent.nearequals(output, output1) should be(true)
    Equivalent.nearequals(gradInput, gradInput1) should be(true)
  }

  "ConvolutionDnn with same params with vgg16" should "work correctly" in {
    val batchSize = 2
    val needPropagateBack: Boolean = true
    val inputShape = Array(batchSize, 3, 224, 224)
    val outputShape = Array(batchSize, 64, 112, 112)

    RNG.setSeed(100)
    val model1 = nn.SpatialConvolution[Float](3, 64, 7, 7, 2, 2, 3, 3, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros)
    model1.zeroGradParameters()
    val (weightAll1, gradWeightAll1) = model1.parameters()

    RNG.setSeed(100)
    val conv = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1)
    val model2 = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(conv)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    model2.zeroGradParameters()

    model2.compile(TrainingPhase)

    conv.weight.dense.copy(model1.weight)
    conv.bias.copy(model1.bias)

    RNG.setSeed(1)
    val input = Tensor(batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)
    val gradOutput = Tensor(outputShape).apply1(_ => RNG.uniform(0, 1).toFloat)

    val (weightAll2, gradWeightAll2) = model2.parameters()

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]

    var userOut2 = Tools.toNCHW(out2, model2.outputFormats()(0))

    Equivalent.nearequals(out1, userOut2, 1e-4) should be(true)

    val grad1 = model1.updateGradInput(input, gradOutput).toTensor[Float]
    val grad2 = model2.updateGradInput(input, gradOutput).toTensor[Float]

    val userGradInput2 = Tools.toNCHW(grad2, model2.gradInputFormats()(0))

    Equivalent.nearequals(grad1, userGradInput2, 1e-4) should be(true)

    model1.accGradParameters(input, gradOutput)
    model2.accGradParameters(input, gradOutput)

    val gw1 = model1.gradWeight
    val gb1 = model1.gradBias

    val gw2 = conv.gradWeight.dense
    val gb2 = conv.gradBias.dense

    Equivalent.nearequals(gw1, gw2, 1e-4) should be(true)
    Equivalent.nearequals(gb1, gb2, 1e-3) should be(true)
  }

  "a simple convolution compared with caffe" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val name = "conv"
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val txt = prototxt(inputShape, name, nOutput, kernel, pad, stride)

    val conv = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
    conv.setName(name)

    val seq = Sequential()
      .add(ReorderMemory(NativeData(inputShape, Memory.Format.nchw)))
      .add(conv)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))
    seq.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    // because after upgrading v0.17, the epsilon should be not 1e-7.
    Tools.compare(txt, seq, inputShape, outputShape, epsilon = 1e-5)
  }

  "conv exists some format conversion" should "work correctly" in {
    val inputShape = Array(4, 3, 224, 224)
    val outputShape = Array(4, 64, 112, 112)

    val name = "conv"
    val conv = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3).setName(name)
    // TODO we should insert a reorder manually
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder2 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(conv)
    seq.add(reorder2)
    seq.compile(Phase.TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    seq.reset()

    val txt = prototxt(inputShape, name, outputShape(1), 7, 3, 2)
    val identity = Collect.run(txt)

    val input = Tools.getTensor("Fwrd_data", inputShape, identity)
    val gradOutput = Tools.getTensor(s"Bwrd_$name.loss", outputShape, identity)
    val output = Tools.getTensor(s"Fwrd_$name", outputShape, identity)
    val gradInput = Tools.getTensor(s"Bwrd_$name", inputShape, identity)

    if (conv.parameters() != null) {
      val params = conv.parameters()._1
      val name = conv.getName()

      for (j <- params.indices) {
        val w = Tools.getTensor(s"Fwrd_$name.Wght.$j", params(j).size(), identity)
        params(j).copy(w)
      }
    }

    seq.forward(input)
    seq.backward(input, gradOutput)

    Tools.compare2Tensors(Tools.dense(seq.output).toTensor, output) should be (true)
    Tools.compare2Tensors(Tools.dense(seq.gradInput).toTensor, gradInput) should be (true)

    val params = seq.parameters()._2
    for (j <- params.indices) {
      val w = Tools.getTensor(s"Bwrd_$name.Grad.$j", params(j).size(), identity)
      Tools.compare2Tensors(params(j), w) should be (true)
    }
  }

  "conv kernel 1x1 with reorder in container" should "work correctly" in {
    val inputShape = Array(4, 64, 56, 56)
    val outputShape = Array(4, 64, 56, 56)

    val name = "conv"
    val conv = SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0).setName(name)
    // TODO we should insert a reorder manually
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder2 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(conv)
    seq.add(reorder2)
    seq.compile(Phase.TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    seq.reset()

    val txt = prototxt(inputShape, name, outputShape(1), 1, 0, 1)
    val identity = Collect.run(txt)

    val input = Tools.getTensor("Fwrd_data", inputShape, identity)
    val gradOutput = Tools.getTensor(s"Bwrd_$name.loss", outputShape, identity)
    val output = Tools.getTensor(s"Fwrd_$name", outputShape, identity)
    val gradInput = Tools.getTensor(s"Bwrd_$name", inputShape, identity)

    if (conv.parameters() != null) {
      val params = conv.parameters()._1
      val name = conv.getName()

      for (j <- params.indices) {
        val w = Tools.getTensor(s"Fwrd_$name.Wght.$j", params(j).size(), identity)
        params(j).copy(w)
      }
    }

    seq.forward(input)
    seq.backward(input, gradOutput)

    Tools.compare2Tensors(Tools.dense(seq.output).toTensor, output) should be (true)
    Tools.compare2Tensors(Tools.dense(seq.gradInput).toTensor, gradInput) should be (true)

    val params = seq.parameters()._2
    for (j <- params.indices.reverse) {
      val w = Tools.getTensor(s"Bwrd_$name.Grad.$j", params(j).size(), identity)
      Tools.compare2Tensors(params(j), w) should be (true)
    }
  }

  "conv + bn" should "work correctly" in {
    val inputShape = Array(4, 3, 224, 224)
    val outputShape = Array(4, 64, 112, 112)
    val channel = 64

    val name = "conv"
    val conv = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3).setName("conv")
    val bn = SpatialBatchNormalization(64, momentum = 1.0, eps = 100).setName("bn")
    // TODO we should insert a reorder manually
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw)).setName("reorder1")
    val reorder2 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw)).setName("reorder2")

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(conv)
    seq.add(bn)
    seq.add(reorder2)
    seq.compile(Phase.TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    seq.reset()
    seq.training()

    val txt = prototxt2(inputShape, name, outputShape(1), 7, 3, 2) +
              """
                |layer {
                |  bottom: "conv"
                |  top: "bn"
                |  name: "bn"
                |  type: "BatchNorm"
                |
                |  batch_norm_param {
                |    moving_average_fraction: 1.0
                |    filler { value: 1 }
                |    bias_filler { value: 0 }
                |    relu: false
                |    eps: 100
                |  }
                |}
              """.stripMargin
    Tools.compare(txt, seq, inputShape, outputShape, 1e-2)
  }

  "conv serialized with java serialization method" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val name = "conv"
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val conv = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
    conv.setName(name)

    val seq = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(conv)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    seq.compile(TrainingPhase)

    val input = Tensor(inputShape).rand(-1, 1)
    seq.forward(input)

    val cloned = SerializationUtils.clone(seq)
    cloned.compile(TrainingPhase)

    cloned.forward(input)

    Tools.dense(seq.output) should be (Tools.dense(cloned.output))

    val gradOutput = Tensor(outputShape).rand(-1, 1)

    seq.backward(input, gradOutput)
    cloned.backward(input, gradOutput)

    Tools.dense(seq.gradInput) should be (Tools.dense(cloned.gradInput))
    seq.getParameters()._1 should be (cloned.getParameters()._1)
  }

  "conv release" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val name = "conv"
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val initCount = DnnStorage.get().count(!_._2)
    val conv = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
    conv.setName(name)

    val seq = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(conv)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))
    seq.compile(TrainingPhase)

    val input = Tensor(inputShape).rand(-1, 1)
    val gradOutput = Tensor(outputShape).rand(-1, 1)
    seq.forward(input)
    seq.backward(input, gradOutput)

    seq.release()
    DnnStorage.get().count(_._2 == false) should be (initCount)
  }

  "conv with dense weights and gradients" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val input = Tensor(inputShape).rand(-1, 1)
    val gradOutput = Tensor(outputShape).rand(-1, 1)

    val initWeight1 = Tensor(Array(nOutput, inputShape(1), kernel, kernel)).rand(-1, 1)
    val initWeight2 = Tensor(Array(nOutput, inputShape(1), kernel, kernel)).rand(-1, 1)
    val initBias1 = Tensor(nOutput).rand(-1, 1)
    val initBias2 = Tensor(nOutput).rand(-1, 1)

    val conv1 = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1,
      initWeight = initWeight1, initBias = initBias1)
    val seq1 = Sequential()
        .add(Input(inputShape, Memory.Format.nchw))
        .add(conv1)
        .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))
    seq1.compile(TrainingPhase)

    seq1.forward(input)
    seq1.backward(input, gradOutput)

    conv1.parameters()._1.zip(Array(initWeight2, initBias2)).foreach(x => x._1.copy(x._2))
    seq1.forward(input)
    seq1.backward(input, gradOutput)

    val conv2 = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1,
      initWeight = initWeight2, initBias = initBias2)

    val seq2 = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(conv2)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))
    seq2.compile(TrainingPhase)

    seq2.forward(input)
    seq2.backward(input, gradOutput)

    Tools.dense(conv1.output) should be (Tools.dense(conv2.output))
    Tools.dense(conv1.gradInput) should be (Tools.dense(conv2.gradInput))

    conv1.parameters()._2.zip(conv2.parameters()._2).foreach(x => x._1 should be (x._2))
  }

  "lenet conv1" should "work correctly" in {
    // test the padding tensor
    val inputShape = Array(4, 1, 28, 28)
    val outputShape = Array(4, 20, 24, 24)
    val dnn = SpatialConvolution(1, 20, 5, 5)
    val blas = com.intel.analytics.bigdl.nn.SpatialConvolution[Float](1, 20, 5, 5)

    val model = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(dnn)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    model.compile(TrainingPhase)

    val input = Tensor[Float](4, 1, 28, 28).rand(-1, 1)
    val gradOutput = Tensor[Float](outputShape).rand(-1, 1)

    model.forward(input)
    model.updateGradInput(input, gradOutput)
    model.accGradParameters(input, gradOutput)

    blas.getParameters()._1.copy(dnn.getParameters()._1)

    blas.forward(input)
    blas.updateGradInput(input, gradOutput)
    blas.accGradParameters(input, gradOutput)

    Equivalent.nearequals(model.output.toTensor, blas.output) should be (true)
    Equivalent.nearequals(model.gradInput.toTensor, blas.gradInput) should be (true)
    Equivalent.nearequals(model.getParameters()._1, blas.getParameters()._1) should be (true)
    // control the epsilon to 1e-4, not 1e-5
    Equivalent.nearequals(model.getParameters()._2, blas.getParameters()._2, 1e-4) should be (true)
  }

  "quantized convolution" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val input = Tensor(inputShape).rand(0, 1)

    val runtime = new MklDnnRuntime

    val heapData = HeapData(inputShape, Memory.Format.nchw, DataType.F32)
    val nativeData = NativeData(Array(4, 3, 5, 5), Memory.Format.nhwc, DataType.U8)
    val inputScales = Array(input.max())

    heapData.setMask(0)
    heapData.setScales(inputScales)

    val reorder = ReorderMemory(nativeData)
    reorder.setRuntime(runtime)
    reorder.initFwdPrimitives(Array(heapData), InferencePhase)

    val reorderedInput = reorder.forward(input)

    {
      val len = inputShape.product
      val output = new Array[Byte](len)
      Memory.CopyPtr2ByteArray(reorderedInput.asInstanceOf[DnnTensor[Byte]].storageAddress(),
        0, output, 0, len, 1)
      output.foreach(println)
    }

    val initWeight1 = Tensor(Array(nOutput, inputShape(1), kernel, kernel)).rand(-1, 1)
    val initWeight2 = Tensor(Array(nOutput, inputShape(1), kernel, kernel)).rand(-1, 1)
    val initBias1 = Tensor(nOutput).rand(-1, 1)
    val initBias2 = Tensor(nOutput).rand(-1, 1)

    val conv1 = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1,
      initWeight = initWeight1, initBias = initBias1)
    conv1.evaluate()
    conv1.setRuntime(runtime)
    conv1.initFwdPrimitives(Array(nativeData), InferencePhase)
    conv1.forward(reorderedInput)

    {
      val len = outputShape.product
      val output = new Array[Byte](len)
      Memory.CopyPtr2ByteArray(conv1.output.asInstanceOf[DnnTensor[Byte]].storageAddress(),
        0, output, 0, len, 1)
      output.foreach(println)
    }

    val heapData2 = HeapData(outputShape, Memory.Format.nchw, DataType.F32)
    val reorder2 = ReorderMemory(heapData2)
    reorder2.setRuntime(runtime)
    reorder2.initFwdPrimitives(conv1.outputFormats(), InferencePhase)
    reorder2.forward(conv1.output)

    println(reorder2.output)

  }

//  "conv quantization" should "work correctly" in {
//    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
//    RNG.setSeed(1)
//    val inputShape = Array(1, 2, 12, 12)
//    val outputShape = Array(1, 8)
//    val model = Sequential()
//      .add(Input(inputShape, Memory.Format.nchw))
//      .add(SpatialConvolution(2, 4, 5, 5).setName("conv2"))
//      .add(ReLU()).setName("relu")
//      .add(MaxPooling(2, 2, 2, 2).setName("pool2"))
//      .add(Linear(4 * 4 * 4, 8).setName("ip1"))
//      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nc)))
//    model.evaluate()
//
//    val input = Tensor[Float](inputShape).rand(-100, 100)
//    model.compile(InferencePhase)
//    println(model.forward(input))
//    val quantized = model.quantize(Array(input))
//
//    quantized.forward(input)
//    println(quantized.output)
//    System.clearProperty("bigdl.mkldnn.fusion.convrelu")
//  }

  "unsigned input quantization" should "work correctly" in {
    RNG.setSeed(1)

    val inputShape = Array(1, 2, 12, 12)
    val outputShape = Array(1, 4, 8, 8)

    val initBias = Tensor[Float](4).fill(1.0f)

    val model = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(SpatialConvolution(2, 4, 5, 5, initBias = initBias)).setName("conv2")
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    model.evaluate()
    val input = Tensor[Float](inputShape).rand(-128, 127)
    model.compile(InferencePhase)
    println(model.forward(input))
    model.generateScalesWithMask(input, Int8ScalesAndMask.SINGLE_SCALE, 2)

    val quantized = model.quantize()
    quantized.forward(input)
    println(quantized.output)
  }

  "generate the convolution scales" should "work correctly" in {
    val inputShape = Array(1, 1, 2, 2)
    val outputShape = Array(1, 2, 1, 1)

    val inputData = Array[Float](-100, 12, 14, 67)
    val input = Tensor[Float](inputData, inputShape)

    val initWeight = Tensor[Float](
      Array[Float](2, -16, 48, 23, -37, 24, -11, -13),
      Array(2, 1, 2, 2))
    val initBias = Tensor[Float](Array[Float](-10, 3), Array(2))

    val conv = SpatialConvolution(1, 2, 2, 2, initWeight = initWeight, initBias = initBias)

    val seq = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(conv)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    seq.compile(InferencePhase)
    seq.forward(input)

    seq.generateScalesWithMask(input, Int8ScalesAndMask.SINGLE_SCALE, 2)

    val outputFP32Model = seq.forward(input).toTensor.clone()

    val quantizedModel = seq.quantize()

    val outputInt8Model = quantizedModel.forward(input).toTensor.clone()

    println(outputFP32Model)
    println(outputInt8Model)
  }

  "generate the convolution scales with random" should "work correctly" in {
    RNG.setSeed(1)
    val inputShape = Array(1, 1, 2, 2)
    val outputShape = Array(1, 2, 1, 1)

    val inputData = Array[Float](-100, 12, 14, 67)
    val input = Tensor[Float](inputShape).rand(-100, 100)

    val initWeight = Tensor[Float](Array(2, 1, 2, 2)).rand(-10, 10)
    val initBias = Tensor[Float](Array(2)).rand(-1, 1)

    val conv = SpatialConvolution(1, 2, 2, 2)

    val seq = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(conv)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    seq.compile(InferencePhase)
    seq.forward(input)

    seq.generateScalesWithMask(input, Int8ScalesAndMask.SINGLE_SCALE, 2)

    val outputFP32Model = seq.forward(input).toTensor.clone()

    val quantizedModel = seq.quantize()

    val outputInt8Model = quantizedModel.forward(input).toTensor.clone()

    println(outputFP32Model)
    println(outputInt8Model)

    outputFP32Model.storage().array().zip(outputInt8Model.storage().array()).foreach { x =>
      (Math.abs(x._1 - x._2) / Math.max(Math.abs(x._1), Math.abs(x._2)) <= 1e-1) should be (true)
    }
  }

  "load lenet5 and quantize" should "work correctly" in {
    System.setProperty("bigdl.mkldnn.fusion.convbn", "true")
    val input = Tensor[Float](100, 1, 28, 28).rand(-0.42, 2.8)
    val model = Module.load[Float]("/tmp/lenet/current/model.1801")

    model.evaluate()
    model.asInstanceOf[Sequential].compile(InferencePhase)
    model.forward(input)
    model.asInstanceOf[Sequential].generateScalesWithMask(input, 0, 2)
    model.release()

    val quant = model.quantize()
    quant.forward(input)
    System.clearProperty("bigdl.mkldnn.fusion.convbn")
  }

  def prototxt(inputShape: Array[Int], name: String,
    nOutput: Int, kernel: Int, pad: Int, stride: Int): String = {
      s"""
         |name: "conv-simple"
         |force_backward: true
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  include {
         |    phase: TRAIN
         |  }
         |  dummy_data_param {
         |    data_filler {
         |      type: "xavier"
         |    }
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "conv"
         |  name: "$name"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: $nOutput
         |    kernel_size: $kernel
         |    pad: $pad
         |    stride: $stride
         |    weight_filler {
         |      type: "msra"
         |      variance_norm: FAN_OUT
         |    }
         |    bias_filler {
         |      type: "gaussian"
         |    }
         |  }
         |}
       """.stripMargin
  }

  def prototxt2(inputShape: Array[Int], name: String,
    nOutput: Int, kernel: Int, pad: Int, stride: Int): String = {
    s"""
       |name: "conv-simple"
       |force_backward: true
       |layer {
       |  name: "data"
       |  type: "DummyData"
       |  top: "data"
       |  include {
       |    phase: TRAIN
       |  }
       |  dummy_data_param {
       |    data_filler {
       |      type: "uniform"
       |      min: -1000
       |      max: 1000
       |    }
       |    shape: { ${shape2Dim(inputShape)} }
       |  }
       |}
       |
         |layer {
       |  bottom: "data"
       |  top: "conv"
       |  name: "$name"
       |  type: "Convolution"
       |  convolution_param {
       |    num_output: $nOutput
       |    kernel_size: $kernel
       |    pad: $pad
       |    stride: $stride
       |    weight_filler {
       |      type: "msra"
       |      variance_norm: FAN_OUT
       |    }
       |    bias_filler {
       |      type: "gaussian"
       |    }
       |  }
       |}
       """.stripMargin
  }

  def normal(src: Tensor[Float], outputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = src.size().length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.oi
      case 4 => Memory.Format.oihw
    }

    if (defaultFormat != outputFormat.layout) {
      val inputFormat = HeapData(src.size(), defaultFormat)
      val reorder = ReorderMemory(inputFormat, outputFormat, null, null)
      reorder.setRuntime(new MklDnnRuntime)
      reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
      reorder.updateOutput(src).toTensor
    } else {
      src
    }
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}
