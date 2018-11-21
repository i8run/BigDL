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

import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor._

import scala.collection.mutable.ArrayBuffer

class Linear(
  val inputSize: Int,
  val outputSize: Int,
  var wRegularizer: Regularizer[Float] = null,
  var bRegularizer: Regularizer[Float] = null,
  private val initWeight: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null,
  private val initGradWeight: Tensor[Float] = null,
  private val initGradBias: Tensor[Float] = null) extends MklDnnLayer with Initializable {

  private[mkldnn] val weight: TensorMMap = new TensorMMap(Array(outputSize, inputSize))
  private[mkldnn] val bias: TensorMMap = new TensorMMap(Array(outputSize))
  private[mkldnn] val gradWeight: TensorMMap = new TensorMMap(Array(outputSize, inputSize))
  private[mkldnn] val gradBias: TensorMMap = new TensorMMap(Array(outputSize))

  @transient private var forwardPrimDesc: Long = 0L

  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradInputMemoryPrimitives: Array[Long] = _
  @transient private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradWMemoryPrimitives: Array[Long] = _
  @transient private var updateGradWTensors: Array[Tensor[Float]] = _

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight.dense, VariableFormat.OUT_IN)
    } else {
      weight.dense.copy(initWeight)
    }

    if (initBias == null) {
      biasInitMethod.init(bias.dense, VariableFormat.ONE_D)
    } else {
      bias.dense.copy(initBias)
    }
  }

  @transient var needQuantize = false

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    needQuantize = false
    val (weightShape, weightLayout) = inputs(0).shape.length match {
      case 4 =>
        (Array(weight.size(1)) ++ inputs(0).shape.slice(1, 4),
          Memory.Format.oihw)
      case 2 => (weight.size(), Memory.Format.nc)
      case 1 => (weight.size(), Memory.Format.x)
    }

    val inputShape = inputs(0).shape
    require(inputs(0).shape.length > 1, s"mkldnn linear unspported input dimension")

    val outputShape = Array(inputs(0).shape(0), outputSize)

    MklDnn.MemoryDescInit(inputShape.length, inputShape,
      DataType.F32, Memory.Format.any)

    val inputDataType = if (needQuantize) DataType.U8 else DataType.F32
    val weightDataType = if (needQuantize) DataType.S8 else DataType.F32
    val biasDataType = if (needQuantize) DataType.S32 else DataType.F32
    val outputDataType = if (needQuantize) DataType.S8 else DataType.F32

    val src = NativeData(inputShape, Memory.Format.any, inputDataType)
    val wei = NativeData(weightShape, Memory.Format.any, weightDataType)
    val bis = NativeData(bias.size(), Memory.Format.x, biasDataType)
    val dst = NativeData(outputShape, Memory.Format.any, outputDataType)

    val desc = MklDnn.LinearForwardDescInit(
      PropKind.ForwardScoring,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      dst.getMemoryDescription())

    forwardPrimDesc = if (needQuantize) {
      val attr = MklDnn.CreateAttr()

      // create output scales for s8/u8 output
      if (_outputFormats != null) {
        MklDnn.AttrSetIntOutputRoundMode(attr, 1)
        _outputFormats(0).setMask(scalesOfOutput.mask)
        _outputFormats(0).setScales(scalesOfOutput.scales(0))
        _inputFormats(0).setMask(scalesOfInput.mask)
        _inputFormats(0).setScales(scalesOfInput.scales(0))

        require(_outputFormats(0).scales != null)
        require(_outputFormats(0).scales.length == 1)
        require(_inputFormats(0).scales != null && !_inputFormats(0).scales.isEmpty)
        require(_inputFormats(0).scales.length == 1)
        require(weight.memoryData().scales != null)

        val scaleOut = _outputFormats(0).scales(0)
        val scaleIn = _inputFormats(0).scales(0)
        //        val scales = weight.memoryData().scales.map(w => scaleOut / (scaleIn * w))

        val scales = weight.memoryData().scales.map(w => (scaleIn * w) / (scaleOut * 127))
        MklDnn.AttrSetOutputScales(attr, scales.length, 2, scales)
      }
      MklDnn.PrimitiveDescCreateV2(desc, attr, runtime.engine, 0)
      // TODO we should destroy these ops
    } else {
      MklDnn.PrimitiveDescCreate(desc, runtime.engine, 0)
    }

    val List(realSrc, realWei, realDst) = List(Query.SrcPd, Query.WeightsPd, Query.DstPd).map {x =>
      MemoryData.operationWant(forwardPrimDesc, x)
    }

    require(weight.size().product == realWei.shape.product,
      s"${getName} weight shape is not correct.")

    if (weight.isMemoryDataSet()) {
      weight.setMemoryData(weight.heapData, realWei, runtime)
      bias.setMemoryData(bias.heapData, bis, runtime)
    } else {
      weight.setMemoryData(HeapData(weightShape, weightLayout), realWei, runtime)
      bias.setMemoryData(HeapData(bis.shape, Memory.Format.x), bis, runtime)
    }

    weight.sync()
    bias.sync()

    val srcs = Array(realSrc.getPrimitive(runtime), realWei.getPrimitive(runtime),
      bis.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDst.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)
    output = initTensor(realDst)

    updateOutputTensors = null

    // quantize weight from fp32 to int8
//    if (scalesOfInput.scales.nonEmpty) {
//      realSrc.setMask(scalesOfInput.mask)
//      realSrc.setScales(scalesOfInput.scales(0))
//    }
//
//    if (scalesOfOutput.scales.nonEmpty) {
//      realDst.setMask(scalesOfOutput.mask)
//      realDst.setScales(scalesOfOutput.scales(0))
//    }


    _inputFormats = Array(realSrc)
    _outputFormats = Array(realDst)
    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(weight.native)
      buffer.append(bias.native)
      buffer.append(output.asInstanceOf[Tensor[Float]])
      updateOutputTensors = buffer.toArray
    }

    updateWithNewTensor(updateOutputTensors, 0, input)

    if (isTraining()) {
      weight.sync()
      bias.sync()
    }

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors.asInstanceOf[Array[Tensor[_]]])

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    val weightShape = inputFormats()(0).shape.length match {
      case 4 => Array(weight.size(1)) ++ inputFormats()(0).shape.slice(1, 4)
      case _ => weight.size()
    }

    val inputShape = inputFormats()(0).shape

    val outputShape = Array(inputFormats()(0).shape(0), outputSize)

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.LinearBackwardDataDescInit(
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      grad(0).getMemoryDescription())
    val backwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realDiffSrc, realWei, realDiffDst) =
      List(Query.DiffSrcPd, Query.WeightsPd, Query.DiffDstPd).map { x =>
        MemoryData.operationWant(backwardPrimDesc, x)
      }

    val srcs = Array(realDiffDst.getPrimitive(runtime), realWei.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDiffSrc.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(backwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradInputMemoryPrimitives = srcs ++ dsts
    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(realDiffSrc)

    _gradInputFormats = Array(realDiffSrc)
    _gradOutputFormats = Array(realDiffDst)
    (_gradOutputFormats, _gradInputFormats)
  }

  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData],
    phase: Phase): Array[MemoryData] = {
    val (weightShape, weightLayout) = inputFormats()(0).shape.length match {
      case 4 =>
        (Array(weight.size(1)) ++ inputFormats()(0).shape.slice(1, 4),
          Memory.Format.oihw)
      case 2 => (weight.size(), Memory.Format.nc)
      case 1 => (weight.size(), Memory.Format.x)
    }

    val inputShape = inputFormats()(0).shape

    val outputShape = Array(inputFormats()(0).shape(0), outputSize)


    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.LinearBackwardWeightsDescInit(
      src.getMemoryDescription(), wei.getMemoryDescription(), bis.getMemoryDescription(),
      dst.getMemoryDescription())
    val gradWeightPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realWei, realDiffDst) = List(Query.DiffWeightsPd, Query.DiffDstPd).map { x =>
      MemoryData.operationWant(gradWeightPrimDesc, x)
    }

    gradWeight.setMemoryData(realWei, HeapData(weightShape, weightLayout),
      runtime)
    gradBias.setMemoryData(bis, HeapData(bis.shape, Memory.Format.x), runtime)

    gradWeight.zero()
    gradBias.zero()

    val srcs = Array(inputFormats()(0).getPrimitive(runtime), realDiffDst.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realWei.getPrimitive(runtime), bis.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(gradWeightPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradWMemoryPrimitives = srcs ++ dsts
    accGradientPrimitives = Array(primitive)

    _gradOutputFormatsForWeight = Array(realDiffDst)
    (_gradOutputFormatsForWeight)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(weight.native)
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      updateGradInputTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 0, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
      updateGradInputPrimitives.length, updateGradInputMemoryPrimitives,
      updateGradInputTensors.asInstanceOf[Array[Tensor[_]]])

    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    if (updateGradWTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(gradWeight.native)
      buffer.append(gradBias.native)
      updateGradWTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradWTensors, 0, input)
    updateWithNewTensor(updateGradWTensors, 1, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, accGradientPrimitives,
      accGradientPrimitives.length, updateGradWMemoryPrimitives,
      updateGradWTensors.asInstanceOf[Array[Tensor[_]]])

    gradWeight.sync()
    gradBias.sync()

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight.dense, gradWeight.dense, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias.dense, gradBias.dense, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weight.dense, bias.dense), Array(gradWeight.dense, gradBias.dense))
  }

  override def zeroGradParameters(): Unit = {
  }

  override def release(): Unit = {
    super.release()
    List(weight, bias, gradWeight, gradBias).foreach(_.release())
  }

  private[bigdl] def updateWeightsScales(): Unit = {
    require(weight.memoryData() != null)
    require(bias.memoryData() != null)

    def scalesOf(tensor: Tensor[Float], indexes: Array[Int]): Tensor[Float] = {
      if (indexes.length == 1) {
        tensor.max(indexes.head)._1
      } else {
        scalesOf(tensor.max(indexes.head)._1, indexes.drop(1))
      }
    }

    {
      // reorder the weights
      val format = if (weight.size().length == 4) {
        Memory.Format.oihw
      } else {
        Memory.Format.nc
      }

      val result = weight.dense

      val (mask, scales) = weight.size().length match {
        case 2 =>
          val mask = math.pow(2, 0).toInt
          val scales = scalesOf(result, Array(2)).storage().array().map(x => x)
          (mask, scales)
        case 4 =>
          val mask = math.pow(2, 0).toInt
          val scales = scalesOf(result, Array(2, 3, 4)).storage().array().map(x => x)
          (mask, scales)
        case _ =>
          throw new UnsupportedOperationException
      }

      weight.memoryData().setMask(mask)
      weight.memoryData().setScales(scales)
    }

    {
      bias.memoryData().setMask(0)
      bias.memoryData().setScales(Array(scalesOfOutput.scales(0)(0) * bias.dense.max()))
    }

    //    {
    // FIXME at last, the output
    //      val scales = inputFormats()(0).scales.zip(weight.memoryData().scales).map(x => x._1 * x._2)
    //      outputFormats()(0).setScales(scales)
    //    }
  }
}

object Linear {
  def apply(
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    wRegularizer: Regularizer[Float] = null,
    bRegularizer: Regularizer[Float] = null,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null): Linear = {
    new Linear(inputSize, outputSize, wRegularizer,
      bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
