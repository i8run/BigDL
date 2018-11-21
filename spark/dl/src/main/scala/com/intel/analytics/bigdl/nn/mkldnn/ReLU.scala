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
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase

class ReLU(value: Float = 0.0f) extends MklDnnLayer {
  private val UNDEFINED: Long = 0

  @transient private var fwdPrimDesc: Long = UNDEFINED

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = singleNativeData(inputs)
    val description = MklDnn.EltwiseForwardDescInit(
      PropKind.Forward, AlgKind.EltwiseRelu, _inputFormats(0).getMemoryDescription(), value, 0)
    fwdPrimDesc = MklDnn.PrimitiveDescCreate(description, runtime.engine, 0L)
    _outputFormats = Array(MemoryData.primitiveOutput(fwdPrimDesc))
    updateOutputPrimitives = Array(
      MklDnn.PrimitiveCreate2(fwdPrimDesc,
        Array(_inputFormats(0).getPrimitive(runtime)), Array(0), _inputFormats.length,
        _outputFormats.map(_.getPrimitive(runtime)), _outputFormats.length)
    )
    if (inputs(0).scales.nonEmpty) {
      _outputFormats(0).setMask(inputs(0).mask)
      _outputFormats(0).setScales(inputs(0).scales)
    }
    output = initTensor(_outputFormats(0))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats
    val description = MklDnn.EltwiseBackwardDescInit(AlgKind.EltwiseRelu,
      _gradOutputFormats(0).getMemoryDescription(), _inputFormats(0).getMemoryDescription(),
      value, 0)
    require(fwdPrimDesc != UNDEFINED, "You should call initFwdPrimitives first")
    val primDesc = MklDnn.PrimitiveDescCreate(description, runtime.engine, fwdPrimDesc)
    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffSrcPd))
    updateGradInputPrimitives = Array(
      MklDnn.PrimitiveCreate2(primDesc, Array(_inputFormats(0),
        _gradOutputFormats(0)).map(_.getPrimitive(runtime)), Array(0), 2,
        _gradInputFormats.map(_.getPrimitive(runtime)), _gradInputFormats.length))
    gradInput = initTensor(_gradInputFormats(0))
    (_gradOutputFormats, _gradInputFormats)
  }

  override def generateInAndOutScales(input: Activity, inAndOutMask: Int): Unit = {
    val defaultInput = HeapData(inputFormats()(0).shape, Memory.Format.nchw)
    val defaultOutput = HeapData(outputFormats()(0).shape, Memory.Format.nchw)

    val inputReorder = ReorderMemory(defaultInput)
    inputReorder.setRuntime(runtime)
    inputReorder.initFwdPrimitives(inputFormats(), InferencePhase)

    val outputReorder = ReorderMemory(defaultOutput)
    outputReorder.setRuntime(runtime)
    outputReorder.initFwdPrimitives(outputFormats(), InferencePhase)

    val defaultInputData = inputReorder.forward(input).toTensor[Float]
    val maxIn = defaultInputData.abs().max()

    scalesOfInput.update(Array(maxIn), 0)

    val maxOut = outputReorder.forward(output).toTensor[Float].abs().max

    scalesOfOutput.update(Array(maxOut), 0)

    inputReorder.release()
    outputReorder.release()
  }
}

object ReLU {
  def apply(value: Float = 0.0f): ReLU = new ReLU(value)
}
