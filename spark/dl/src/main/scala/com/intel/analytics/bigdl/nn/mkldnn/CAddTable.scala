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

import com.intel.analytics.bigdl.mkl.{DataType, Memory, MklDnn}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.utils.T

class CAddTable extends MklDnnLayer {
  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = nativeData(inputs)
    val shape = inputs(0).shape.clone()
    for(i <- 1 until inputs.length) {
      require(shape.length == inputs(i).shape.length, "dimension not match")
      require(inputs(i).layout == inputs(0).layout, "layout not match")
      for(j <- 0 until shape.length) {
        require(shape(j) == inputs(i).shape(j), "size not match")
      }
    }

    val outputMD = MklDnn.MemoryDescInit(shape.length, shape, inputs(0).dataType, Memory.Format.any)

    val scales = inputs.map(x => if (x.scales.nonEmpty) x.scales(0) / inputs(0).scales(0) else 1.0f)

    val pd = MklDnn.SumPrimitiveDescCreate(outputMD, inputs.length, scales,
      inputs.map(_.getPrimitiveDescription(runtime)))
    _outputFormats = Array(MemoryData.primitiveOutput(pd))
    _outputFormats(0).setMask(0)
    _outputFormats(0).setScales(inputs(0).scales)
    updateOutputPrimitives = Array(MklDnn.PrimitiveCreate2(pd,
      _inputFormats.map(_.getPrimitive(runtime)), new Array[Int](inputs.length),
      _inputFormats.length, _outputFormats.map(_.getPrimitive(runtime)), 1))
    output = initTensor(_outputFormats(0))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = grad
    _gradOutputFormatsForWeight = grad
    _gradInputFormats = new Array[MemoryData](_inputFormats.length).map(a => grad(0))
    gradInput = T()
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    require(gradOutput.isTensor, "gradOutput should be a tensor")
    val _gradInput = gradInput.toTable
    var i = 1
    while(i <= _inputFormats.length) {
      _gradInput(i) = gradOutput
      i += 1
    }
    gradInput
  }

  override def generateInAndOutScales(input: Activity, inAndOutMask: Int): Unit = {
    val defaultOutput = HeapData(outputFormats()(0).shape, Memory.Format.nchw)

    val outputReorder = ReorderMemory(defaultOutput)
    outputReorder.setRuntime(runtime)
    outputReorder.initFwdPrimitives(outputFormats(), InferencePhase)

    val maxOut = outputReorder.forward(output).toTensor[Float].abs().max

    scalesOfOutput.update(Array(maxOut), 0)

    outputReorder.release()
  }
}

object CAddTable {
  def apply(): CAddTable = new CAddTable()
}
