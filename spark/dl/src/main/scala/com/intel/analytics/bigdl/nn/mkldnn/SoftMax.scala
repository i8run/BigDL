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

import com.intel.analytics.bigdl.mkl.{DataType, Engine, Memory, MklDnn, PropKind, Stream => DnnStream}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.{DenseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class SoftMax[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val nnSoftMax = nn.SoftMax()
  var hasForwarded = false

  // TODO should be refactored to on module
  val engine: Long = Engine.Create(Engine.Kind.Cpu, 0)

  var forwardStream = 0L

  var forwardPrim = 0L
  @transient var userSrcMemoryPrim = 0L
  @transient var userDstMemoryPrim = 0L

  private def initDataMemory(dim: Int, dims: Array[Int], format: Int,
    dataType: Int, engine: Long, tensor: Tensor[T]): Long = {
    val primMd = MklDnn.MemoryDescInit(dim, dims, dataType, format)
    val userPd = MklDnn.MemoryPrimitiveDescCreate(primMd, engine)
    val memory = MklDnn.PrimitiveCreate0(userPd)

    MklDnn.PrimitiveDescDestroy(userPd)
    memory
  }

  private def setHandle(tensor: Tensor[T], primitive: Long): Unit = {
    val data = tensor.storage().array().asInstanceOf[Array[Float]]
    val offset = tensor.storageOffset() - 1
    MklDnn.MemorySetDataHandle(primitive, data, offset)
  }

  private def releaseHandles(input: Tensor[T], ptr: Long): Unit = {
    MklDnn.MemoryReleaseDataHandle(
      input.storage().array().asInstanceOf[Array[Float]], ptr)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    if (forwardPrim == 0L) {
      val (format, ndim, dims, axis) = input.dim() match {
        case 1 => (Memory.Format.x, 1, input.size(), 0)
        case 2 => (Memory.Format.nc, 2, input.size(), 1)
        case 3 => (Memory.Format.nchw, 4, Array(1) ++ input.size(), 1)
        case 4 => (Memory.Format.nchw, 4, input.size(), 1)
        case _ => throw new UnsupportedOperationException(
          s"1 <= input.nDimension() && input.nDimension() <= 4, 1D, 2D, 3D or 4D tensor expected " +
            s"input dimension ${input.nDimension()}")
      }

      val srcMemDesc = MklDnn.MemoryDescInit(ndim, dims, DataType.F32, format)

      // TODO the axis should depend on the input dimension
      // it's always the first dim. Is it correct?
      val opDesc = MklDnn.SoftMaxForwardDescInit(PropKind.ForwardInference,
        srcMemDesc, axis)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)

      userSrcMemoryPrim = initDataMemory(ndim, dims, format, DataType.F32, engine, input)
      userDstMemoryPrim = initDataMemory(ndim, dims, format, DataType.F32, engine, output)

      val srcs = Array(userSrcMemoryPrim)
      val indexes = Array(0)
      val dsts = Array(userDstMemoryPrim)

      forwardPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)
    }

    if (forwardStream == 0L) {
      forwardStream = DnnStream.Create(DnnStream.Kind.Eager)
    }

    setHandle(input, userSrcMemoryPrim)
    setHandle(output, userDstMemoryPrim)

    DnnStream.Submit(forwardStream, 1, Array(forwardPrim))

    releaseHandles(input, userSrcMemoryPrim)
    releaseHandles(output, userDstMemoryPrim)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!hasForwarded) {
      nnSoftMax.forward(input)
      hasForwarded = true
    }

    gradInput = nnSoftMax.backward(input, gradOutput)
    gradInput
  }
}

object SoftMax {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): SoftMax[T] = {
    new SoftMax[T]()
  }
}

class RefactorSoftMax() extends MklDnnLayer {
  val nnSoftMax = nn.SoftMax[Float]()

  var updateOutputTensors: Array[Tensor[Float]] = _
  var updateOutputMemoryPrimitives: Array[Long] = _

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    phase match {
      case TrainingPhase => (inputs, inputs) // do nothing, because mkl dnn doesn't support training
      case InferencePhase =>
        val axis = inputs(0).shape.length match {
          case 1 => 0
          case 2 => 1
//          case 3 => 1 // TODO should support this?
          case 4 => 1
          case _ => throw new UnsupportedOperationException("1D, 2D, or 4D tensor expected")
        }

        _inputFormats = singleNativeData(inputs)
        val desc = MklDnn.SoftMaxForwardDescInit(PropKind.ForwardInference,
          inputFormats()(0).getMemoryDescription(), axis)
        val forwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, 0L)

        _outputFormats = Array(MemoryData.primitiveOutput(forwardPrimDesc))

        val srcs = Array(inputs(0).getPrimitive(runtime))
        val indexes = Array(0)
        val dsts = Array(_outputFormats(0).getPrimitive(runtime))

        val primitive = MklDnn.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length, dsts,
          dsts.length)

        updateOutputPrimitives = Array(primitive)
        updateOutputMemoryPrimitives = srcs ++ dsts

        output = initTensor(_outputFormats(0))

        (_inputFormats, _outputFormats)
      case _ => throw new UnsupportedOperationException
    }
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    (grad, grad)
  }

  override def updateOutput(input: Activity): Activity = {
      if (this.isTraining()) {
        nnSoftMax.forward(input)
        output = nnSoftMax.output
      } else {
        if (updateOutputTensors == null) {
          val buffer = new ArrayBuffer[Tensor[Float]]()
          buffer.append(input.asInstanceOf[Tensor[Float]])
          buffer.append(output.asInstanceOf[Tensor[Float]])
          updateOutputTensors = buffer.toArray
        }

        input.toTensor[Float].getTensorType match {
          case DenseType => updateOutputTensors(0) = input.toTensor
          case _ =>
        }

        MklDnnOps.streamSubmit(runtime.stream, 1,
          updateOutputPrimitives,
          updateOutputPrimitives.length,
          updateOutputMemoryPrimitives, updateOutputTensors)
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = nnSoftMax.backward(input, gradOutput)
    gradInput
  }
}

object RefactorSoftMax{
  def apply(): RefactorSoftMax = {
    new RefactorSoftMax()
  }
}
