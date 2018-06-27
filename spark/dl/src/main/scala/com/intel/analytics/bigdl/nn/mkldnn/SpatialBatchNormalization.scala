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

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.mkl.{AlgKind, DataType, Memory, MklDnn, PropKind, Query, Stream => DnnStream}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable, TensorModule}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{Ones, VariableFormat, Zeros}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.annotation.strictfp
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class SpatialBatchNormalization[T: ClassTag](
  val nOutput: Int,
  val eps: Double = 1e-5,
  val momentum: Double = 0.1,
  val affine: Boolean = true,
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {
  var relu = false // use bn->relu
  val mean: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))
  val variance: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))

  val all = createParams(initWeight, initBias)
  val gradAll = createParams(initGradWeight, initGradBias)
  val diffAll = MklDnnTensor[T](all.size())
  val prvAll = MklDnnTensor[T](all.size())

  val allWeight = all.view(Array(2, nOutput))
  val weight = allWeight.select(1, 1)
  val bias = allWeight.select(1, 2)

  val runningMean: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))
  val runningVar: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))
  // this two factors come from Caffe, which are different with BigDL
  var scaleFactor: Float = 0.0f
  var useGlobalStats: Boolean = false

  Memory.Zero(runningMean.ptr, runningMean.nElement(), 4)
  Memory.Zero(runningVar.ptr, runningVar.nElement(), 4)

  @transient var engine = 0L
  @transient var stream = 0L

  @transient var forwardTrainPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var forwardInferPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var forwardTrainReorderPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var forwardInferReorderPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var backwardPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var backwardReorderPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var forwardTrainPrimDesc = 0L
  @transient var forwardInferPrimDesc = 0L

  {
    val wInit = Ones // RandomUniform(0, 1)
    val bInit = Zeros
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    val weightAndBias = all.view(Array(2, nOutput))
    if (initWeight != null) {
      require(initWeight.size(1) == nOutput)
      weightAndBias.select(1, 1).copy(initWeight)
    } else {
      val weight = weightAndBias.select(1, 1)
      weightInitMethod.init(weight, VariableFormat.ONE_D)
    }

    if (initBias != null) {
      require(initBias.size(1) == nOutput)
      weightAndBias.select(1, 2).copy(initBias)
    } else {
      val bias = weightAndBias.select(1, 2)
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }

    zeroGradParameters()
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    forwardTrainPrims = ArrayBuffer.empty
    forwardInferPrims = ArrayBuffer.empty
    forwardTrainReorderPrims = ArrayBuffer.empty
    forwardInferReorderPrims = ArrayBuffer.empty
    backwardPrims = ArrayBuffer.empty
    backwardReorderPrims = ArrayBuffer.empty

    // initialize runingMean and running Var
    for (tensor <- List(runningMean, runningVar)) {
      tensor.syncFromHeap()
    }
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    // first we save the runningMean and runningVar
    for (tensor <- List(runningMean, runningVar)) {
      tensor.syncToHeap()
    }

    out.defaultWriteObject()
  }

  var _shouldConvert: Boolean = false
  def shouldConvert: Boolean = _shouldConvert
  def setShouldConvert(v: Boolean): this.type = {
    _shouldConvert = v
    this
  }

  object OpPrim {
    val input, output, weightAndBias, mean, variance,
        diffInput, diffOutput, diffWeightAndBias = new MemoryPrimitive[T]()
  }

  private def init1(primDesc: Long): Long = {
    MklDnn.PrimitiveCreate0(primDesc)
  }

  private def init4(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Long = {
    // TODO refactor for linear
    val (dim, size) = if (tensor.dim() == 1 && (format == Memory.Format.nc ||
      format == Memory.Format.oi)) {
      (2, Array(1) ++ tensor.size())
    } else if (tensor.dim() == 2 && (format == Memory.Format.oihw)) {
      (4, tensor.size() ++ Array(1, 1))
    } else {
      (tensor.dim(), tensor.size())
    }

    val desc = MklDnn.MemoryDescInit(dim, size, dataType, format)
    val primDesc = MklDnn.MemoryPrimitiveDescCreate(desc, engine)
    val primitive = MklDnn.PrimitiveCreate0(primDesc)

    MklDnn.PrimitiveDescDestroy(primDesc)
    primitive
  }

  def initUser(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Long = {
    val primDesc = tensor.getPrimitiveDesc()
    val primitive = if (primDesc != 0L) { // if the tensor comes from mkldnn layer
      init1(primDesc)
    } else {
      init4(tensor, dataType, format, engine)
    }
    primitive
  }

  def initInternal(userPrim: Long, layerPrimDesc: Long, queryType: Int,
    userToPrim: Boolean = true): (Long, Long) = {
    val primDescFromLayer = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    val res = MklDnnOps.prepareReorder(userPrim, primDescFromLayer, userToPrim)
    val memoryPrimitive = res._2
    val reorderPrimitive = res._1
    (memoryPrimitive, reorderPrimitive)
  }

  def initUser(tensor: Tensor[T], layerPrimDesc: Long, queryType: Int, index: Int): Long = {
    val primDesc = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    tensor.setPrimitiveDesc(primDesc)
    val primitive = MklDnn.PrimitiveCreate0(primDesc)
    primitive
  }

  // TODO train and inference mode

  @transient var internalInput, internalOutput: MklDnnTensor[T] = _

  var defaultFormat = Memory.Format.nchw

  private def toMklDnnTensor(t: Tensor[T]): MklDnnTensor[T] = t.asInstanceOf[MklDnnTensor[T]]

  @transient var inputUserPrim = 0L
  @transient var inputTrainReorderMemoryPrim = 0L
  @transient var inputTrainReorderPrim = 0L
  @transient var inputInferReorderMemoryPrim = 0L
  @transient var inputInferReorderPrim = 0L
  @transient var outputTrainUserPrim = 0L
  @transient var outputInferUserPrim = 0L
  @transient var weightAndBiasUserPrim = 0L
  @transient var meanUserPrim = 0L
  @transient var varianceUserPrim = 0L
  @transient var previousSize: Array[Int] = _
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val s1 = System.nanoTime()
    if (previousSize == null) {
      previousSize = input.size()
    } else if (previousSize.deep != input.size().deep) {
      previousSize = input.size()
      for (i <- forwardTrainPrims ++ forwardInferPrims ++ backwardPrims
        ++ forwardTrainReorderPrims ++ forwardInferReorderPrims ++ backwardReorderPrims) {
        MklDnn.PrimitiveDestroy(i)
      }
      forwardTrainPrims = ArrayBuffer.empty
      forwardInferPrims = ArrayBuffer.empty
      backwardPrims = ArrayBuffer.empty
      forwardTrainReorderPrims = ArrayBuffer.empty
      forwardInferReorderPrims = ArrayBuffer.empty
      backwardReorderPrims = ArrayBuffer.empty
    }

    // we create train primitive and test primitive all together.
    if (forwardTrainPrims.isEmpty) {
      if (output.getTensorType == MklDnnType) {
        toMklDnnTensor(output).release()
      }
      output = MklDnnTensor[T](input.size())

      engine = this.getDnnEngine(0)
      stream = this.getStream()

      val srcMemDesc = if (input.getPrimitiveDesc() == 0L) {
        MklDnn.MemoryDescInit(input.dim(), input.size(), DataType.F32, defaultFormat)
      } else {
        MklDnnOps.primitiveDescQueryMemory(input.getPrimitiveDesc())
      }

      val bnTrainDesc =
        MklDnn.BatchNormForwardDescInit(PropKind.Forward,
          srcMemDesc, eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      // we always use the weight and bias / scale and offset. So the flags should be combined
      // with use_scaleshift and use_global_stats.
      val bnInferDesc =
        MklDnn.BatchNormForwardDescInit(PropKind.ForwardInference,
          srcMemDesc, eps.toFloat,
          MklDnn.BatchNormFlag.mkldnn_use_global_stats | MklDnn.BatchNormFlag.mkldnn_use_scaleshift)

      val opTrainPrimDesc = MklDnn.PrimitiveDescCreate(bnTrainDesc, engine, 0)
      val opInferPrimDesc = if (relu) {
        val postOps = MklDnn.CreatePostOps()
        MklDnn.PostOpsAppendEltwise(postOps, 1.0f, AlgKind.EltwiseRelu, 0.0f, 0.0f)
        val attr = MklDnn.CreateAttr()
        MklDnn.AttrSetPostOps(attr, postOps)
        MklDnn.PrimitiveDescCreateV2(bnInferDesc, attr, engine, 0)
        // TODO we should destroy these ops
      } else {
        MklDnn.PrimitiveDescCreate(bnInferDesc, engine, 0)
      }

      forwardTrainPrimDesc = opTrainPrimDesc
      forwardInferPrimDesc = opInferPrimDesc

      val dataFormat = defaultFormat
      val paramsFormat = Memory.Format.x
      val dataType = DataType.F32

      inputUserPrim = initUser(input, dataType, dataFormat, engine)
      val i1t = initInternal(inputUserPrim, opTrainPrimDesc, Query.SrcPd)
      inputTrainReorderMemoryPrim = i1t._1
      inputTrainReorderPrim = i1t._2
      val i1i = initInternal(inputUserPrim, opInferPrimDesc, Query.SrcPd)
      inputInferReorderMemoryPrim = i1i._1
      inputInferReorderPrim = i1i._2
      outputTrainUserPrim = initUser(output, opTrainPrimDesc, Query.DstPd, 0)
      outputInferUserPrim = initUser(output, opInferPrimDesc, Query.DstPd, 0)

      // because they're 1-d, so we need not to initialize it.
      weightAndBiasUserPrim = initUser(all, dataType, paramsFormat, engine)
      meanUserPrim = initUser(mean, dataType, paramsFormat, engine)
      varianceUserPrim = initUser(variance, dataType, paramsFormat, engine)

      val inputTrainMemoryPrim = if (inputTrainReorderPrim != 0) {
        forwardTrainReorderPrims += inputTrainReorderPrim
        inputTrainReorderMemoryPrim
      } else {
        inputUserPrim
      }
      val inputInferMemoryPrim = if (inputInferReorderPrim != 0) {
        forwardInferReorderPrims += inputInferReorderPrim
        inputInferReorderMemoryPrim
      } else {
        inputUserPrim
      }

      {
        val srcs = Array(inputTrainMemoryPrim, weightAndBiasUserPrim)
        val indexes = Array.fill(srcs.length)(0)
        val dsts = Array(outputTrainUserPrim, meanUserPrim, varianceUserPrim)

        forwardTrainPrims += MklDnn.PrimitiveCreate2(opTrainPrimDesc, srcs, indexes, srcs.length,
          dsts, dsts.length)
      }

      {
        val srcs = Array(inputInferMemoryPrim, meanUserPrim, varianceUserPrim,
          weightAndBiasUserPrim)
        val indexes = Array.fill(srcs.length)(0)
        val dsts = Array(outputInferUserPrim)

        forwardInferPrims += MklDnn.PrimitiveCreate2(opInferPrimDesc, srcs, indexes, srcs.length,
          dsts, dsts.length)
      }

      if (inputTrainReorderPrim == 0 && inputInferReorderPrim == 0
        && input.getTensorType == MklDnnType) {
        internalInput = input.asInstanceOf[MklDnnTensor[T]]
      } else {
        if (internalInput != null) {
          internalInput.release()
        }
        internalInput = MklDnnTensor[T](input.size())
      }

      if (internalInput.size().deep != input.size().deep) {
        internalInput.release()
        internalInput = MklDnnTensor[T](input.size())
      }
    }

    if (input.getTensorType == DenseType) {
      internalInput.set(input)
    }

    var inputPtr = 0L
    if (isTraining() && inputTrainReorderPrim != 0) {
      if (input.getTensorType == DenseType) {
        inputPtr = MklDnn.MemorySetDataHandle(inputUserPrim,
          input.storage().array().asInstanceOf[Array[Float]],
          input.storageOffset() - 1)
        Memory.SetDataHandle(inputTrainReorderMemoryPrim, internalInput.ptr, 0)
      } else {
        Memory.SetDataHandle(inputUserPrim,
          input.asInstanceOf[MklDnnTensor[T]].ptr,
          0)
        Memory.SetDataHandle(inputTrainReorderMemoryPrim, internalInput.ptr, 0)
      }
    } else if (!isTraining() && inputInferReorderPrim != 0) {
      if (input.getTensorType == DenseType) {
        inputPtr = MklDnn.MemorySetDataHandle(inputUserPrim,
          input.storage().array().asInstanceOf[Array[Float]],
          input.storageOffset() - 1)
        Memory.SetDataHandle(inputInferReorderMemoryPrim, internalInput.ptr, 0)
      } else {
        Memory.SetDataHandle(inputUserPrim,
          input.asInstanceOf[MklDnnTensor[T]].ptr,
          0)
        Memory.SetDataHandle(inputInferReorderMemoryPrim, internalInput.ptr, 0)
      }
    } else {
      if (input.getTensorType == DenseType) {
        MklDnnTensor.syncFromHeap(internalInput, input.storage().array(), input.storageOffset() - 1)
        Memory.SetDataHandle(inputUserPrim, internalInput.ptr, 0)
      } else if (input.getTensorType == MklDnnType) {
        Memory.SetDataHandle(inputUserPrim, input.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    MklDnnTensor.syncFromHeap(prvAll, all.storage().array(), all.storageOffset() - 1)
    Memory.SetDataHandle(weightAndBiasUserPrim, prvAll.ptr, 0)
    if (!isTraining()) {
      // BUG: this comes from Caffe. But I don't known how to relative this to BigDL.
//      val scale = 1 // if (scaleFactor == 0) { 0 } else { 1 / scaleFactor }
//      Memory.scale(nOutput, scale, runningMean.ptr, mean.ptr)
//      Memory.scale(nOutput, scale, runningVar.ptr, variance.ptr)
      Memory.SetDataHandle(meanUserPrim, runningMean.ptr, 0)
      Memory.SetDataHandle(varianceUserPrim, runningVar.ptr, 0)
    } else {
      Memory.SetDataHandle(meanUserPrim, mean.ptr, 0)
      Memory.SetDataHandle(varianceUserPrim, variance.ptr, 0)
    }

    if (isTraining()) {
      Memory.SetDataHandle(outputTrainUserPrim, output.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      if (inputTrainReorderPrim != 0) {
        DnnStream.Submit(stream, forwardTrainReorderPrims.length,
          forwardTrainReorderPrims.toArray)
        if (input.getTensorType == DenseType && inputPtr != 0) {
          MklDnn.MemoryReleaseDataHandle(input.storage().array().asInstanceOf[Array[Float]],
            inputPtr)
        }
      }

      DnnStream.Submit(stream, forwardTrainPrims.length, forwardTrainPrims.toArray)
    } else {
      Memory.SetDataHandle(outputInferUserPrim, output.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      if (inputInferReorderPrim != 0) {
        DnnStream.Submit(stream, forwardInferReorderPrims.length,
          forwardInferReorderPrims.toArray)
        if (input.getTensorType == DenseType && inputPtr != 0) {
          MklDnn.MemoryReleaseDataHandle(input.storage().array().asInstanceOf[Array[Float]],
            inputPtr)
        }
      }

      DnnStream.Submit(stream, forwardInferPrims.length, forwardInferPrims.toArray)
    }

    if (shouldConvert) {
      output.asInstanceOf[MklDnnTensor[T]].syncToHeap()
    }

    if (this.isTraining()) {
      // update running(Mean, Var) and scaleFactor
      scaleFactor = scaleFactor * momentum.toFloat + 1
      val m = input.nElement() / this.nOutput
      val biasFactor = if (m > 1) { m.toFloat / (m - 1) } else { 1 }

      Memory.Axpby(nOutput, 1, mean.ptr, momentum.toFloat, runningMean.ptr)
      Memory.Axpby(nOutput, biasFactor, variance.ptr, momentum.toFloat, runningVar.ptr)
    }

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugFwInfo(this.getName(), end1, input.getFormat(), output.getFormat())
    }
    output
  }

  @transient var gradOutputUserPrim = 0L
  @transient var gradOutputReorderPrim = 0L
  @transient var gradOutputReorderMemoryPrim = 0L
  @transient var gradInputUserPrim = 0L
  @transient var gradWeightAndBiasUserPrim = 0L
  @transient var internalGradInput, internalGradOutput: MklDnnTensor[T] = _
  def backward1(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val s1 = System.nanoTime()
    if (backwardPrims.isEmpty) {
      if (gradInput.getTensorType == MklDnnType) {
        toMklDnnTensor(gradInput).release()
      }
      gradInput = MklDnnTensor[T](input.size())

      val srcMemDesc = if (input.getPrimitiveDesc() == 0) {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          DataType.F32, defaultFormat)
      } else {
        MklDnnOps.primitiveDescQueryMemory(input.getPrimitiveDesc())
      }

      // [PERF] the format of gradInput should be the same as input
      val diffDstMemDesc = MklDnn.MemoryDescInit(input.dim(), input.size(),
        DataType.F32, MklDnn.getFormat(srcMemDesc))

      val desc = MklDnn.BatchNormBackwardDescInit(PropKind.Backward,
        diffDstMemDesc, srcMemDesc, eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      val primDesc = MklDnn.PrimitiveDescCreate(desc, engine, forwardTrainPrimDesc)

      val dataFormat = defaultFormat
      val paramsFormat = Memory.Format.x
      val dataType = DataType.F32

      gradOutputUserPrim = initUser(gradOutput, dataType, dataFormat, engine)
      val g1 = initInternal(gradOutputUserPrim, primDesc, Query.DiffDstPd)
      gradOutputReorderMemoryPrim = g1._1
      gradOutputReorderPrim = g1._2
      gradWeightAndBiasUserPrim = initUser(diffAll, dataType, paramsFormat, engine)
      gradInputUserPrim = initUser(gradInput, primDesc, Query.DiffSrcPd, 0)

      val inputMemoryPrim = if (inputTrainReorderPrim != 0) {
        inputTrainReorderMemoryPrim
      } else {
        inputUserPrim
      }

      val gradOutputMemoryPrim = if (gradOutputReorderPrim != 0) {
        backwardReorderPrims += gradOutputReorderPrim
        gradOutputReorderMemoryPrim
      } else {
        gradOutputUserPrim
      }

      val dataSrcs = Array(inputMemoryPrim, meanUserPrim,
        varianceUserPrim, gradOutputMemoryPrim,
        weightAndBiasUserPrim)
      val dataIndexes = Array.fill(dataSrcs.length)(0)
      val dataDsts = Array(gradInputUserPrim, gradWeightAndBiasUserPrim)

      backwardPrims += MklDnn.PrimitiveCreate2(primDesc, dataSrcs, dataIndexes, dataSrcs.length,
        dataDsts, dataDsts.length)

      if (backwardReorderPrims.isEmpty && gradOutput.getTensorType == MklDnnType) {
        internalGradOutput = toMklDnnTensor(gradOutput)
      } else {
        if (internalGradOutput != null) {
          internalGradOutput.release()
        }
        internalGradOutput = MklDnnTensor[T](input.size())
      }

      if (internalGradOutput.size().deep != input.size().deep) {
        internalGradOutput.release()
        internalGradOutput = MklDnnTensor[T](input.size())
      }
    }

    if (gradOutput.getTensorType == DenseType) {
      internalGradOutput.set(gradOutput)
    }

    var gradOutputPtr = 0L
    if (gradOutputReorderPrim != 0) {
      if (gradOutput.getTensorType == DenseType) {
        gradOutputPtr = MklDnn.MemorySetDataHandle(gradOutputUserPrim,
          gradOutput.storage().array().asInstanceOf[Array[Float]],
          gradOutput.storageOffset() - 1)
        Memory.SetDataHandle(gradOutputReorderMemoryPrim, internalGradOutput.ptr, 0)
      } else {
        Memory.SetDataHandle(gradOutputUserPrim,
          gradOutput.asInstanceOf[MklDnnTensor[T]].ptr,
          0)
        Memory.SetDataHandle(gradOutputReorderMemoryPrim, internalGradOutput.ptr, 0)
      }
    } else {
      if (gradOutput.getTensorType == DenseType) {
        MklDnnTensor.syncFromHeap(internalGradOutput, gradOutput.storage().array(),
          gradOutput.storageOffset() - 1)
        Memory.SetDataHandle(gradOutputUserPrim, internalGradOutput.ptr, 0)
      } else if (gradOutput.getTensorType == MklDnnType) {
        Memory.SetDataHandle(gradOutputUserPrim, gradOutput.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    Memory.SetDataHandle(gradWeightAndBiasUserPrim, diffAll.ptr, 0)
    Memory.SetDataHandle(meanUserPrim, mean.ptr, 0)
    Memory.SetDataHandle(varianceUserPrim, variance.ptr, 0)
    Memory.SetDataHandle(gradInputUserPrim, gradInput.asInstanceOf[MklDnnTensor[T]].ptr, 0)

    if (gradOutputReorderPrim != 0) {
      DnnStream.Submit(stream, backwardReorderPrims.length, backwardReorderPrims.toArray)
      if (gradOutput.getTensorType == DenseType && gradOutputPtr != 0) {
        MklDnn.MemoryReleaseDataHandle(gradOutput.storage().array().asInstanceOf[Array[Float]],
          gradOutputPtr)
      }
    }
    DnnStream.Submit(stream, backwardPrims.length, backwardPrims.toArray)

    diffAll.syncToHeap()
//    gradAll.add(diffAll)

    if (shouldConvert) {
      gradInput.asInstanceOf[MklDnnTensor[T]].syncToHeap()
    }

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, gradOutput.getFormat(), gradInput.getFormat())
    }
    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    backward1(input, gradOutput)
  }

  // there's no relavant accGrasdParameters in mkl-dnn. we use @backward instead of
  // @updateGradInput and @accGradParameters
  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
  }

  private type Params[R] = (Tensor[R], Tensor[R], Tensor[R])
  // in mkl dnn, the weight and bias should be all in the same array
  private def createParams(initWeight: Tensor[T], initBias: Tensor[T]): Tensor[T] = {
    val weightAndBias: Tensor[T] = if (affine) {
      Tensor[T](Array(2 * nOutput))
    } else {
      null
    }

    val concat = Tensor[T]().resize(Array(2, nOutput)).fill(ev.fromType(0))

    if (initWeight != null) {
      require(initWeight.size(1) == nOutput)
      concat.select(1, 1).copy(initWeight)
    } else {
      val weight = concat.select(1, 1)
      weightInitMethod.init(weight, VariableFormat.ONE_D)
    }

    if (initBias != null) {
      require(initBias.size(1) == nOutput)
      concat.select(1, 2).copy(initBias)
    } else {
      val bias = concat.select(1, 2)
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }

    weightAndBias.copy(concat.view(Array(2 * nOutput)))
    weightAndBias
  }

  override def zeroGradParameters(): Unit = {
    if (affine) {
      gradAll.zero()
      diffAll.zero()
      Memory.Zero(diffAll.ptr, diffAll.nElement(), 4)
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (affine) {
      (Array(all), Array(diffAll))
    } else {
      null
    }
  }

  override def getParametersTable(): Table = {
    if (affine) {
      T(getName() -> T("weight" -> all,
        "gradWeight" -> diffAll,
        "runningMean" -> mean, "runningVar" -> variance))
    } else {
      T(getName() -> T("runningMean" -> mean, "runningVar" -> variance))
    }
  }

  override def toString(): String = {
    s"mkldnn.BatchNormalization($nOutput, $eps, $momentum, $affine)"
  }
}

object SpatialBatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null)
    (implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {

    new SpatialBatchNormalization[T](
      nOutput, eps, momentum, affine, initWeight, initBias, initGradWeight, initGradBias)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
    affine: Option[Int])(implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {
    new SpatialBatchNormalization[T](nOutput = affine.getOrElse(1), affine = affine.isDefined)
  }
}

class RefactorSpatialBatchNormalization(
  val nOutput: Int,
  val eps: Double = 1e-5,
  val momentum: Double = 0.1,
  val affine: Boolean = true,
  private val initWeight: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null,
  private val initGradWeight: Tensor[Float] = null,
  private val initGradBias: Tensor[Float] = null
) extends MklDnnLayer with Initializable {

  private var forwardDesc: Long = 0L
  private var _relu: Boolean = false

  def setReLU(): Unit = _relu = true
  def relu: Boolean = _relu

  var updateOutputTensors: Array[Tensor[Float]] = _
  var updateOutputMemoryPrimitives: Array[Long] = _
  var updateGradInputTensors: Array[Tensor[Float]] = _
  var updateGradInputMemoryPrimitives: Array[Long] = _

  @transient var mean: DnnTensor[Float] = DnnTensor[Float](nOutput)
  @transient var variance: DnnTensor[Float] = DnnTensor[Float](nOutput)
  @transient var runningMean: DnnTensor[Float] = DnnTensor[Float](nOutput)
  @transient var runningVariance: DnnTensor[Float] = DnnTensor[Float](nOutput)
  @transient var weightAndBias: DnnTensor[Float] = DnnTensor[Float](Array(nOutput * 2))
  @transient var gradWeightAndBias: DnnTensor[Float] = DnnTensor[Float](Array(nOutput * 2))

  var scaleFactor: Float = 0.0f
  var biasFactor: Float = 0.0f

  {
    val wInit = Ones // RandomUniform(0, 1)
    val bInit = Zeros
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    val init = Tensor[Float]().resize(Array(2, nOutput))
    val weight = init.select(1, 1)
    val bias = init.select(1, 2)

    if (initWeight != null) {
      require(initWeight.size(1) == nOutput)
      weight.copy(initWeight)
    } else {
      weightInitMethod.init(weight, VariableFormat.ONE_D)
    }

    if (initBias != null) {
      require(initBias.size(1) == nOutput)
      bias.copy(initBias)
    } else {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }

    weightAndBias.copy(init.view(2 * nOutput))

    val zeros = Tensor[Float](Array(nOutput)).fill(0)
    mean.copy(zeros)
    variance.copy(zeros)
  }

  object Index {
    val input = 0
    val weight = 1
    val output = 2
    val mean = 3
    val variance = 4
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = inputs

    val m = inputFormats()(0).shape.product / this.nOutput
    biasFactor = if (m > 1) { m.toFloat / (m - 1) } else { 1 }

    val List(mean, variance, runningMean, runningVariance): List[NativeData] =
      (0 until 4).map { _ =>
        NativeData(Array(nOutput), Memory.Format.x)
      }.toList
    // weight and bias should be combined
    val weightAndBias: NativeData = NativeData(Array(nOutput * 2), Memory.Format.x)

    forwardDesc = phase match {
      case TrainingPhase =>
        MklDnn.BatchNormForwardDescInit(PropKind.Forward,
          inputs(0).getMemoryDescription(), eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      case InferencePhase =>
        // we always use the weight and bias / scale and offset. So the flags should be combined
        // with use_scaleshift and use_global_stats.
        MklDnn.BatchNormForwardDescInit(PropKind.ForwardInference,
          inputs(0).getMemoryDescription(), eps.toFloat,
          MklDnn.BatchNormFlag.mkldnn_use_global_stats | MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      case _ => throw new UnsupportedOperationException
    }

    val primDesc = if (phase == InferencePhase && relu) {
      val postOps = MklDnn.CreatePostOps()
      MklDnn.PostOpsAppendEltwise(postOps, 1.0f, AlgKind.EltwiseRelu, 0.0f, 0.0f)
      val attr = MklDnn.CreateAttr()
      MklDnn.AttrSetPostOps(attr, postOps)
      MklDnn.PrimitiveDescCreateV2(forwardDesc, attr, runtime.engine, 0)
      // TODO we should destroy these ops
    } else {
      MklDnn.PrimitiveDescCreate(forwardDesc, runtime.engine, 0)
    }

    _inputFormats = Array(MemoryData.operationWant(primDesc, Query.SrcPd))
    _outputFormats = Array(MemoryData.operationWant(primDesc, Query.DstPd))

    val (srcs, dsts) = if (phase == TrainingPhase) {
      val srcs = Array(inputFormats()(0), weightAndBias).map(_.getPrimitive(runtime))
      val dsts = Array(outputFormats()(0), mean, variance).map(_.getPrimitive(runtime))
      (srcs, dsts)
    } else {
      val srcs = Array(inputFormats()(0), runningMean, runningVariance, weightAndBias).map { x =>
        x.getPrimitive(runtime)
      }
      val dsts = Array(outputFormats()(0).getPrimitive(runtime))
      (srcs, dsts)
    }
    val indexes = Array.fill(srcs.length)(0)

    val primitive = MklDnn.PrimitiveCreate2(primDesc, srcs, indexes, srcs.length, dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)
    output = initTensor(outputFormats()(0))

    if (this.isTraining()) {
      this.runningMean.zero()
      this.runningVariance.zero()
    }

    if (updateOutputTensors != null) {
      updateOutputTensors = Array.empty
    }

    (inputFormats(), outputFormats())
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      if (this.isTraining()) {
        val buffer = new ArrayBuffer[Tensor[Float]]()
        buffer.append(input.asInstanceOf[Tensor[Float]])
        buffer.append(weightAndBias)
        buffer.append(output.asInstanceOf[Tensor[Float]])
        buffer.append(mean)
        buffer.append(variance)
        updateOutputTensors = buffer.toArray
      } else {
        val buffer = new ArrayBuffer[Tensor[Float]]()
        buffer.append(input.asInstanceOf[Tensor[Float]])
        buffer.append(runningMean)
        buffer.append(runningVariance)
        buffer.append(weightAndBias)
        buffer.append(output.asInstanceOf[Tensor[Float]])
        updateOutputTensors = buffer.toArray
      }
    }

    updateWithNewTensor(updateOutputTensors, 0, input)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors)

    if (this.isTraining()) {
      // update running(Mean, Var) and scaleFactor
      scaleFactor = scaleFactor * momentum.toFloat + 1

      mean.axpby(1, momentum.toFloat, runningMean)
      variance.axpby(biasFactor, momentum.toFloat, runningVariance)
    }

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = Array(NativeData(outputFormats()(0).shape, outputFormats()(0).layout))

    // [PERF] the format of gradInput should be the same as input
    val backwardDesc = phase match {
      case TrainingPhase =>
        MklDnn.BatchNormBackwardDescInit(PropKind.Backward,
          inputFormats()(0).getMemoryDescription(),
          inputFormats()(0).getMemoryDescription(), eps.toFloat,
          MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      case _ => throw new UnsupportedOperationException
    }

    val gradWeightAndBias: NativeData = NativeData(Array(nOutput * 2), Memory.Format.x)
    val gradWeightPrimitive = gradWeightAndBias.getPrimitive(runtime)

    val primDesc = MklDnn.PrimitiveDescCreate(backwardDesc, runtime.engine, 0)

    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffSrcPd))

    // maybe will throw null exception
    val srcs = Array(updateOutputMemoryPrimitives(Index.input),
      updateOutputMemoryPrimitives(Index.mean),
      updateOutputMemoryPrimitives(Index.variance),
      grad(0).getPrimitive(runtime),
      updateOutputMemoryPrimitives(Index.weight))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(gradInputFormats()(0), gradWeightAndBias).map(_.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(primDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradInputMemoryPrimitives = srcs ++ dsts
    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(gradInputFormats()(0))

    (_gradOutputFormats, gradInputFormats())
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(mean)
      buffer.append(variance)
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(weightAndBias)
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      buffer.append(gradWeightAndBias.asInstanceOf[Tensor[Float]])
      updateGradInputTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 0, input)
    updateWithNewTensor(updateGradInputTensors, 3, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
      updateGradInputPrimitives.length, updateGradInputMemoryPrimitives, updateGradInputTensors)

    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    // do nothing
  }

  override def zeroGradParameters(): Unit = {
    if (affine) { gradWeightAndBias.zero() }
    if (gradInput != null) { gradInput.asInstanceOf[DnnTensor[Float]].zero() }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weightAndBias), Array(gradWeightAndBias))
  }

  override def parametersWithShape(): (Array[MemoryData], Array[MemoryData]) = {
    (Array(NativeData(weightAndBias.size(), Memory.Format.x)),
      Array(NativeData(gradWeightAndBias.size(), Memory.Format.x)))
  }

  override def toString(): String = {
    s"nn.mkl.SpatialBatchNormalization($nOutput, $eps, $momentum, $affine)"
  }
}

object RefactorSpatialBatchNormalization {
  def apply(
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null): RefactorSpatialBatchNormalization = {
    new RefactorSpatialBatchNormalization(nOutput, eps, momentum, affine,
      initWeight, initBias, initGradWeight, initGradBias)
  }
}