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

import com.intel.analytics.bigdl.nn.DynamicContainer
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable

/**
 * Helper utilities when integrating Module with MKL-DNN
 */
trait MklDnnModule {
  /**
   * MklDnn runtime, which includes a MKL-DNN engine and a MKL-DNN stream.
   * Note that this instance will be erased when send to remote worker, so you
   * should recreate a MklDnnRuntime.
   */
  @transient
  protected var runtime : MklDnnRuntime = _

  /**
   * Compute the output formats based on the input formats
   */
  private[mkldnn] def inferShape(shapes: Array[Array[Int]]): Array[Array[Int]]

  /**
   * Init the MKL-DNN primitives for the model
   * @param runtime
   */
  private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit
  private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit
  private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase): Unit

  private[mkldnn] def initMemory(): Unit

  private[mkldnn] def inputFormats(): Array[MemoryData]

  private[mkldnn] def gradInputFormats(): Array[MemoryData]

  private[mkldnn] def outputFormats(): Array[MemoryData]

  private[mkldnn] def gradOutputFormats(): (Array[MemoryData], Array[MemoryData])
}

trait MklDnnLayer extends MklDnnModule {
  /**
   * MKL-DNN primitives of the module. Note you should only initialize this field by calling
   * initPrimitives method. This field will be erased when sending model to remote worker. So you
   * need to reinitialize it after sending the model.
   */
  @transient
  protected var updateOutputPrimitives: Array[Long] = _
  @transient
  protected var updateGradInputPrimitives: Array[Long] = _
  @transient
  protected var accGradientPrimitives: Array[Long] = _

  @transient
  protected var inputPrimitives: Array[Long] = _
  @transient
  protected var gradInputPrimitives: Array[Long] = _
  @transient
  protected var outputPrimitives: Array[Long] = _
  @transient
  protected var gradOutputPrimitives: Array[Long] = _
  @transient
  protected var gradOutputPrimitivesWeight: Array[Long] = _
}

/**
 * Helper utilities when integrating containers with MKL-DNN
 */
trait MklDnnContainer extends DynamicContainer[Activity, Activity, Float] with MklDnnModule {
  protected val reorderManager = new ReorderManager()
  protected var mklDnnModules : Array[MklDnnModule] = _

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, Float]): this.type = {
    require(mklDnnModules == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  /**
   * Create MklDnnRuntime and compile the model
   * @param phase
   */
  final def compile(phase: Phase): Unit = {
    compile(phase, new MklDnnRuntime())
  }

  /**
   * Compile the model, which includes infer memory shapes, allocate memory, optimize computing
   * path and create MKL-DNN primitives
   * @param phase
   * @param runtime
   */
  final def compile(phase: Phase, runtime: MklDnnRuntime): Unit = {
    freeze()
    inputFormats().foreach(f => require(f.isLayoutFixed(), "Model input layout should be fixed"))
    fusion(phase)
    inferShape(inputFormats.map(_.shape))
    initFwdPrimitives(runtime, phase)
    if (phase == Phase.TrainingPhase) {
      initBwdPrimitives(runtime, phase)
      initGradWPrimitives(runtime, phase)
    }
    initMemory()
  }

  /**
   * Modify the computing path by fuse some layers into one to improve the performance
   */
  private[mkldnn] def fusion(phase: Phase): Unit = {
    modules.filter(_.isInstanceOf[MklDnnContainer])
      .map { case mc: MklDnnContainer => mc.fusion(phase) }
  }

  private def freeze(): Unit = {
    if (mklDnnModules == null) {
      mklDnnModules = modules.map(_.asInstanceOf[MklDnnModule]).toArray
    }
    modules.filter(_.isInstanceOf[MklDnnContainer])
      .map { case mc: MklDnnContainer => mc.freeze() }
  }

  override private[mkldnn] def initMemory() = {
    modules.foreach { case m: MklDnnModule => m.initMemory() }
  }
}

private[mkldnn] class ReorderManager() {
  // (MemoryFormatId, TargetFormat) -> Reorder
  val reorders = mutable.HashMap[(Int, MemoryData), ReorderMemory]()
  // ReorderId -> RefCount
  val refCounts = mutable.HashMap[Int, Int]()
  val useCounts = mutable.HashMap[Int, Int]()

  def register(from: MemoryData, to: MemoryData, runtime: MklDnnRuntime, phase: Phase): Unit = {
    val mId = System.identityHashCode(from)
    if (needReorder(from, to)) {
      if (reorders.contains((mId, to))) {
        refCounts(System.identityHashCode(reorders((mId, to)))) += 1
      } else {
        val reorder = ReorderMemory(from, to)
        reorder.initFwdPrimitives(runtime, phase)
        reorder.initMemory()
        reorders((mId, to)) = reorder
        val reorderId = System.identityHashCode(reorder)
        refCounts(reorderId) = 1
        useCounts(reorderId) = 0
      }
    }
  }

  def infer(from: Array[MemoryData], to: Array[MemoryData], output: Activity)
  : Activity = {
    if (from.length == 1) {
      require(output.isTensor, "output activity should be a tensor")
      inferTensor(from(0), to(0), output.asInstanceOf[Tensor[Float]])
    } else {
      require(output.toTable.length() == from.length,
        "output activity length doesn't match")
      val outputTable = T()
      var i = 0
      while(i < from.length) {
        outputTable(i + 1) = inferTensor(from(i), to(i), output.toTable(i + 1))
        i += 1
      }
      output
    }
  }

  private def inferTensor(from: MemoryData, to : MemoryData, output: Tensor[Float])
  : Tensor[Float] = {
    val mId = System.identityHashCode(from)
    if (reorders.contains((mId, to))) {
      val reorder = reorders((mId, to))
      val reorderId = System.identityHashCode(reorder)
      val result = if (useCounts(reorderId) == 0) {
        reorder.forward(output)
      } else {
        reorder.output
      }
      useCounts(reorderId) += 1
      if (useCounts(reorderId) == refCounts(reorderId)) {
        useCounts(reorderId) = 0
      }
      result
    } else {
      output
    }
  }

  private def needReorder(from: MemoryData, to: MemoryData): Boolean = {
    from match {
      case h: HeapData =>
        to match {
          case hh: HeapData =>
            require(h.layout == hh.layout, "Heap data layout should be same")
            false
          case nn: NativeData => true
          case _ => throw new UnsupportedOperationException("Not support such memory format")
        }
      case n: NativeData =>
        to match {
          case hh: HeapData => true
          case nn: NativeData =>
            nn.layout != n.layout
          case _ => throw new UnsupportedOperationException("Not support such memory format")
        }
      case _ => throw new UnsupportedOperationException("Not support such memory format")
    }
  }
}
