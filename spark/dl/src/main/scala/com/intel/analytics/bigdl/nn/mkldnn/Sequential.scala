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

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{DynamicContainer, Sequential => Seq}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Sequential[T: ClassTag](implicit ev: TensorNumeric[T])
  extends DynamicContainer[Activity, Activity, T] with MklDnnContainer {

  private val reorderManager = new ReorderManager()

  private var updateOutputExecutions: Array[AbstractModule[_, _, T]] = _
  private var updateGradInputExecutions: Array[AbstractModule[_, _, T]] = _
  private var accGradExecutions: Array[AbstractModule[_, _, T]] = _

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(updateOutputExecutions == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  override private[mkldnn] def fusion(): Unit = {
    modules.map { case mc: MklDnnContainer => mc.fusion() }
  }

  override private[mkldnn] def inferShape(shapes: Array[Array[Int]]) = {
    var lastShape = shapes
    modules.foreach { case m: MklDnnModule => lastShape = m.inferShape(lastShape)}
    lastShape
  }

  override private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    val mklModules = modules.map(_.asInstanceOf[MklDnnModule])
    require(MemoryData.noUndef(mklModules(0).inputFormats()), "Memory formats should be inited")
    mklModules(0).initFwdPrimitives(runtime, phase)
    var lastOutputFormats = mklModules(0).outputFormats()
    for (i <- 1 until modules.length) {
      val m = modules(i).asInstanceOf[MklDnnModule]
      lastOutputFormats.zip(m.inputFormats()).foreach {
        case (o, i) => if (i.layout == MklDnn.MemoryFormat.format_undef) {
          i.setLayout(o.layout)
        }
      }
      m.initFwdPrimitives(runtime, phase)
      lastOutputFormats.zip(m.inputFormats()).foreach {
        case (o, i) => reorderManager.register(o, i, runtime, phase)
      }
      lastOutputFormats = m.outputFormats()
    }
  }

  override private[mkldnn] def inputFormats() = {
    modules(0).asInstanceOf[MklDnnModule].inputFormats()
  }

  override private[mkldnn] def gradInputFormats() = {
    modules(0).asInstanceOf[MklDnnModule].gradInputFormats()
  }

  override private[mkldnn] def outputFormats() = {
    modules.last.asInstanceOf[MklDnnModule].outputFormats()
  }

  override private[mkldnn] def gradOutputFormats() = {
    modules.last.asInstanceOf[MklDnnModule].gradOutputFormats()
  }

  override private[mkldnn] def initMemory() = {
    modules.foreach { case m: MklDnnModule => m.initMemory() }
  }

  override private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    val mklModules = modules.map(_.asInstanceOf[MklDnnModule])
    require(MemoryData.noUndef(mklModules.last.gradOutputFormats()._1),
      "Memory formats should be inited")
    mklModules.last.initBwdPrimitives(runtime, phase)
    var lastInputFormats = mklModules.last.inputFormats()
    for (i <- modules.length - 1 to 0 by -1) {
      val m = modules(i).asInstanceOf[MklDnnModule]
      lastInputFormats.zip(m.gradOutputFormats()._1).foreach {
        case (o, i) => if (i.layout == MklDnn.MemoryFormat.format_undef) {
          i.setLayout(o.layout)
        }
      }
      m.initBwdPrimitives(runtime, phase)
      lastInputFormats.zip(m.gradOutputFormats()._1).foreach {
        case (o, i) => reorderManager.register(o, i, runtime, phase)
      }
      lastInputFormats = m.gradOutputFormats()._1
    }
  }

  override private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    val mklModules = modules.map(_.asInstanceOf[MklDnnModule])
    require(MemoryData.noUndef(mklModules.last.gradOutputFormats()._2),
      "Memory formats should be inited")
    mklModules.last.initGradWPrimitives(runtime, phase)
    var lastInputFormats = mklModules.last.inputFormats()
    for (i <- modules.length - 1 to 0 by -1) {
      val m = modules(i).asInstanceOf[MklDnnModule]
      lastInputFormats.zip(m.gradOutputFormats()._2).foreach {
        case (o, i) => if (i.layout == MklDnn.MemoryFormat.format_undef) {
          i.setLayout(o.layout)
        }
      }
      m.initGradWPrimitives(runtime, phase)
      lastInputFormats.zip(m.gradOutputFormats()._2).foreach {
        case (o, i) => reorderManager.register(o, i, runtime, phase)
      }
      lastInputFormats = m.gradOutputFormats()._2
    }
  }

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    var result = input
    while (i < modules.length) {
      result = reorderManager.infer(
        modules(i).asInstanceOf[MklDnnModule], modules(i + 1).asInstanceOf[MklDnnModule],
        modules(i).forward(result)
      )
      i += 1
    }

    this.output = result
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = ???
}
