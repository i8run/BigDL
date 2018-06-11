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

sealed trait MemoryData {
  def shape: Array[Int]
  def layout: Int
  def setShape(shape: Array[Int]): Unit
  def setLayout(layout: Int): Unit
}

case class HeapData(private var _shape: Array[Int], private var _layout: Int) extends MemoryData {

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout

  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout
}

case class NativeData(private var _shape: Array[Int], private var _layout: Int) extends MemoryData {
  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout
}

private[mkldnn] object MemoryData {

  def noUndef(formats: Array[MemoryData]): Boolean = {
    if (formats == null || formats.length == 0) return true
    formats.foreach(f => if (f.layout == MklDnn.MemoryFormat.format_undef) return false)
    return true
  }

  def isFixed(formats: MemoryData): Boolean = {
    formats.layout == MklDnn.MemoryFormat.format_undef
  }

  def isSizeCompatible(actual: MemoryData, expect: MemoryData): Boolean = {
    if (expect == null) return true
    if (actual == null) return false
    if (actual.shape.length != expect.shape.length) return false
    actual.shape.zip(expect.shape).foreach {case (a, e) => if (a != e) return false}
    return true
  }
}