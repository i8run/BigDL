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

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.nn.{Im2Col, SpatialConvolution}
import com.intel.analytics.bigdl.tensor.Tensor

case class Conv(
  nInputPlane: Int,
  nOutputPlane: Int,
  kW: Int,
  kH: Int,
  dW: Int,
  dH: Int,
  padW: Int,
  padH: Int,

  batchSize: Int,
  inputHeight: Int,
  inputWidth: Int
)

object ConvPerf {
  val convParams: List[Conv] = List(
    Conv(1024, 128, 1, 1, 1, 1, 0, 0, 32, 2, 2),
    Conv(1024, 128, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(1024, 160, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(1024, 192, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(1024, 352, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(112, 224, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(128, 128, 3, 3, 1, 1, 1, 1, 32, 112, 112),
    Conv(128, 128, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(128, 160, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(128, 160, 3, 3, 2, 2, 1, 1, 32, 28, 28),
    Conv(128, 192, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(128, 192, 3, 3, 1, 1, 1, 1, 32, 28, 28),
    Conv(128, 192, 3, 3, 2, 2, 1, 1, 32, 14, 14),
    Conv(128, 256, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(128, 256, 3, 3, 1, 1, 1, 1, 32, 56, 56),
    Conv(144, 288, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(160, 160, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(160, 192, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(160, 224, 3, 3, 1, 1, 1, 1, 32, 7, 7),
    Conv(160, 320, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(160, 320, 3, 3, 1, 1, 1, 1, 32, 7, 7),
    Conv(16, 32, 5, 5, 1, 1, 2, 2, 32, 28, 28),
    Conv(16, 48, 5, 5, 1, 1, 2, 2, 32, 14, 14),
    Conv(192, 16, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(192, 192, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(192, 224, 3, 3, 1, 1, 1, 1, 32, 7, 7),
    Conv(192, 256, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(192, 320, 3, 3, 1, 1, 1, 1, 32, 7, 7),
    Conv(192, 32, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(192, 384, 3, 3, 1, 1, 1, 1, 32, 7, 7),
    Conv(192, 64, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(192, 96, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(224, 224, 3, 3, 1, 1, 1, 1, 32, 7, 7),
    Conv(24, 64, 5, 5, 1, 1, 2, 2, 32, 14, 14),
    Conv(256, 128, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(256, 256, 3, 3, 1, 1, 1, 1, 32, 56, 56),
    Conv(256, 256, 3, 3, 2, 2, 1, 1, 32, 14, 14),
    Conv(256, 32, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(256, 512, 3, 3, 1, 1, 1, 1, 32, 28, 28),
    Conv(256, 64, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(320, 128, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(320, 64, 1, 1, 1, 1, 0, 0, 32, 28, 28),
    Conv(32, 128, 5, 5, 1, 1, 2, 2, 32, 14, 14),
    Conv(32, 128, 5, 5, 1, 1, 2, 2, 32, 7, 7),
    Conv(32, 64, 5, 5, 1, 1, 2, 2, 32, 14, 14),
    Conv(32, 96, 5, 5, 1, 1, 2, 2, 32, 28, 28),
    Conv(3, 64, 3, 3, 1, 1, 1, 1, 32, 224, 224),
    Conv(3, 64, 7, 7, 2, 2, 3, 3, 32, 224, 224),
    Conv(480, 16, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(480, 192, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(480, 64, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(480, 96, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(48, 128, 5, 5, 1, 1, 2, 2, 32, 7, 7),
    Conv(512, 112, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(512, 128, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(512, 128, 1, 1, 1, 1, 0, 0, 32, 4, 4),
    Conv(512, 144, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(512, 160, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(512, 24, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(512, 32, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(512, 512, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(512, 512, 3, 3, 1, 1, 1, 1, 32, 28, 28),
    Conv(512, 64, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(528, 128, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(528, 128, 1, 1, 1, 1, 0, 0, 32, 4, 4),
    Conv(528, 160, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(528, 256, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(528, 32, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(576, 128, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(576, 128, 1, 1, 1, 1, 0, 0, 32, 4, 4),
    Conv(576, 160, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(576, 192, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(576, 224, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(576, 64, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(576, 96, 1, 1, 1, 1, 0, 0, 32, 14, 14),
    Conv(64, 128, 3, 3, 1, 1, 1, 1, 32, 112, 112),
    Conv(64, 192, 3, 3, 1, 1, 1, 1, 32, 56, 56),
    Conv(64, 64, 1, 1, 1, 1, 0, 0, 32, 56, 56),
    Conv(64, 64, 3, 3, 1, 1, 1, 1, 32, 224, 224),
    Conv(64, 64, 3, 3, 1, 1, 1, 1, 32, 28, 28),
    Conv(64, 96, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(64, 96, 3, 3, 1, 1, 1, 1, 32, 28, 28),
    Conv(832, 128, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(832, 160, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(832, 192, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(832, 256, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(832, 32, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(832, 384, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(832, 48, 1, 1, 1, 1, 0, 0, 32, 7, 7),
    Conv(96, 128, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(96, 128, 3, 3, 1, 1, 1, 1, 32, 28, 28),
    Conv(96, 208, 3, 3, 1, 1, 1, 1, 32, 14, 14),
    Conv(96, 96, 3, 3, 1, 1, 1, 1, 32, 28, 28),
    Conv(96, 96, 3, 3, 2, 2, 1, 1, 32, 28, 28),
    Conv(1024, 1024, 1, 1, 1, 1, 0, 0, 32, 19, 19),
    Conv(1024, 126, 3, 3, 1, 1, 1, 1, 32, 19, 19),
    Conv(1024, 24, 3, 3, 1, 1, 1, 1, 32, 19, 19),
    Conv(256, 16, 3, 3, 1, 1, 1, 1, 32, 1, 1),
    Conv(256, 84, 3, 3, 1, 1, 1, 1, 32, 1, 1),
    Conv(256, 16, 3, 3, 1, 1, 1, 1, 32, 3, 3),
    Conv(256, 84, 3, 3, 1, 1, 1, 1, 32, 3, 3),
    Conv(256, 126, 3, 3, 1, 1, 1, 1, 32, 5, 5),
    Conv(256, 24, 3, 3, 1, 1, 1, 1, 32, 5, 5),
    Conv(512, 126, 3, 3, 1, 1, 1, 1, 32, 10, 10),
    Conv(512, 24, 3, 3, 1, 1, 1, 1, 32, 10, 10),
    Conv(512, 16, 3, 3, 1, 1, 1, 1, 32, 38, 38),
    Conv(512, 84, 3, 3, 1, 1, 1, 1, 32, 38, 38)
  )
  def main(args: Array[String]): Unit = {
//    if (args.length < 1) {
//      println("[ERROR] unknown im2col method")
//    }
//
//    if (args(0) == "true") {
//      Im2Col.useOptimization = true
//    } else {
//      Im2Col.useOptimization = false
//    }

    val warmIterations = 10
    val iterations = 100
    for (test <- convParams) {
      val conv = new SpatialConvolution[Float](
        test.nInputPlane,
        test.nOutputPlane,
        test.kH,
        test.kW,
        test.dH,
        test.dW,
        test.padH,
        test.padW
      )

      val input = Tensor[Float](Array(test.batchSize, test.nInputPlane,
        test.inputHeight, test.inputWidth)).randn()

//      // warm up
//      for (i <- 0 until warmIterations) {
//        conv.updateOutput(input)
//      }
//
//      // real time
//      val start = System.nanoTime()
//      for (i <- 0 until iterations) {
//        conv.updateOutput(input)
//      }
//      val end = System.nanoTime()
//
//      println(s"costs: ${(end - start) / 1e6} ms")

      val outputHeight = (test.inputHeight + 2 * test.padH - test.kH) / test.dH + 1
      val outputWidth = (test.inputWidth + 2 * test.padW - test.kW) / test.dW + 1

      val size = test.nInputPlane * test.kH * test.kW * outputHeight * outputWidth * 4 / 1024.0
      println(s"$size KB -> ${size / test.nInputPlane} ${outputHeight * outputWidth * test.nOutputPlane * 4 / 1024}")
    }
  }
}
