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

import java.util.concurrent.{Executors, ThreadFactory}

import com.intel.analytics.bigdl.tensor.Tensor

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

object CacheMM{
  val cores = Runtime.getRuntime.availableProcessors() / 2

  // TODO use default
  val context = new ExecutionContext {
    val threadPool = Executors.newFixedThreadPool(cores, new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setDaemon(true)
        t
      }
    })

    def execute(runnable: Runnable) {
      threadPool.submit(runnable)
    }

    def reportFailure(t: Throwable) {}
  }

  def invoke[T](task: () => T): Future[T] = {
    Future {
      task()
    }(context)
  }

  def sync(futures: Seq[Future[_]], timeout: Duration = Duration.Inf): Unit = {
    futures.foreach(f => {
      Await.result(f, timeout)
    })
  }

  def mul(output: Tensor[Float], weight: Tensor[Float], dataCol: Tensor[Float]): Unit = {
    output.addmm(0, output, 1, weight, dataCol)
  }

  def time[R](block: => R): Double = {
    val warmTimes = 10
    val iterations = 20
    for (i <- 0 until warmTimes) {
      block
    }

    val start = System.nanoTime()
    for (i <- 0 until iterations) {
      block
    }
    val end = System.nanoTime()

    (end - start) / 1e6 / iterations
  }

  def one(conv: Conv): Unit = {
//    val conv = Conv(512, 84, 3, 3, 1, 1, 1, 1, 32, 38, 38)
//    val conv = Conv(512, 16, 3, 3, 1, 1, 1, 1, 32, 38, 38)
//    val conv = Conv(16, 32, 5, 5, 1, 1, 2, 2, 32, 28, 28)
//    val conv = Conv(832, 160, 1, 1, 1, 1, 0, 0, 32, 7, 7)

    val outputHeight = (conv.inputHeight + 2 * conv.padH - conv.kH) / conv.dH + 1
    val outputWidth = (conv.inputWidth + 2 * conv.padW - conv.kW) / conv.dW + 1
    val size = conv.nInputPlane * conv.kH * conv.kW * outputHeight * outputWidth * 4 / 1024.0

    val output = new Array[Tensor[Float]](cores)
    val dataCol = new Array[Tensor[Float]](cores)
    val weight = new Array[Tensor[Float]](cores)
    val results = new Array[Future[Unit]](cores)

    for (i <- 0 until cores) {
      output(i) = Tensor[Float]().resize(Array(conv.nOutputPlane, outputHeight * outputWidth))
      output(i).fill(0)

      dataCol(i) = Tensor[Float]().resize(Array(conv.kW * conv.kH * conv.nInputPlane,
        outputHeight * outputWidth)).randn()
      weight(i) = Tensor[Float]().resize(Array(conv.nOutputPlane,
        conv.nInputPlane * conv.kH * conv.kW)).randn()
    }

    var oneImageTime = 0.0
    oneImageTime += time {
      var i = 0
      while (i < cores) {
        val _i = i
        results(i) = invoke(() => mul(output(_i), weight(_i), dataCol(_i)))
        i += 1
      }

      sync(results)
    }

    //    println(s"$size kB costs: $elapsed")

    var oneChannel = 0.0

    val frameSize = conv.kH * conv.kW * outputHeight * outputWidth
    for (i <- 0 until cores) {
      output(i) = Tensor[Float]().resize(Array(conv.nOutputPlane, outputHeight * outputWidth))
      output(i).fill(0)

      dataCol(i) = Tensor[Float]().resize(Array(conv.nInputPlane, conv.kW * conv.kH,
        outputHeight * outputWidth)).randn()
      weight(i) = Tensor[Float]().resize(Array(conv.nOutputPlane, conv.kH * conv.kW)).randn()
    }

    oneChannel += time {
      var i = 0
      while (i < cores) {
        val _i = i

        var j = 0
        results(i) = invoke(() => while (j < conv.nInputPlane) {
          val _j = j
          val outputFrame = output(_i)
          val weightFrame = weight(_i)
          val dataColFrame = dataCol(_i).select(1, _j + 1)
          mul(outputFrame, weightFrame, dataColFrame)
          j += 1
        })

        i += 1
      }

      sync(results)
    }
//    println(s"$elapsed")

    println(s"$conv\t$size\t${"%.3f".format(oneImageTime)}\t${"%.3f".format(oneChannel)}")
  }

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
    for (test <- convParams) {
      one(test)
    }
  }
}
