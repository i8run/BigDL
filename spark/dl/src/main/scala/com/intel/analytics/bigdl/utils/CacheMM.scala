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

  def main(args: Array[String]): Unit = {
     val conv = Conv(512, 84, 3, 3, 1, 1, 1, 1, 32, 38, 38)
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

    var elapsed = 0.0
    elapsed += time {
      var i = 0
      while (i < cores) {
        val _i = i
        results(i) = invoke(() => mul(output(_i), weight(_i), dataCol(_i)))
        i += 1
      }

      sync(results)
    }

    println(s"$size kB costs: $elapsed")

    elapsed = 0.0

    val frameSize = conv.kH * conv.kW * outputHeight * outputWidth
    for (i <- 0 until cores) {
      output(i) = Tensor[Float]().resize(Array(conv.nOutputPlane, outputHeight * outputWidth))
      output(i).fill(0)

      dataCol(i) = Tensor[Float]().resize(Array(conv.nInputPlane, conv.kW * conv.kH,
        outputHeight * outputWidth)).randn()
      weight(i) = Tensor[Float]().resize(Array(conv.nOutputPlane, conv.kH * conv.kW)).randn()
    }

    elapsed += time {
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
    println(s"$elapsed")
  }

}
