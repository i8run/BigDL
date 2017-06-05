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

import com.intel.analytics.bigdl.mkl.MKL
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global

object CacheMM {
  val cores = Runtime.getRuntime.availableProcessors() / 2
  MKL.setNumThreads(1)

  def time[R](block: => R): Double = {
    val warmTimes = 10
    val iterations = 200
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
    val outputHeight = (conv.inputHeight + 2 * conv.padH - conv.kH) / conv.dH + 1
    val outputWidth = (conv.inputWidth + 2 * conv.padW - conv.kW) / conv.dW + 1
    val size = conv.nInputPlane * conv.kH * conv.kW * outputHeight * outputWidth * 4 / 1024.0

    val output = new Array[Array[Float]](cores)
    val dataCol = new Array[Array[Float]](cores)
    val weight = new Array[Array[Float]](cores)
    val results = new Array[Future[Unit]](cores)

    for (i <- 0 until cores) {
      output(i) = new Array(conv.nOutputPlane * outputHeight * outputWidth)
      dataCol(i) = new Array(conv.nInputPlane * conv.kW * conv.kH * outputHeight * outputWidth)
      weight(i) = new Array(conv.nOutputPlane * conv.nInputPlane * conv.kH * conv.kW)
    }

    val m = conv.nOutputPlane
    val k = conv.nInputPlane * conv.kH * conv.kW
    val n = outputHeight * outputWidth

    val lda = m
    val ldb = k
    val ldc = m

    var oneImageTime = 0.0
    oneImageTime += time {
      var i = 0
      while (i < cores) {
        val _i = i
        results(i) = Future {
          MKL.vsgemm('N', 'N', m, n, k, 1, weight(_i), 0, lda, dataCol(_i), 0, ldb,
            0, output(_i), 0, ldc)
        }

        i += 1
      }

      results foreach { f =>
          Await.result(f, Duration.Inf)
      }
    }

    var oneChannel = 0.0
    for (i <- 0 until cores) {
      weight(i) = new Array[Float](conv.nOutputPlane * conv.kH * conv.kW)
    }

    oneChannel += time {
       var i = 0
       while (i < cores) {
         val _i = i

         var j = 0
         results(i) = Future{
           while (j < conv.nInputPlane) {
             val m = conv.nOutputPlane
             val k = conv.kH * conv.kW
             val n = outputHeight * outputWidth

             val lda = m
             val ldb = k
             val ldc = m

             val _j = j
             MKL.vsgemm('N', 'N', m, n, k, 1, weight(_i), 0, lda, dataCol(_i),
               _j * conv.kH * conv.kW * outputHeight * outputWidth, ldb, 0, output(_i), 0, ldc)
             j += 1
           }
         }

         i += 1
       }


      results foreach { f =>
        Await.result(f, Duration.Inf)
      }
    }

    println(s"$conv\t$size\t${"%.3f".format(oneImageTime)}\t${"%.3f".format(oneChannel)}")
  }

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      println("usage: cmd ...")
      System.exit(1)
    }

    val Array(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, batchSize,
      inputHeight, inputWidth) = args.map(_.toInt)

    val conv = Conv(nInputPlane,
      nOutputPlane,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      batchSize,
      inputHeight,
      inputWidth)

    one(conv)
  }
  //    val m = 3
  //    val k = 2
  //    val n = 3
  //
  //    val lda = m
  //    val ldb = k
  //    val ldc = m
  //
  //    val a = Array[Float](1, 2, 3, 4, 5, 6)
  //    val b = Array[Float](7, 8, 9, 10, 11, 12)
  //    val c = new Array[Float](9)
  //    java.util.Arrays.fill(c, 0.0f)
  //
  //    MKL.vsgemm(
  //      'N',
  //      'N',
  //      m,
  //      n,
  //      k,
  //      1,
  //      a,
  //      0,
  //      lda,
  //      b,
  //      0,
  //      ldb,
  //      0,
  //      c,
  //      0,
  //      ldc
  //    )

  // the result should be
  // 39 49 69
  // 54 68 82
  // 69 87 105
  //    println(c.mkString("\t"))

}
