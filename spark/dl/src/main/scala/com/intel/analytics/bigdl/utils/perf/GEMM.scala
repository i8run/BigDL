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

package com.intel.analytics.bigdl.utils.perf

import com.intel.analytics.bigdl.tensor.DenseTensorBLAS
import com.intel.analytics.bigdl.utils.ThreadPool

object GEMM {
  val iterations = 10

  def time[R](block: => R): (Double, R) = {
    val start = System.nanoTime()
    val result = block
    val end = System.nanoTime()
//    println("Elapsed time: " + (end - start) / 1e6 + "ms")
    ((end - start) / 1e6, result)
  }

  def fillMatrix(length: Int, matrix: Array[Float]): Unit = {
    val rand = new scala.util.Random(5)

    for (i <- 0 until length) {
      matrix(i) = rand.nextFloat()
    }
  }

  def gemm(transA: Char, transB: Char, m: Int, n: Int, k: Int, batchSize: Int): Double = {
    val matrixA = new Array[Float](m * k * batchSize)
    val matrixB = new Array[Float](k * n * batchSize)
    val matrixC = new Array[Float](m * n * batchSize)

    val alpha = 1.0f
    val beta = 0.0f

    // load data to cache
    fillMatrix(m * k * batchSize, matrixA)
    fillMatrix(k * n * batchSize, matrixB)
    fillMatrix(m * n * batchSize, matrixC)
    for (i <- 0 until batchSize) {
      time[Unit] {
        DenseTensorBLAS.gemm[Float](
          'N', 'N', m, n, k,
          alpha, matrixA, i * m * k, m,
          matrixB, i * k * n, k, beta,
          matrixC, i * m * n, m)
      }._1
    }
    // computing time
    var elapsed = 0.0
    for (i <- 0 until iterations) {
      fillMatrix(m * k * batchSize, matrixA)
      fillMatrix(k * n * batchSize, matrixB)
      fillMatrix(m * n * batchSize, matrixC)

      for (j <- 0 until batchSize) {
        elapsed += time[Unit] {
          DenseTensorBLAS.gemm[Float](
            'N', 'N', m, n, k,
            alpha, matrixA, j * m * k, m,
            matrixB, j * k * n, k, beta,
            matrixC, j * m * n, m)
        }._1
      }
    }

//    println(s"$m, $n, $k, ${elapsed / iterations} ms")
    elapsed / iterations
/*
    time[Unit] {
      DenseTensorBLAS.gemm[Float]('T', 'N', m, n, k, alpha, matrixA, 0, K, matrixB, 0, K, beta,
        matrixC, 0, M)
      matrixC.foreach(v => print(v + " "))
    }

    time[Unit] {
      DenseTensorBLAS.gemm[Float]('N', 'T', m, n, k, alpha, matrixA, 0, M, matrixB, 0, N, beta,
        matrixC, 0, M)
      matrixC.foreach(v => print(v + " "))
    }

    time[Unit] {
      DenseTensorBLAS.gemm[Float]('T', 'T', m, n, k, alpha, matrixA, 0, K, matrixB, 0, N, beta,
        matrixC, 0, M)
      matrixC.foreach(v => print(v + " "))
    }
    */
  }

  def main(args: Array[String]) {
//    println(args.mkString("\t"))
    val m = args(0).toInt
    val n = args(1).toInt
    val k = args(2).toInt
    val batchSize = if (args.length >= 4) {
      args(3).toInt // batchsize = 4
    } else {
      1 // batchsize = 1
    }

    if (args.length > 5 && args(4).startsWith("singleThread")) {
      val elapsed = gemm('N', 'N', m, n, k, batchSize)
      println(s"$m, $n, $k, ${elapsed} ms")
    } else {
      val processors = Runtime.getRuntime.availableProcessors() / 2
      val threads = new ThreadPool(processors)
      val elapsedArray = new Array[Double](processors)

      val start = System.nanoTime()
      threads.invokeAndWait {
        (0 until processors).map {
          i => () => {
            elapsedArray(i) = gemm('N', 'N', m, n, k, batchSize)
          }
        }
      }
      val end = System.nanoTime()

      elapsedArray.foldLeft(0.0)((x, y) => x + y)

      println(s"$m, $n, $k, ${(end - start) / 1e6 / iterations} ms")
    }
  }
}

