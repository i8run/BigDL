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

object GEMM {
  val rand = new scala.util.Random(5)

  def time[R](block: => R): (Double, R) = {
    val start = System.nanoTime()
    val result = block
    val end = System.nanoTime()
//    println("Elapsed time: " + (end - start) / 1e6 + "ms")
    ((end - start) / 1e6, result)
  }

  def fillMatrix(m: Int, n: Int, matrix: Array[Float]): Unit = {
    for (i <- 0 until m * n) {
      matrix(i) = rand.nextFloat()
    }
  }

  def gemm(transA: Char, transB: Char, m: Int, n: Int, k: Int): Unit = {
    val matrixA = new Array[Float](m * k)
    val matrixB = new Array[Float](k * n)
    val matrixC = new Array[Float](m * n)

    val iterations = 10

    val alpha = 1.0f
    val beta = 0.0f

    // load data to cache
    fillMatrix(m, k, matrixA)
    fillMatrix(k, n, matrixB)
    fillMatrix(m, n, matrixC)
    time[Unit] {
      DenseTensorBLAS.gemm[Float]('N', 'N', m, n, k, alpha, matrixA, 0, m, matrixB, 0, n, beta,
        matrixC, 0, m)
    }._1

    // computing time
    var elapsed = 0.0
    for (i <- 0 until iterations) {
      fillMatrix(m, k, matrixA)
      fillMatrix(k, n, matrixB)
      fillMatrix(m, n, matrixC)

      elapsed += time[Unit] {
        DenseTensorBLAS.gemm[Float](
          transA, transB, m, n, k,
          alpha, matrixA, 0, m,
          matrixB, 0, n, beta,
          matrixC, 0, m)
      }._1
    }

    println(s"$transA, $transB -- Elapsed time: ${elapsed / iterations} ms")
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
    val m = args(1).toInt
    val n = args(2).toInt
    val k = args(3).toInt

    gemm('N', 'N', m, n, k)
  }
}

