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

package com.intel.analytics.bigdl.example.quantization

import java.io.{File, FileOutputStream, PrintWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, Sample, Transformer}
import com.intel.analytics.bigdl.example.quantization.Utils.loadMeanFile
import com.intel.analytics.bigdl.example.quantization.transformers._
import com.intel.analytics.bigdl.models.lenet.{Utils => LeNetUtils}
import com.intel.analytics.bigdl.models.resnet.{Cifar10DataSet => ResNetCifar10DataSet, Utils => ResNetUtils}
import com.intel.analytics.bigdl.models.vgg.{Utils => VggUtils}
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.transform.vision.image.MatToFloats
import com.intel.analytics.zoo.transform.vision.image.augmentation.{CenterCrop, Resize}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scopt.OptionParser

object Utils {
  case class TestParams(folder: String = "./",
    model: String = "unkonwn_model",
    modelPath: String = "",
    batchSize: Int = 30,
    quantize: Boolean = true)

  val testParser = new OptionParser[TestParams]("BigDL Models Test with Quant") {
    opt[String]('f', "folder")
            .text("Where's the data location?")
            .action((x, c) => c.copy(folder = x))
    opt[String]("modelPath")
            .text("Where's the model location?")
            .action((x, c) => c.copy(modelPath = x))
            .required()
    opt[String]("model")
            .text("What's the model?")
            .action((x, c) => c.copy(model = x))
            .required()
    opt[Int]('b', "batchSize")
            .text("How many samples in a bach?")
            .action((x, c) => c.copy(batchSize = x))
    opt[Boolean]('q', "quantize")
            .text("Quantize the model?")
            .action((x, c) => c.copy(quantize = x))

  }

  def getRddData(model: String, sc: SparkContext, partitionNum: Int,
    folder: String): RDD[ByteRecord] = {
    model match {
      case "lenet" => LeNet.rdd(folder, sc, partitionNum)
      case "vgg" => Cifar.rdd(folder, sc, partitionNum)
      case m if m.contains("alexnet") => ImageNet.rdd(folder, sc, partitionNum)
      case m if m.contains("inception_v1") || m.contains("googlenet") => ImageNet.rdd(folder,
        sc, partitionNum)
      case "inception_v2" => ImageNet.rdd(folder, sc, partitionNum)
      case m if m.toLowerCase.contains("resnet") && !m.toLowerCase.contains("cifar10") =>
        ImageNet.rdd(folder, sc, partitionNum)
      case m if m.toLowerCase.contains("resnet") && m.toLowerCase.contains("cifar10") =>
        Cifar.rdd(folder, sc, partitionNum)
      case "vgg_16" => ImageNet.rdd(folder, sc, partitionNum)
      case "vgg_19" => ImageNet.rdd(folder, sc, partitionNum)

      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  def getTransformer(model: String): Transformer[ByteRecord, Sample[Float]] = {
    model match {
      case "lenet" => LeNet.transformer()
      case "vgg_on_cifar" => Cifar.transformer()
      case m if m.contains("alexnet") => ImageNet.caffe(227, 227, true)
      case m if m.contains("inception_v1") || m.contains("googlenet") =>
        ImageNet.caffe(224, 224, withMean = false)
      case "inception_v2" =>
        ImageNet.caffe(224, 224, withMean = false)
      case m if m.toLowerCase.contains("resnet") && !m.toLowerCase.contains("cifar10") =>
        ImageNet.torch(224, 224)
      case m if m.toLowerCase.contains("resnet") && m.toLowerCase.contains("cifar10") =>
        Cifar.transformer()
      case m if "vgg_16" == m || "vgg_19" == m => ImageNet.caffe(224, 224, withMean = false)
      case _ => throw new UnsupportedOperationException(s"unknown model: $model")
    }
  }

  def time[R](block: => R): (R, Double) = {
    val start = System.nanoTime()
    val result = block
    val end = System.nanoTime()
    (result, (end - start) / 1e9)
  }

  def test(model: Module[Float], evaluationSet: RDD[Sample[Float]], batchSize: Int)
  : Array[(ValidationResult, ValidationMethod[Float])] = {
    println(model)
    val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float],
      new Top5Accuracy[Float]), Some(batchSize))
    result.foreach(r => println(s"${r._2} is ${r._1}"))
    result
  }

  def writeToLog(model: String, quantized: Boolean, totalNum: Int, accuracies: Array[Float],
    costs: Double): Unit = {
    val name = Paths.get(System.getProperty("user.dir"), "model_inference.log").toString
    val file = new File(name)

    val out = if (file.exists() && !file.isDirectory) {
      new PrintWriter(new FileOutputStream(new File(name), true))
    } else {
      new PrintWriter(name)
    }

    out.append(model)
    if (quantized) {
      out.append("\tQuantized")
    } else {
      out.append("\tMKL")
    }
    out.append("\t" + totalNum.toString)
    accuracies.foreach(a => out.append(s"\t${a}"))
    out.append(s"\t${costs}")
    out.append("\n")
    out.close()
  }

  def loadMeanFile(path: String): Tensor[Float] = {
    val lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)
    val array = new Array[Float](lines.size())

    lines.toArray.zipWithIndex.foreach {x =>
      array(x._2) = x._1.toString.toFloat
    }

    Tensor[Float](array, Array(array.length))
  }
}

object ImageNet {
  def caffe(outputHeight: Int, outputWidth: Int,
    withMean: Boolean): Transformer[ByteRecord, Sample[Float]] = {
    if (!withMean) {
      BytesToMat() ->
        Resize(256, 256) ->
        CenterCrop(outputHeight, outputWidth) ->
        MatToFloats(outputHeight, outputWidth, meanRGB = Some(123f, 117f, 104f)) ->
        FeatureToSample(toRGB = false)
    } else {
      val name = Paths.get(System.getProperty("user.dir"), "mean.txt").toString
      val means = loadMeanFile(name)
      BytesToMat() ->
        Resize(256, 256) ->
        MatNormWithMeanFile(means) ->
        CenterCrop(outputHeight, outputWidth) ->
        MatToFloats(outputHeight, outputWidth) ->
        FeatureToSample(toRGB = false)
    }
  }

  def torch(outputHeight: Int,
    outputWidth: Int): Transformer[ByteRecord, Sample[Float]] = {
    BytesToBGRImg() -> BGRImgCropper(224, 224, CropCenter) ->
      HFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) ->
      BGRImgToSample()
  }

  def rdd(folder: String, sc: SparkContext, partitionNum: Int): RDD[ByteRecord] =
    DataSet.SeqFileFolder.filesToRdd(folder, sc, 1000)
}

object LeNet {
  def rdd(folder: String, sc: SparkContext, partitionNum: Int): RDD[ByteRecord] = {
    val validationData = folder + "/t10k-images-idx3-ubyte"
    val validationLabel = folder + "/t10k-labels-idx1-ubyte"
    sc.parallelize(LeNetUtils.load(validationData, validationLabel), partitionNum)
  }

  def transformer(): Transformer[ByteRecord, Sample[Float]] = {
    BytesToGreyImg(28, 28) -> GreyImgNormalizer(LeNetUtils.testMean,
      LeNetUtils.testStd) -> GreyImgToSample()
  }
}

object Cifar {
  def rdd(folder: String, sc: SparkContext, partitionNum: Int): RDD[ByteRecord] = {
    sc.parallelize(VggUtils.loadTest(folder), partitionNum)
  }

  def transformer(): Transformer[ByteRecord, Sample[Float]] = {
    BytesToBGRImg() -> BGRImgNormalizer(VggUtils.testMean, VggUtils.testStd) -> BGRImgToSample()
  }
}
