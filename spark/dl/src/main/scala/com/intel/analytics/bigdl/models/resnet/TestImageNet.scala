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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelScaledNormalizer, RandomCropper, RandomResize}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object TestImageNet {
  System.setProperty("bigdl.mkldnn.fusion", "true")

  import com.intel.analytics.bigdl.models.resnet.Utils._
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)


  val imageSize = 224

  def main(args: Array[String]) {
    testParser.parse(args, new TestParams()).foreach { param =>
      val batchSize = param.batchSize
      val conf = Engine.createSparkConf().setAppName("Test VGG16 on ImageNet")
      val sc = new SparkContext(conf)
      Engine.init

      // We set partition number to be node*core, actually you can also assign other partitionNum
      val partitionNum = Engine.nodeNumber()
      val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(param.folder, sc, 1000,
        partitionNum = Option(partitionNum))
      val transformer =
        PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, false, CropCenter) ->
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float]() ->
          ImageFrameToSample[Float](inputKeys = Array("imageTensor"), targetKeys = Array("label"))
      imageFrame -> transformer

      val model = if (param.quantize) {
        Module.load[Float](param.model).quantize()
      } else {
        Module.load[Float](param.model)
      }

      val result = model.evaluateImage(
        imageFrame,
        Array(new Top1Accuracy[Float](), new Top5Accuracy[Float]()),
        Some(param.batchSize))
      result.foreach(r => {
        Logger.getLogger(getClass).info(s"${ r._2 } is ${ r._1 }")
      })

      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    }
  }
}
