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

package com.intel.analytics.bigdl.example.quantization.transformers

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature

import scala.collection.Iterator

object FeatureToSample {
  def apply(toRGB: Boolean = true): FeatureToSample = {
    new FeatureToSample(toRGB)
  }
}

class FeatureToSample(toRGB: Boolean = true) extends Transformer[ImageFeature, Sample[Float]] {

  private val featureBuffer = Tensor[Float]()
  private val labelBuffer = Tensor[Float](1)

  override def apply(prev: Iterator[ImageFeature]): Iterator[Sample[Float]] = {
    prev.map(img => {
      labelBuffer.storage.array()(0) = img.getLabel[Float]
      if (featureBuffer.nElement() != 3 * img.getHeight() * img.getWidth()) {
        featureBuffer.resize(3, img.getHeight(), img.getWidth())
      }

      img.copyTo(featureBuffer.storage().array(), 0, toRGB = toRGB)
      Sample(featureBuffer, labelBuffer)
    })
  }
}
