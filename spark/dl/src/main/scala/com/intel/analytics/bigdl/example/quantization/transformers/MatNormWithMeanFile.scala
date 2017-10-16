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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.transform.vision.image.{FeatureTransformer, ImageFeature}
import org.opencv.core.CvType

class MatNormWithMeanFile(means: Tensor[Float]) extends FeatureTransformer {

  var content: Array[Float] = _
  override protected def transformMat(feature: ImageFeature): Unit = {
    val mat = feature.opencvMat()
    if (content == null || content.length < mat.width() * mat.height() * 3) {
      content = new Array[Float](mat.width() * mat.height() * 3)
    }
    mat.convertTo(mat, CvType.CV_32FC3)
    mat.get(0, 0, content)
    val meansData = means.storage().array()
    require(content.length % 3 == 0)
    require(content.length == means.nElement())
    var i = 0
    while (i < content.length) {
      content(i + 2) = content(i + 2) - meansData(i + 2)
      content(i + 1) = content(i + 1) - meansData(i + 1)
      content(i + 0) = content(i + 0) - meansData(i + 0)
      i += 3
    }
    mat.put(0, 0, content)
  }
}

object MatNormWithMeanFile {
  def apply(means: Tensor[Float]): MatNormWithMeanFile = new MatNormWithMeanFile(means)
}

