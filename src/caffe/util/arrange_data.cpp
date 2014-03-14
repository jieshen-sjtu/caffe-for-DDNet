/*
 * arrange_data.cpp
 *
 *  Created on: Mar 14, 2014
 *      Author: jieshen
 */

#include <caffe/util/arrange_data.hpp>
#include <mkl.h>

namespace caffe
{
  void img_to_channel(const float* const src, const int height, const int width,
                      const int channel, float* const dst)
  {
    const int sz = height * width;

    for (int c = 0; c < channel; ++c)
    {
      cblas_scopy(sz, src + c, channel, dst + c * sz, 1);
    }
  }

  void channel_to_img(const float* const src, const int height, const int width,
                      const int channel, float* const dst)
  {
    const int sz = height * width;

    for (int c = 0; c < channel; ++c)
    {
      cblas_scopy(sz, src + c * sz, 1, dst + c, channel);
    }
  }

}
