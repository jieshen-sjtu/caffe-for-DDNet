/*
 * arrange_data.hpp
 *
 *  Created on: Mar 14, 2014
 *      Author: jieshen
 */

#ifndef __caffe_ARRANGE_DATA_HPP__
#define __caffe_ARRANGE_DATA_HPP__

namespace caffe
{
  void img_to_channel(const float* const src, const int height, const int width,
                      const int channel, float* const dst);
  void channel_to_img(const float* const src, const int height, const int width,
                      const int channel, float* const dst);
}

#endif /* __caffe_ARRANGE_DATA_HPP__ */
