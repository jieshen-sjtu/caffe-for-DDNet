/*
 * sgn.hpp
 *
 *  Created on: Mar 16, 2014
 *      Author: jieshen
 */

#ifndef __caffe_SGN_HPP__
#define __caffe_SGN_HPP__

namespace caffe
{
  template<typename T>
  int sgn(T val)
  {
    return (T(0) < val) - (val < T(0));
  }
}

#endif /* __caffe_SGN_HPP__ */
