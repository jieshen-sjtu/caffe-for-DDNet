/*
 * js_l1_loss_layer.cpp
 *
 *  Created on: Mar 16, 2014
 *      Author: jieshen
 */

#include <caffe/vision_layers.hpp>
#include <caffe/util/math_functions.hpp>
#include <caffe/util/sgn.hpp>
#include <mkl.h>

#include <iostream>
using std::cout;
using std::endl;

namespace caffe
{

  template<typename Dtype>
  void L1LossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 2)<< "Loss Layer takes two blobs as input.";
    CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
  }

  template<typename Dtype>
  Dtype L1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const bool propagate_down,
                                         vector<Blob<Dtype>*>* bottom)
  {
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const int dim = count / num;
    caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
              difference_.mutable_cpu_data());
    const float* const difference = reinterpret_cast<const float*>(difference_
        .cpu_data());
    Dtype loss = 1.0 / num * cblas_sasum(count, difference, 1);
    // Compute the gradient
    float* grad = reinterpret_cast<float*>((*bottom)[0]->mutable_cpu_diff());
    for (int i = 0; i < count; ++i)
      grad[i] = sgn(difference[i]);

    //LOG(INFO) << "L1 bp";
    for (int i = 0; i < num; ++i)
    {
      for (int j = 0; j < dim; ++j)
      {
        cout << (*bottom)[1]->cpu_data()[i * dim + j] << " ";
      }
      cout << endl;
      for (int j = 0; j < dim; ++j)
      {
        cout << (*bottom)[0]->cpu_data()[i * dim + j] << " ";
      }
      cout << endl;
      for (int j = 0; j < dim; ++j)
      {
        cout << grad[i * dim + j] << " ";
      }
      cout << endl;
    }
    /*
    for (int i = 0; i < count; ++i)
    {
      cout << (*bottom)[1]->cpu_data()[i] << " ";
      if ((i + 1) % dim == 0)
        cout << endl;
    }
    for (int i = 0; i < count; ++i)
    {
      cout << (*bottom)[0]->cpu_data()[i] << " ";
      if ((i + 1) % dim == 0)
        cout << endl;
    }
    for (int i = 0; i < count; ++i)
    {
      cout << grad[i] << " ";
      if ((i + 1) % dim == 0)
        cout << endl;
    }*/

    cblas_sscal(count, 1.0 / num, grad, 1);

    return loss;
  }

  INSTANTIATE_CLASS(L1LossLayer);
}

