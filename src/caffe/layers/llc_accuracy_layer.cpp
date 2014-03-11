/*
 * llc_accuracy_layer.cpp
 *
 *  Created on: Mar 1, 2014
 *      Author: jieshen
 */

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe
{
  const double kLOG_THRESHOLD = 1e-20;

  template<typename Dtype>
  void LLCAccuracyLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                      vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 2)<< "Accuracy Layer takes two blobs as input.";
    CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";

    // check the label
    CHECK_GT(bottom[1]->channels(), 0);// llc code dim
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    (*top)[0]->Reshape(1, 2, 1, 1);// only compute the loss and log(loss)

    difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
  }

  template<typename Dtype>
  void LLCAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            vector<Blob<Dtype>*>* top)
  {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();

    caffe_sub(bottom[0]->count(), bottom_data, bottom_label,
              difference_.mutable_cpu_data());

    Dtype euclid_loss = 0.5 / num
        * caffe_cpu_dot(bottom[0]->count(), difference_.cpu_data(),
                        difference_.cpu_data());

    // LOG(INFO) << "Accuracy: " << accuracy;
    (*top)[0]->mutable_cpu_data()[0] = euclid_loss;
    (*top)[0]->mutable_cpu_data()[1] = -1.0
        * log(max((double) euclid_loss, kLOG_THRESHOLD));
  }

  INSTANTIATE_CLASS(LLCAccuracyLayer);
}
