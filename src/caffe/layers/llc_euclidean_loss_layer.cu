/*
 * llc_euclidean_loss_layer.cu
 *
 *  Created on: Feb 26, 2014
 *      Author: jieshen
 */

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe
{
  template<typename Dtype>
  void LLCEuclideanLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                           vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 2)<< "Loss Layer takes two blobs as input: data, llc codes";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());

    // check if we want fine-tune
    /*
     if(this->start_fine_tune_)
     CHECK_EQ(top->size(), 1) << "LLC Loss Layer takes one output for fine tune";
     else
     CHECK_EQ(top->size(), 0);
     */

    CHECK_EQ(top->size(), 1) << "LLC Loss Layer takes one output for fine tune";
    (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
  }

  template<typename Dtype>
  void LLCEuclideanLossLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
  {
    if (!this->start_fine_tune())
      return;

    // when fine, we just copy the data
    const Dtype* const bottom_data = bottom[0]->cpu_data();
    Dtype* const top_data = (*top)[0]->mutable_cpu_data();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  template<typename Dtype>
  void LLCEuclideanLossLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
  {
    if (!this->start_fine_tune())
      return;

    const Dtype* const bottom_data = bottom[0]->gpu_data();
    Dtype* const top_data = (*top)[0]->mutable_gpu_data();
    caffe_gpu_copy(bottom[0]->count(), bottom_data, top_data);
  }

  template<typename Dtype>
  Dtype LLCEuclideanLossLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const bool propagate_down,
      vector<Blob<Dtype>*>* bottom)
  {
    if (this->start_fine_tune())
    {
      const Dtype* const top_diff = top[0]->cpu_diff();
      Dtype* const bottom_diff = (*bottom)[0]->mutable_cpu_diff();
      caffe_copy(top[0]->count(), top_diff, bottom_diff);

      return Dtype(0.);
    }

    int count = (*bottom)[0]->count();
    int num = (*bottom)[0]->num();
    caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
              difference_.mutable_cpu_data());
    Dtype loss = caffe_cpu_dot(count, difference_.cpu_data(),
                               difference_.cpu_data()) / num / Dtype(2);
    // Compute the gradient
    caffe_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
                (*bottom)[0]->mutable_cpu_diff());
    return loss;
  }

  /*
   template<typename Dtype>
   Dtype LLCEuclideanLossLayer<Dtype>::Backward_gpu(
   const vector<Blob<Dtype>*>& top, const bool propagate_down,
   vector<Blob<Dtype>*>* bottom)
   {
   if (this->start_fine_tune())
   {
   const Dtype* const top_diff = top[0]->gpu_diff();
   Dtype* const bottom_diff = (*bottom)[0]->mutable_gpu_diff();
   caffe_gpu_copy(top[0]->count(), top_diff, bottom_diff);

   return Dtype(0.);
   }

   int count = (*bottom)[0]->count();
   int num = (*bottom)[0]->num();
   caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
   difference_.mutable_cpu_data());
   Dtype loss = caffe_cpu_dot(count, difference_.cpu_data(),
   difference_.cpu_data()) / num / Dtype(2);
   // Compute the gradient
   caffe_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
   (*bottom)[0]->mutable_cpu_diff());
   return loss;
   }
   */

  INSTANTIATE_CLASS(LLCEuclideanLossLayer);
}
