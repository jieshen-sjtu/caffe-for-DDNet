/*
 * zjq_context_layer.cpp
 *
 *  Created on: Mar 19, 2014
 *      Author: jieshen
 */

#include <mkl.h>
#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

  template<typename Dtype>
  void ZJQContextLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                     vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 2)<<"Context Layer takes two blobs as input.";
    CHECK_EQ(top->size(), 1) << "Context Layer takes a single blob as output.";
    // Figure out the dimensions
    num_feat_map_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    context_dim_ = bottom[1]->count() / bottom[1]->num();

    (*top)[0]->Reshape(bottom[0]->num(), num_feat_map_, height_, width_);
    // Check if we need to set up the weights
    if (this->blobs_.size() > 0)
    {
      LOG(INFO) << "Skipping parameter initialization";
    }
    else
    {
      this->blobs_.resize(1);

      // Intialize the weight
      this->blobs_[0].reset(new Blob<Dtype>(1, num_feat_map_, context_dim_, 1));
      // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(
          GetFiller<Dtype>(this->layer_param_.weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }

    w_multi_context_.Reshape(1, num_feat_map_, 1, 1);

    {
      all_ones_.Reshape(1, 1, height_, width_);
      Dtype* all_ones = all_ones_.mutable_cpu_data();
      for(int i=0; i<all_ones_.count(); ++i)
      {
        all_ones[i] = 1.0;
      }
    }

    {
      all_ones_sample_.Reshape(bottom[0]->num(), 1, 1, 1);
      Dtype* all_one_sample = all_ones_sample_.mutable_cpu_data();
      for(int i=0; i<all_ones_sample_.count(); ++i)
      {
        all_one_sample[i] = 1.0;
      }
    }

    tmp_.Reshape(1, num_feat_map_, height_, width_);

    bias_.Reshape(1, 1, 1, num_feat_map_);

    bias_multiplier_.reset(new SyncedMemory(bottom[0]->num() * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
    reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < bottom[0]->num(); ++i)
    {
      bias_multiplier_data[i] = 1.;
    }
  }

  template<typename Dtype>
  void ZJQContextLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           vector<Blob<Dtype>*>* top)
  {
    const Dtype* const bottom_data = bottom[0]->cpu_data();
    const Dtype* const context = bottom[1]->cpu_data();
    const Dtype* const all_ones = all_ones_.cpu_data();
    const Dtype* const weight = this->blobs_[0]->cpu_data();
    Dtype* const w_multi_context = w_multi_context_.mutable_cpu_data();
    Dtype* const top_data = (*top)[0]->mutable_cpu_data();

    caffe_cpu_gemv(CblasNoTrans, num_feat_map_, context_dim_, 1.0, weight,
                   context, 0, w_multi_context);

    const int num = bottom[0]->num();
    const int dim = bottom[0]->dim();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_feat_map_,
                          height_ * width_, 1, 1.0, w_multi_context, all_ones,
                          0, tmp_.mutable_cpu_data());

    memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, (Dtype) 1.,
                          tmp_.cpu_data(), all_ones_sample_.cpu_data(),
                          (Dtype) 1., top_data);
  }

  template<typename Dtype>
  void ZJQContextLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           vector<Blob<Dtype>*>* top)
  {
    const Dtype* const bottom_data = bottom[0]->gpu_data();
    const Dtype* const context = bottom[1]->gpu_data();
    const Dtype* const all_ones = all_ones_.gpu_data();
    const Dtype* const weight = this->blobs_[0]->gpu_data();
    Dtype* const w_multi_context = w_multi_context_.mutable_gpu_data();
    Dtype* const top_data = (*top)[0]->mutable_gpu_data();

    caffe_gpu_gemv(CblasNoTrans, num_feat_map_, context_dim_, 1.0, weight,
                   context, 0, w_multi_context);

    const int num = bottom[0]->num();
    const int dim = bottom[0]->dim();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_feat_map_,
                          height_ * width_, 1, 1.0, w_multi_context, all_ones,
                          0, tmp_.mutable_gpu_data());

    CUDA_CHECK(
        cudaMemcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count(),
                   cudaMemcpyDeviceToDevice));

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, (Dtype) 1.,
                          tmp_.gpu_data(), all_ones_sample_.gpu_data(),
                          (Dtype) 1., top_data);
  }

  template<typename Dtype>
  Dtype ZJQContextLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const bool propagate_down,
                                             vector<Blob<Dtype>*>* bottom)
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    Dtype* sum_top_dif = sum_top_diff_.mutable_cpu_data();

    const int num = top[0]->num();
    const int dim = top[0]->count() / num;

    // Gradient with respect to weight
    /*
     const Dtype* const all_ones = all_ones_.cpu_data();
     caffe_cpu_gemv<Dtype>(CblasNoTrans, num * num_feat_map_, height_ * width_,
     1.0, top_diff, all_ones, 0, sum_top_dif);

     for (int i = 0; i < num_feat_map_; ++i)
     w_multi_context_.mutable_cpu_data()[i] = 1.0;
     caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * num_feat_map_,
     num_feat_map_, 1, 1.0, sum_top_dif,
     w_multi_context_.cpu_data(), 0,
     multi_context_diff_.mutable_cpu_data());
     */

    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(
        CblasTrans, bottom[0]->num(), num_feat_map_, (Dtype) 1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
        (Dtype) 0., bias_->mutable_cpu_diff());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_feat_map_,
                          context_dim_, 1, 1.0, bias_.cpu_data(),
                          (*bottom)[1]->cpu_data(), 0,
                          this->blobs_[0].mutable_cpu_data());

    if (propagate_down)
    {
      // Gradient with respect to bottom data
      memcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count());
    }
    return Dtype(0);
  }

  template<typename Dtype>
  Dtype ZJQContextLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                             const bool propagate_down,
                                             vector<Blob<Dtype>*>* bottom)
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();

    // Gradient with respect to weight
    caffe_gpu_gemv<Dtype>(
        CblasTrans, bottom[0]->num(), num_feat_map_, (Dtype) 1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype) 0., bias_->mutable_gpu_diff());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_feat_map_,
                          context_dim_, 1, 1.0, bias_.gpu_data(),
                          (*bottom)[1]->gpu_data(), 0,
                          this->blobs_[0].mutable_gpu_data());

    if (propagate_down)
    {
      // Gradient with respect to bottom data
      CUDA_CHECK(
          cudaMemcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count(),
                     cudaMemcpyDeviceToDevice));
    }
    return Dtype(0);
  }

  INSTANTIATE_CLASS(ZJQContextLayer);

}
// namespace caffe

