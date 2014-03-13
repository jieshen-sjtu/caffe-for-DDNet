/*
 * ddnet_data_layer.cpp
 *
 *  Created on: Mar 13, 2014
 *      Author: jieshen
 */

#include <caffe/vision_layers.hpp>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <mkl.h>

namespace caffe
{
  template<typename Dtype>
  PatchCodeLayer<Dtype>::PatchCodeLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        tmp_data_(NULL)
  {
    dsift_step_ = param.dsift_step_;
    dsift_sizes_.resize(param.dsift_sizes().size(), 0);
    for (int i = 0; i < dsift_sizes_.size(); ++i)
      dsift_sizes_[i] = param.dsift_sizes(i);

    dsift_std_size_ = param.dsift_std_size();

    dsift_off_.resize(dsift_sizes_.size(), 0);
    const uint32_t max_sz = std::max_element(dsift_sizes_.begin(),
                                             dsift_sizes_.end());
    for (int i = 0; i < dsift_off_.size(); ++i)
    {
      const uint32_t sz = dsift_sizes_[i];
      dsift_off_[i] = std::floor(1.5 * (max_sz - sz));
    }

    dsift_num_patches_.resize(dsift_sizes_.size(), 0);

    dsift_model_.set_step(dsift_step_);

    // LLC
    shared_ptr<float> centers;
    uint32_t cb_dim(0), K(0);
    {
      LOG(INF) << "start loading the codebook...";

      FILE* pfile = fopen(param.codebook().c_str(), "r");

      EYE::CodeBook::load(pfile, centers, &cb_dim, &K);
      fclose(pfile);
    }

    llc_model_.set_base(centers, cb_dim, K);
    llc_model_.SetUp();
  }

  template<typename Dtype>
  void PatchCodeLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                    vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 1)<<"DDNetDataLayer takes one blob as input";

    CHECK_EQ(top->size(), 2) << "DDNetDataLayer takes two blobs as output";

    img_channels_ = bottom[0]->channels();
    img_height_ = bottom[0]->height();
    img_width_ = bottom[0]->width();
    img_size_ = img_channels_ * img_height_ * img_width_;

    int num_patches(0);

    for(int i = 0; i < dsift_sizes_.size(); ++i)
    {
      const int num_x = std::ceil((img_width_ - dsift_off_[i] -
              2 * 1.5 * dsift_sizes_[i]) * 1.0 / dsift_step_);
      const int num_y = std::ceil((img_height_ - dsift_off_[i] -
              2 * 1.5 * dsift_sizes_[i]) * 1.0 / dsift_step_);
      dsift_num_patches_[i] = num_x * num_y;

      num_patches += dsift_num_patches_[i];
    }

    const int llc_dim = llc_model_.get_num_base();
    // patch data
    (*top)[0]->Reshape(bottom[0]->num() * num_patches, img_channels_, dsift_std_size_, dsift_std_size_);
    (*top)[1]->Reshape(bottom[0]->num() * num_patches, llc_dim, 1, 1);

    tmp_data_ = (Dtype*)malloc(sizeof(Dtype) * img_channels_ * img_height_ * img_width_);
    memset(tmp_data_, 0, sizeof(Dtype) * img_channels_ * img_height_ * img_width_);
  }

  template<typename Dtype>
  void PatchCodeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          vector<Blob<Dtype>*>* top)
  {
    const float* const bottom_data = reinterpret_cast<const float*>(bottom[0]
        ->cpu_data());
    float* const tmp = reinterpret_cast<float*>(tmp_data_);

    const int sz_c = img_height_ * img_width_;

    for (int imgid = 0; imgid < bottom[0]->num(); ++imgid)
    {
      const float* const imgdata = bottom_data + imgid * img_size_;
      for (int c = 0; c < img_channels_; ++c)
        cblas_scopy(sz_c, imgdata + c * sz_c, 1, tmp + c, img_channels_);

      if (img_channels_ == 1)
      {
        cv::Mat img(img_height_, img_width_, CV_32FC1, imgdata);
      }
      else if (img_channels_ == 3)
      {
        cv::Mat img(img_height_, img_width_, CV_32FC3, imgdata);
      }

    }
  }
}
