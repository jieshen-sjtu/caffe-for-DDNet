/*
 * ddnet_data_layer.cpp
 *
 *  Created on: Mar 13, 2014
 *      Author: jieshen
 */

#include <caffe/vision_layers.hpp>
#include <caffe/util/arrange_data.hpp>

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <mkl.h>

namespace caffe
{
  template<typename Dtype>
  PatchLayer<Dtype>::PatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        tmp_(NULL),
        img_height_(0),
        img_width_(0),
        img_channels_(0),
        img_size_(0),
        all_num_patches_(0),
        patch_height_(0),
        patch_width_(0),
        patch_channels_(0),
        patch_size_(0)
  {
    dsift_step_ = param.dsift_step();
    const int num_sz = param.dsift_sizes().size();
    dsift_sizes_.resize(num_sz, 0);
    for (int i = 0; i < num_sz; ++i)
      dsift_sizes_[i] = param.dsift_sizes(i);

    dsift_std_size_ = param.dsift_std_size();

    dsift_off_.resize(num_sz, 0);
    dsift_start_x_.resize(num_sz, 0);
    dsift_start_y_.resize(num_sz, 0);

    const uint32_t max_sz = *(std::max_element(dsift_sizes_.begin(),
                                               dsift_sizes_.end()));
    for (int i = 0; i < num_sz; ++i)
    {
      const uint32_t sz = dsift_sizes_[i];
      dsift_off_[i] = std::floor(1.5 * (max_sz - sz));
      dsift_start_x_[i] = dsift_off_[i] + 1.5 * sz;
      dsift_start_y_[i] = dsift_off_[i] + 1.5 * sz;
    }

    dsift_num_patches_.resize(dsift_sizes_.size(), 0);
  }

  template<typename Dtype>
  void PatchLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 1)<<"PatchLayer takes one blob as input";

    CHECK_EQ(top->size(), 1) << "PatchLayer takes one blob as output";

    img_channels_ = bottom[0]->channels();
    img_height_ = bottom[0]->height();
    img_width_ = bottom[0]->width();
    img_size_ = img_channels_ * img_height_ * img_width_;

    patch_height_ = dsift_std_size_;
    patch_width_ = dsift_std_size_;
    patch_channels_ = img_channels_;
    patch_size_ = patch_height_ * patch_width_ * patch_channels_;

    if(img_channels_ == 1)
    tmpimg_.create(img_height_, img_width_, CV_32FC1);
    else if (img_channels_ == 3)
    tmpimg_.create(img_height_, img_width_, CV_32FC3);
    else
    LOG(FATAL) << "channel must be either 1 or 3";

    tmp_ = (float*)malloc(sizeof(float) * img_channels_ *
        dsift_std_size_ * dsift_std_size_);
    memset(tmp_, 0, sizeof(float) * img_channels_ *
        dsift_std_size_ * dsift_std_size_);

    const int num_sz = dsift_sizes_.size();
    dsift_end_x_.resize(num_sz, 0);
    dsift_end_y_.resize(num_sz, 0);

    for(int i = 0; i < dsift_sizes_.size(); ++i)
    {
      dsift_end_x_[i] = img_width_ - 1.5 * dsift_sizes_[i];
      dsift_end_y_[i] = img_height_ - 1.5 * dsift_sizes_[i];

      const int num_x = std::ceil((dsift_end_x_[i] - dsift_start_x_[i])
          * 1.0 / dsift_step_);
      const int num_y = std::ceil((dsift_end_y_[i] - dsift_start_y_[i])
          * 1.0 / dsift_step_);

      dsift_num_patches_[i] = num_x * num_y;

      all_num_patches_ += dsift_num_patches_[i];
    }

    // patch data
    (*top)[0]->Reshape(bottom[0]->num() * all_num_patches_, img_channels_,
        dsift_std_size_, dsift_std_size_);
    //(*top)[1]->Reshape(bottom[0]->num() * num_patches, llc_dim, 1, 1);

  }

  template<typename Dtype>
  void PatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      vector<Blob<Dtype>*>* top)
  {
    const Dtype* const bottom_data = bottom[0]->cpu_data();
    Dtype* const top_data = (*top)[0]->mutable_cpu_data();

    float* const tmp = (float*) tmpimg_.data;

    for (int imgid = 0; imgid < bottom[0]->num(); ++imgid)
    {
      const float* const imgdata = reinterpret_cast<const float*>(bottom_data)
          + imgid * img_size_;

      channel_to_img(imgdata, img_height_, img_width_, img_channels_, tmp);

      Dtype* const cur_top = top_data + imgid * all_num_patches_ * patch_size_;

      forward_patch(tmpimg_, reinterpret_cast<float*>(cur_top));
    }
  }

  template<typename Dtype>
  Dtype PatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const bool propagate_down,
                                        vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.0);
  }

  template<typename Dtype>
  void PatchLayer<Dtype>::forward_patch(const cv::Mat& img, float* const top)
  {

    const int num_sz = dsift_sizes_.size();

    int out_idx(0);

    for (int sidx = 0; sidx < num_sz; ++sidx)
    {
      const int sz = dsift_sizes_[sidx];
      const float scale = 1.0 * dsift_std_size_ / sz;
      cv::Mat newimg;
      cv::resize(img, newimg, cv::Size(0, 0), scale, scale);

      const int step = dsift_step_ * dsift_std_size_ / sz;
      const int xmin = dsift_start_x_[sidx] * dsift_std_size_ / sz;
      const int ymin = dsift_start_y_[sidx] * dsift_std_size_ / sz;
      const int xmax = dsift_end_x_[sidx] * dsift_std_size_ / sz;
      const int ymax = dsift_end_y_[sidx] * dsift_std_size_ / sz;

      float* const out = top + out_idx;
      out_idx += dsift_num_patches_[sidx] * patch_size_;

      int pidx(0);

      for (int y = ymin; y < ymax; y += step)
      {
        cv::Mat rowimg = newimg.rowRange(y - dsift_std_size_ / 2,
                                         y + dsift_std_size_ / 2);
        for (int x = xmin; x < xmax; x += step)
        {
          cv::Mat patch = rowimg.colRange(x - dsift_std_size_ / 2,
                                          x + dsift_std_size_ / 2);

          img_to_channel((float*) patch.data, patch_height_, patch_width_,
                         patch_channels_, out + pidx * patch_size_);

          ++pidx;
        }  //x
      }  //y
    }  //sidx
  }

  INSTANTIATE_CLASS(PatchLayer);
}
