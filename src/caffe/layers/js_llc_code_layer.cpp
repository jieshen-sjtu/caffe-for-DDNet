/*
 * llc_code_layer.cpp
 *
 *  Created on: Mar 14, 2014
 *      Author: jieshen
 */

/*
 * ddnet_data_layer.cpp
 *
 *  Created on: Mar 13, 2014
 *      Author: jieshen
 */

#include <caffe/vision_layers.hpp>
#include <caffe/util/arrange_data.hpp>

#include <glog/logging.h>

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <mkl.h>

namespace caffe
{
  template<typename Dtype>
  LLCCodeLayer<Dtype>::LLCCodeLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
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

    dsift_model_.set_step(dsift_step_);
    dsift_model_.set_sizes(dsift_sizes_);

    // LLC
    shared_ptr<float> centers;
    uint32_t cb_dim(0), K(0);
    {
      LOG(INFO)<< "start loading the codebook...";

      FILE* pfile = fopen(param.codebook().c_str(), "r");

      EYE::CodeBook::load(pfile, centers, &cb_dim, &K);
      fclose(pfile);
    }

    llc_model_.set_base(centers, cb_dim, K);
    llc_model_.SetUp();
    llc_dim_ = llc_model_.get_num_base();
  }

  template<typename Dtype>
  void LLCCodeLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                  vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 1)<<"DDNetDataLayer takes one blob as input";

    CHECK_EQ(top->size(), 1) << "DDNetDataLayer takes one blob as output";

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

    dsift_model_.SetUp(img_width_, img_height_);

    // LLC code
    (*top)[0]->Reshape(bottom[0]->num() * all_num_patches_, llc_dim_,
        1, 1);
    //(*top)[1]->Reshape(bottom[0]->num() * all_num_patches_, 2, 1, 1);
  }

  template<typename Dtype>
  void LLCCodeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        vector<Blob<Dtype>*>* top)
  {
    const Dtype* const bottom_data = bottom[0]->cpu_data();
    Dtype* const top_data = (*top)[0]->mutable_cpu_data();
    //Dtype* const top_pos = (*top)[1]->mutable_cpu_data();

    float* const tmp = (float*) tmpimg_.data;

    for (int imgid = 0; imgid < bottom[0]->num(); ++imgid)
    {
      const float* const imgdata = reinterpret_cast<const float*>(bottom_data)
          + imgid * img_size_;

      channel_to_img(imgdata, img_height_, img_width_, img_channels_, tmp);

      if (img_channels_ == 3)
        cv::cvtColor(tmpimg_, grayimg_, CV_BGR2GRAY);
      else
        tmpimg_.copyTo(grayimg_);

      Dtype* const cur_top = top_data + imgid * all_num_patches_ * llc_dim_;
      //Dtype* const cur_pos = top_pos + imgid * all_num_patches_ * 2;

      const float* const gray_data = (const float*) grayimg_.data;

      //vector<VlDsiftKeypoint> frames;
      vector<float> descrs;
      uint32_t sift_dim(0);
      dsift_model_.Extract(gray_data, img_width_, img_height_, NULL, &descrs,
                           &sift_dim);
      llc_model_.Encode(descrs.data(), sift_dim, descrs.size() / sift_dim,
                        reinterpret_cast<float*>(cur_top));
      /*
       for (int f = 0; f < frames.size(); ++f)
       {
       cur_pos[2 * f] = static_cast<Dtype>((int) frames[f].x);
       cur_pos[2 * f + 1] = static_cast<Dtype>((int) frames[f].y);
       }*/
    }
  }

  template<typename Dtype>
  Dtype LLCCodeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const bool propagate_down,
                                          vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.0);
  }

  INSTANTIATE_CLASS(LLCCodeLayer);
}

