/*
 * spm_layer.cu
 *
 *  Created on: Mar 3, 2014
 *      Author: jieshen
 */

#include <algorithm>
#include <cfloat>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe
{
  template<typename Dtype>
  void SPMLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                              vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 2)<< "SPMLayer takes two blobs as input:"
    <<" code and position";
    CHECK_EQ(top->size(), 1) << "SPMLayer takes a single blob as output.";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "number of bottoms should be equal";

    num_spm_level_ = this->layer_param_.spm_level();
    num_patch_ = this->layer_param_.num_patch();
    hor_pool_ = this->layer_param_.hor_pool();

    const int dim = bottom[0]->channels();
    int channels(0);

    for(int i=0; i<num_spm_level_; ++i)
    channels += std::pow(2, 2 * i);
    channels *= dim;

    // merge the image patches
    (*top)[0]->Reshape(bottom[0]->num() / num_patch_, channels, 1, 1);
  }

  template<typename Dtype>
  void SPMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    vector<Blob<Dtype>*>* top)
  {
    float* top_data = reinterpret_cast<float*>((*top)[0]->mutable_cpu_data());
    const float* bottom_data = reinterpret_cast<const float*>((*bottom)[0]
        ->cpu_data());
    const int* patch_pos =
        reinterpret_cast<const int*>((*bottom)[1]->cpu_data());

    const int num_img = (*top)[0]->num();
    const int top_dim = (*top)[0]->count() / num_img;
    const int bot_dim = (*bottom)[0]->count() / (*bottom)[0]->num();
    const int pos_dim = 2;

    const int num_cell_x = std::sqrt(num_patch_);
    const int num_cell_y = num_cell_x;

    for (int imgid = 0; imgid < num_img; ++imgid)
    {
      float* cur_top_data = top_data + imgid * top_dim;
      const float* cur_bot_data = (bottom_data) + imgid * num_patch_ * bot_dim;
      const int* cur_pos = patch_pos + imgid * num_patch_ * pos_dim;

      // fine to coarse pooling
      for(int lv = num_spm_level_ - 1; lv >= 0; --lv)
      {
        const int spm_patch_x = num_cell_x / std::pow(2, lv);
        const int spm_patch_y = num_cell_y / std::pow(2, lv);


      }
    }
  }

  INSTANTIATE_CLASS(SPMLayer);
}
