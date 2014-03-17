/*
 * spm_layer.cpp
 *
 *  Created on: Mar 3, 2014
 *      Author: jieshen
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::abs;

namespace caffe
{
  const float FLT_THRD = 1e-10;

  template<typename Dtype>
  void SPMLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                              vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 2)<< "SPMLayer takes two blobs as input:"
    <<" code and position";
    CHECK_EQ(top->size(), 1) << "SPMLayer takes a single blob as output.";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "number of bottoms should be equal";

    num_img_ = this->layer_param_.batchsize();
    //build_spm();
    num_spm_level_ = this->layer_param_.spm_level();

    feat_dim_ = (bottom)[0]->count() / (bottom)[0]->num();
    img_width_ = this->layer_param_.img_width();
    img_height_ = this->layer_param_.img_height();

    spm_model_.set_num_spm_level(num_spm_level_);
    spm_model_.set_same_geom(true);
    spm_model_.SetUp(img_width_, img_height_);

    const float* const pos = reinterpret_cast<const float*>(bottom[1]->cpu_data());
    num_patch_ = bottom[1]->num() / num_img_;
    spm_model_.build_cell_blk_map(pos, num_patch_);

    map_cell_blk_ = spm_model_.get_map_cell_blks();
    map_blk_cell_ = spm_model_.get_map_blk_cells();

    spm_dim_ = feat_dim_ * spm_model_.get_total_num_blk();

    // merge the image patches
    (*top)[0]->Reshape(num_img_, spm_dim_, 1, 1);
  }

  /*
   template<typename Dtype>
   void SPMLayer<Dtype>::build_spm()
   {
   num_spm_level_ = this->layer_param_.spm_level();
   num_patch_ = this->layer_param_.num_patch();
   hor_pool_ = this->layer_param_.hor_pool();

   num_cell_x_y_ = std::sqrt(num_patch_);
   finest_num_blk_ = std::pow(2, num_spm_level_ - 1);
   finest_num_cell_ = num_cell_x_y_ / finest_num_blk_;

   CHECK_GE(num_patch_, finest_num_cell_ * finest_num_cell_);

   // build the level statistics
   {
   level_num_blk_.resize(num_spm_level_, 0);
   level_start_idx_.resize(num_spm_level_, 0);

   level_num_blk_[num_spm_level_ - 1] = finest_num_blk_;
   level_start_idx_[num_spm_level_ - 1] = 0;

   for (int lv = num_spm_level_ - 2; lv >= 0; --lv)
   {
   const int prev_lv = lv + 1;
   level_num_blk_[lv] = level_num_blk_[prev_lv] / 2;
   level_start_idx_[lv] = level_start_idx_[prev_lv]
   + level_num_blk_[prev_lv] * level_num_blk_[prev_lv];
   }
   }

   // build the map from cell to block
   for (int ycell = 0; ycell < num_cell_x_y_; ++ycell)
   for (int xcell = 0; xcell < num_cell_x_y_; ++xcell)
   {
   const pair<int, int> cell = std::make_pair(ycell, xcell);
   CHECK(map_cell_blk_start_idx_.find(cell) ==
   map_cell_blk_start_idx_.end()) << "ERROR in the data of position";

   vector<int> v_blk_idx(num_spm_level_, -1);

   for (int lv = 0; lv < num_spm_level_; ++lv)
   {
   const int num_blk = level_num_blk_[lv];
   const int num_cell = num_cell_x_y_ / num_blk;

   const int blk_y_idx = ycell / num_cell;
   const int blk_x_idx = xcell / num_cell;

   const int blk_idx = level_start_idx_[lv] + blk_y_idx * num_blk
   + blk_x_idx;

   v_blk_idx[lv] = blk_idx;
   }

   map_cell_blk_start_idx_.insert(std::make_pair(cell, v_blk_idx));
   }
   }
   */

  template<typename Dtype>
  void SPMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    vector<Blob<Dtype>*>* top)
  {
    Dtype* const top_data = ((*top)[0]->mutable_cpu_data());
    const Dtype* const bottom_data = ((bottom)[0]->cpu_data());
    const Dtype* const patch_pos = ((bottom)[1]->cpu_data());

    for (int imgid = 0; imgid < num_img_; ++imgid)
    {
      Dtype* const out = top_data + spm_dim_;
      const Dtype* const in_code = bottom_data + num_patch_ * feat_dim_;
      const Dtype* const in_pos = patch_pos + num_patch_ * 2;
      spm_model_.MaxPooling(reinterpret_cast<const float*>(in_code), feat_dim_,
                            num_patch_, reinterpret_cast<const float*>(in_pos),
                            reinterpret_cast<float*>(out));
    }

    /*
     const int top_dim = (*top)[0]->count() / num_img_;

     //LOG(INFO) << "total img: " << num_img_;

     for (int imgid = 0; imgid < num_img_; ++imgid)
     {
     //LOG(INFO) << "imgid: " << imgid;

     Dtype* const cur_top_data = top_data + imgid * top_dim;
     const Dtype* const cur_bot_data = (bottom_data)
     + imgid * num_patch_ * llc_dim_;
     const Dtype* const cur_pos = patch_pos + imgid * num_patch_ * pos_dim_;

     vector<vector<int> > cell_pos_map(num_cell_x_y_,
     vector<int>(num_cell_x_y_, 0));
     for (int tmpi = 0; tmpi < num_patch_; ++tmpi)
     {
     const int x = static_cast<int>(*(cur_pos + tmpi * pos_dim_));
     const int y = static_cast<int>(*(cur_pos + tmpi * pos_dim_ + 1));
     cell_pos_map[y][x] = tmpi;
     }

     // first we should compute the finest bin
     // check if we need to compute the code
     if (finest_num_blk_ == num_cell_x_y_)
     {
     memcpy(cur_top_data, cur_bot_data,
     sizeof(Dtype) * num_patch_ * llc_dim_);
     }
     else
     {
     int code_idx(0);

     for (int ybin = 0; ybin < finest_num_blk_; ++ybin)
     for (int xbin = 0; xbin < finest_num_blk_; ++xbin)
     {
     Dtype* const out_code = cur_top_data + llc_dim_ * code_idx;
     ++code_idx;

     for (int ycell = 0; ycell < finest_num_cell_; ++ycell)
     for (int xcell = 0; xcell < finest_num_cell_; ++xcell)
     {
     const int ycell_full = ybin * finest_num_cell_ + ycell;
     const int xcell_full = xbin * finest_num_cell_ + xcell;

     const int patchidx = cell_pos_map[ycell_full][xcell_full];

     //cout << "patchidx: " << patchidx << endl;

     const Dtype* const in_code = cur_bot_data + llc_dim_ * patchidx;

     // for the first time, initialize
     if (ycell == 0 && xcell == 0)
     memcpy(out_code, in_code, sizeof(Dtype) * llc_dim_);
     else
     {
     // max pooling
     for (int dd = 0; dd < llc_dim_; ++dd)
     out_code[dd] =
     out_code[dd] > in_code[dd] ? out_code[dd] : in_code[dd];
     }
     }      // xcell and ycell of input
     }      // xbin and ybin of output
     }

     // then, according to the finest code, we can simply compute the remaining
     for (int lv = num_spm_level_ - 2; lv >= 0; --lv)
     {
     const int num_blk = level_num_blk_[lv];
     const int level_start_idx = level_start_idx_[lv];
     const int prev_level_start_idx = level_start_idx_[lv + 1];

     //cout << "level " << lv << ", start index: " << level_start_idx << endl;

     int idx(0);

     for (int ybin = 0; ybin < num_blk; ++ybin)
     for (int xbin = 0; xbin < num_blk; ++xbin)
     {
     const int out_idx = level_start_idx + idx;
     Dtype* out_code = cur_top_data + llc_dim_ * out_idx;

     //cout << "out idx: " << out_idx << " ";

     for (int y_subbin = 0; y_subbin < 2; ++y_subbin)
     for (int x_subbin = 0; x_subbin < 2; ++x_subbin)
     {
     // figure out which subbin we should use for current pooling
     const int in_subbin_idx = prev_level_start_idx + 4 * idx
     + y_subbin * 2 + x_subbin;
     //cout << "in index: " << in_subbin_idx << endl;
     const Dtype* const in_code = cur_top_data
     + in_subbin_idx * llc_dim_;

     if (y_subbin == 0 && x_subbin == 0)
     memcpy(out_code, in_code, llc_dim_);
     else
     {
     for (int dd = 0; dd < llc_dim_; ++dd)
     out_code[dd] =
     out_code[dd] > in_code[dd] ? out_code[dd] : in_code[dd];
     }
     }

     ++idx;
     }
     }      // spm level
     }      // image
     */
  }

  template<typename Dtype>
  Dtype SPMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const bool propagate_down,
                                      vector<Blob<Dtype>*>* bottom)
  {
    if (!propagate_down)
      return Dtype(0.);

    const Dtype* const top_diff = top[0]->cpu_diff();
    const Dtype* const top_data = top[0]->cpu_data();
    const Dtype* const bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* const patch_pos = ((*bottom)[1]->cpu_data());

    Dtype* const bottom_diff = (*bottom)[0]->mutable_cpu_diff();

    memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));

    for (int imgid = 0; imgid < num_img_; ++imgid)
    {
      const Dtype* const cur_top_data = top_data + imgid * spm_dim_;
      const Dtype* const cur_top_diff = top_diff + imgid * spm_dim_;
      const Dtype* const cur_bot_data = (bottom_data)
          + imgid * num_patch_ * feat_dim_;
      const Dtype* const cur_pos = patch_pos + imgid * num_patch_ * 2;

      Dtype* const cur_bot_diff = bottom_diff + imgid * num_patch_ * feat_dim_;

      for (int i = 0; i < num_patch_; ++i)
      {
        map<uint32_t, vector<uint32_t> >::const_iterator it =
            map_cell_blk_.find(i);
        CHECK(it != map_cell_blk_.end());

        const vector<uint32_t>& blks = it->second;

        Dtype* const out = cur_bot_diff + i * feat_dim_;
        const Dtype* const in_bot_data = cur_bot_data + i * feat_dim_;

        for (size_t j = 0; j < blks.size(); ++j)
        {
          const uint32_t blk_id = blks[j];
          const Dtype* const in_top_data = cur_top_data + blk_id * feat_dim_;
          const Dtype* const in_top_diff = cur_top_diff + blk_id * feat_dim_;

          for (int dd = 0; dd < feat_dim_; ++dd)
          {
            out[dd] += in_top_diff[dd]
                * (std::abs(in_bot_data[dd] - in_top_data[dd]) < FLT_THRD);
          }
        }
      }

      /*
       vector<vector<int> > cell_pos_map(num_cell_x_y_,
       vector<int>(num_cell_x_y_, 0));
       for (int tmpi = 0; tmpi < num_patch_; ++tmpi)
       {
       const int x = static_cast<int>(*(cur_pos + tmpi * pos_dim_));
       const int y = static_cast<int>(*(cur_pos + tmpi * pos_dim_ + 1));
       cell_pos_map[y][x] = tmpi;
       }

       for (int ycell = 0; ycell < num_cell_x_y_; ++ycell)
       for (int xcell = 0; xcell < num_cell_x_y_; ++xcell)
       {
       const int patchidx = cell_pos_map[ycell][xcell];
       Dtype* const in_diff = cur_bot_diff + llc_dim_ * patchidx;
       const Dtype* const in_data = cur_bot_data + llc_dim_ * patchidx;

       map<pair<int, int>, vector<int> >::const_iterator it =
       map_cell_blk_start_idx_.find(std::make_pair(ycell, xcell));
       CHECK(it != map_cell_blk_start_idx_.end());

       const vector<int>& v_blks_idx = it->second;

       // collect all the gradients from all levels, starting with finest level
       for (size_t blki = 0; blki < v_blks_idx.size(); ++blki)
       {
       const Dtype* const out_data = cur_top_data
       + v_blks_idx[blki] * llc_dim_;
       const Dtype* const out_diff = cur_top_diff
       + v_blks_idx[blki] * llc_dim_;

       for (int dd = 0; dd < llc_dim_; ++dd)
       {
       in_diff[dd] += out_diff[dd]
       * (abs(in_data[dd] - out_data[dd]) < FLT_THRD);
       }
       }
       }
       */
    }      // image

    return Dtype(0.);
  }

  INSTANTIATE_CLASS(SPMLayer);
}
