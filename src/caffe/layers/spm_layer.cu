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

    const int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
    int channels(0);

    for(int i=0; i<num_spm_level_; ++i)
    channels += std::pow(4, i);

    channels *= dim;

    // merge the image patches
    (*top)[0]->Reshape(bottom[0]->num() / num_patch_, channels, 1, 1);
  }

  template<typename Dtype>
  void SPMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    vector<Blob<Dtype>*>* top)
  {
    Dtype* const top_data = ((*top)[0]->mutable_cpu_data());
    const Dtype* const bottom_data = ((*bottom)[0]->cpu_data());
    const Dtype* const patch_pos = ((*bottom)[1]->cpu_data());

    const int num_img = (*top)[0]->num();
    const int top_dim = (*top)[0]->count() / num_img;
    const int bot_dim = (*bottom)[0]->count() / (*bottom)[0]->num();
    const int pos_dim = 2;

    // we have confirm the square shape of patch
    const int num_cell_x_y = std::sqrt(num_patch_);

    for (int imgid = 0; imgid < num_img; ++imgid)
    {
      Dtype* const cur_top_data = top_data + imgid * top_dim;
      const Dtype* const cur_bot_data = (bottom_data)
          + imgid * num_patch_ * bot_dim;
      const Dtype* const cur_pos = patch_pos + imgid * num_patch_ * pos_dim;

      vector<vector<int> > cell_pos_map(num_cell_x_y,
                                        vector<int>(num_cell_x_y, 0));
      for (int tmpi = 0; tmpi < num_patch_; ++tmpi)
      {
        const int x = static_cast<int>(*(cur_pos + tmpi * pos_dim));
        const int y = static_cast<int>(*(cur_pos + tmpi * pos_dim + 1));
        cell_pos_map[y][x] = tmpi;
      }

      // fine to coarse pooling
      int prev_cell_for_spm_x = num_cell_x_y / std::pow(2, num_spm_level_);
      int prev_cell_for_spm_y = prev_cell_for_spm_x;

      int prev_out_start = 0;

      for (int lv = num_spm_level_ - 1; lv >= 0; --lv)
      {
        Dtype* const cur_level_out = cur_top_data + prev_out_start;

        // compute the number of cells for spm
        const int num_cell_for_spm_x = 2 * prev_cell_for_spm_x;
        const int num_cell_for_spm_y = 2 * prev_cell_for_spm_y;

        // each block outputs a feature vector
        const int num_blk_x = num_cell_x_y / num_cell_for_spm_x;
        const int num_blk_y = num_cell_x_y / num_cell_for_spm_y;

        // now perform max pooling for each block
        for (int by = 0; by < num_blk_y; ++by)
          for (int bx = 0; bx < num_blk_x; ++bx)
          {
            Dtype* const cur_blk_out = cur_level_out + by * num_blk_x + bx;

            for (int cy = 0; cy < num_cell_for_spm_y; ++cy)
              for (int cx = 0; cx < num_cell_for_spm_x; ++cx)
              {
                const int cell_id_y = by * num_cell_for_spm_y + cy;
                const int cell_id_x = bx * num_cell_for_spm_x + cx;
                const int cell_id_full = cell_pos_map[cell_id_y][cell_id_x];

                const Dtype* const cur_cell = cur_bot_data
                    + cell_id_full * bot_dim;
                if (cy == 0 && cx == 0)
                  memcpy(cur_blk_out, cur_cell, sizeof(Dtype) * bot_dim);
                else
                {
                  // do elemetwise max
                  for (int d = 0; d < bot_dim; ++d)
                    cur_blk_out[d] =
                        cur_blk_out[d] > cur_cell[d] ?
                            cur_blk_out[d] : cur_cell[d];
                }
              }
          }

        // for the next pooling
        prev_cell_for_spm_x = num_cell_for_spm_x;
        prev_cell_for_spm_y = num_cell_for_spm_y;

        prev_out_start += num_blk_x * num_blk_y * bot_dim;
      }
    }
  }

  INSTANTIATE_CLASS(SPMLayer);
}
