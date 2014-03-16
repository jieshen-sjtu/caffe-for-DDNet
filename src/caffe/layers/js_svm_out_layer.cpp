/*
 * svm_out_layer.cpp
 *
 *  Created on: Mar 11, 2014
 *      Author: jieshen
 */

#include <caffe/vision_layers.hpp>

namespace caffe
{
  const int FLT_THRD = 1e-5;

  template<typename Dtype>
  SVMOutLayer<Dtype>::SVMOutLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        dim_(0),
        out_file_(param.svm_out_file()),
        output_(NULL)
  {
    ;
  }

  template<typename Dtype>
  SVMOutLayer<Dtype>::~SVMOutLayer()
  {
    if (output_)
    {
      fclose(output_);
      output_ = NULL;
    }
  }

  template<typename Dtype>
  void SVMOutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 2)<<"SVMOut Layer takes two blobs: "
    <<"code and label";
    CHECK_EQ(top->size(), 0) << "SVMOut Layer takes no output blob";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());

    dim_ = bottom[0]->count() / bottom[0]->num();

    output_ = fopen(out_file_.c_str(), "w");
    CHECK(output_ != NULL) <<"Fail to open " << out_file_;
  }

  template<typename Dtype>
  void SVMOutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       vector<Blob<Dtype>*>* top)
  {
    const int num = bottom[0]->num();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* labels = bottom[1]->cpu_data();

    for (int i = 0; i < num; ++i)
    {
      const Dtype* data = bottom_data + dim_;
      const int label = static_cast<int>(labels[i]);

      fprintf(output_, "%d", label);

      for (int d = 0; d < dim_; ++d)
      {
        const float val = static_cast<float>(data[d]);
        if (val < FLT_THRD)
          continue;
        fprintf(output_, " %d:%f", d + 1, val);
      }

      fprintf(output_, "\n");
    }

  }

  template<typename Dtype>
  Dtype SVMOutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const bool propagate_down,
                                         vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.);
  }

  INSTANTIATE_CLASS(SVMOutLayer);
}
