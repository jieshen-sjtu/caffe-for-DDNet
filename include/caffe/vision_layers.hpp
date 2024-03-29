// Copyright 2013 Yangqing Jia

#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <opencv2/opencv.hpp>
#include <leveldb/db.h>
#include <pthread.h>

#include <cstdio>
#include <vector>
#include <string>
#include <map>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "EYE.hpp"

using std::vector;
using std::string;
using std::map;
using std::pair;

namespace caffe
{

// The neuron layer is a specific type of layers that just works on single
// celements.
  template<typename Dtype>
  class NeuronLayer : public Layer<Dtype>
  {
     public:
      explicit NeuronLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);
  };

  template<typename Dtype>
  class ReLULayer : public NeuronLayer<Dtype>
  {
     public:
      explicit ReLULayer(const LayerParameter& param)
          : NeuronLayer<Dtype>(param)
      {
      }

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
  };

  template<typename Dtype>
  class TanHLayer : public NeuronLayer<Dtype>
  {
     public:
      explicit TanHLayer(const LayerParameter& param)
          : NeuronLayer<Dtype>(param)
      {
      }

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
  };

  template<typename Dtype>
  class SigmoidLayer : public NeuronLayer<Dtype>
  {
     public:
      explicit SigmoidLayer(const LayerParameter& param)
          : NeuronLayer<Dtype>(param)
      {
      }

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
  };

  template<typename Dtype>
  class BNLLLayer : public NeuronLayer<Dtype>
  {
     public:
      explicit BNLLLayer(const LayerParameter& param)
          : NeuronLayer<Dtype>(param)
      {
      }

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
  };

  template<typename Dtype>
  class DropoutLayer : public NeuronLayer<Dtype>
  {
     public:
      explicit DropoutLayer(const LayerParameter& param)
          : NeuronLayer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      shared_ptr<SyncedMemory> rand_vec_;
      float threshold_;
      float scale_;
      unsigned int uint_thres_;
  };

  template<typename Dtype>
  class SplitLayer : public Layer<Dtype>
  {
     public:
      explicit SplitLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      int count_;
  };

  template<typename Dtype>
  class FlattenLayer : public Layer<Dtype>
  {
     public:
      explicit FlattenLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      int count_;
  };

  template<typename Dtype>
  class InnerProductLayer : public Layer<Dtype>
  {
     public:
      explicit InnerProductLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      int M_;
      int K_;
      int N_;
      bool biasterm_;
      shared_ptr<SyncedMemory> bias_multiplier_;
  };

  template<typename Dtype>
  class PaddingLayer : public Layer<Dtype>
  {
     public:
      explicit PaddingLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      unsigned int PAD_;
      int NUM_;
      int CHANNEL_;
      int HEIGHT_IN_;
      int WIDTH_IN_;
      int HEIGHT_OUT_;
      int WIDTH_OUT_;
  };

  template<typename Dtype>
  class LRNLayer : public Layer<Dtype>
  {
     public:
      explicit LRNLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      // scale_ stores the intermediate summing results
      Blob<Dtype> scale_;
      int size_;
      int pre_pad_;
      Dtype alpha_;
      Dtype beta_;
      int num_;
      int channels_;
      int height_;
      int width_;
  };

  template<typename Dtype>
  class Im2colLayer : public Layer<Dtype>
  {
     public:
      explicit Im2colLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      int KSIZE_;
      int STRIDE_;
      int CHANNELS_;
      int HEIGHT_;
      int WIDTH_;
  };

  template<typename Dtype>
  class PoolingLayer : public Layer<Dtype>
  {
     public:
      explicit PoolingLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      int KSIZE_;
      int STRIDE_;
      int CHANNELS_;
      int HEIGHT_;
      int WIDTH_;
      int POOLED_HEIGHT_;
      int POOLED_WIDTH_;
      Blob<float> rand_idx_;
  };

  template<typename Dtype>
  class ConvolutionLayer : public Layer<Dtype>
  {
     public:
      explicit ConvolutionLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      Blob<Dtype> col_bob_;

      int KSIZE_;
      int STRIDE_;
      int NUM_;
      int CHANNELS_;
      int HEIGHT_;
      int WIDTH_;
      int NUM_OUTPUT_;
      int GROUP_;
      Blob<Dtype> col_buffer_;
      shared_ptr<SyncedMemory> bias_multiplier_;
      bool biasterm_;
      int M_;
      int K_;
      int N_;
  };

// This function is used to create a pthread that prefetches the data.
  template<typename Dtype>
  void* DataLayerPrefetch(void* layer_pointer);

  template<typename Dtype>
  class DataLayer : public Layer<Dtype>
  {
      // The function used to perform prefetching.
      friend void* DataLayerPrefetch<Dtype>(void* layer_pointer);

     public:
      explicit DataLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual ~DataLayer();
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      shared_ptr<leveldb::DB> db_;
      shared_ptr<leveldb::Iterator> iter_;
      int datum_channels_;
      int datum_height_;
      int datum_width_;
      int datum_size_;
      pthread_t thread_;
      shared_ptr<Blob<Dtype> > prefetch_data_;
      shared_ptr<Blob<Dtype> > prefetch_label_;
      Blob<Dtype> data_mean_;
  };

  template<typename Dtype>
  class SoftmaxLayer : public Layer<Dtype>
  {
     public:
      explicit SoftmaxLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      // sum_multiplier is just used to carry out sum using blas
      Blob<Dtype> sum_multiplier_;
      // scale is an intermediate blob to hold temporary results.
      Blob<Dtype> scale_;
  };

  template<typename Dtype>
  class MultinomialLogisticLossLayer : public Layer<Dtype>
  {
     public:
      explicit MultinomialLogisticLossLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      // The loss layer will do nothing during forward - all computation are
      // carried out in the backward pass.
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  };

  template<typename Dtype>
  class InfogainLossLayer : public Layer<Dtype>
  {
     public:
      explicit InfogainLossLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            infogain_()
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      // The loss layer will do nothing during forward - all computation are
      // carried out in the backward pass.
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

      Blob<Dtype> infogain_;
  };

// SoftmaxWithLossLayer is a layer that implements softmax and then computes
// the loss - it is preferred over softmax + multinomiallogisticloss in the
// sense that during training, this will produce more numerically stable
// gradients. During testing this layer could be replaced by a softmax layer
// to generate probability outputs.
  template<typename Dtype>
  class SoftmaxWithLossLayer : public Layer<Dtype>
  {
     public:
      explicit SoftmaxWithLossLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            softmax_layer_(new SoftmaxLayer<Dtype>(param))
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
      // prob stores the output probability of the layer.
      Blob<Dtype> prob_;
      // Vector holders to call the underlying softmax layer forward and backward.
      vector<Blob<Dtype>*> softmax_bottom_vec_;
      vector<Blob<Dtype>*> softmax_top_vec_;
  };

  template<typename Dtype>
  class EuclideanLossLayer : public Layer<Dtype>
  {
     public:
      explicit EuclideanLossLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            difference_()
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      // The loss layer will do nothing during forward - all computation are
      // carried out in the backward pass.
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
      Blob<Dtype> difference_;
  };

  template<typename Dtype>
  class AccuracyLayer : public Layer<Dtype>
  {
     public:
      explicit AccuracyLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      // The accuracy layer should not be used to compute backward operations.
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom)
      {
        NOT_IMPLEMENTED;
        return Dtype(0.);
      }
  };

  template<typename Dtype>
  void* LLCDataUnsupLayerPrefetch(void* layer_pointer);

  template<typename Dtype>
  class LLCDataUnsupLayer : public Layer<Dtype>
  {
      // The function used to perform prefetching.
      friend void* LLCDataUnsupLayerPrefetch<Dtype>(void* layer_pointer);

     public:
      explicit LLCDataUnsupLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            datum_channels_(0),
            datum_height_(0),
            datum_width_(0),
            datum_size_(0),
            thread_(0),
            llc_dim_(0),
            patch_height_(0),
            patch_width_(0),
            patch_size_(0),
            num_patch_(0),
            prefetch_patch_data_(),
            prefetch_llc_codes_()
      {
      }
      virtual ~LLCDataUnsupLayer();
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      shared_ptr<leveldb::DB> db_;
      shared_ptr<leveldb::Iterator> iter_;
      int datum_channels_;
      int datum_height_;
      int datum_width_;
      int datum_size_;

      int llc_dim_;
      int patch_height_;
      int patch_width_;
      int patch_size_;
      int num_patch_;

      pthread_t thread_;
      shared_ptr<Blob<Dtype> > prefetch_patch_data_;
      shared_ptr<Blob<Dtype> > prefetch_llc_codes_;

      Blob<Dtype> data_mean_;
  };

  template<typename Dtype>
  void* LLCDataFTLayerPrefetch(void* layer_pointer);

  template<typename Dtype>
  class LLCDataFTLayer : public Layer<Dtype>
  {
      // The function used to perform prefetching.
      friend void* LLCDataFTLayerPrefetch<Dtype>(void* layer_pointer);

     public:
      explicit LLCDataFTLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            datum_channels_(0),
            datum_height_(0),
            datum_width_(0),
            datum_size_(0),
            thread_(0),
            llc_dim_(0),
            patch_height_(0),
            patch_width_(0),
            patch_size_(0),
            num_patch_(0),
            prefetch_patch_data_(),
            prefetch_llc_patch_pos_(),
            prefetch_label_()
      {
      }
      virtual ~LLCDataFTLayer();
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      shared_ptr<leveldb::DB> db_;
      shared_ptr<leveldb::Iterator> iter_;
      int datum_channels_;
      int datum_height_;
      int datum_width_;
      int datum_size_;

      int llc_dim_;
      int patch_height_;
      int patch_width_;
      int patch_size_;
      int num_patch_;

      pthread_t thread_;
      shared_ptr<Blob<Dtype> > prefetch_patch_data_;
      shared_ptr<Blob<Dtype> > prefetch_llc_patch_pos_;
      shared_ptr<Blob<Dtype> > prefetch_label_;

      Blob<Dtype> data_mean_;
  };

  template<typename Dtype>
  void* LLCDataSVMLayerPrefetch(void* layer_pointer);

  template<typename Dtype>
  class LLCDataSVMLayer : public Layer<Dtype>
  {
      // The function used to perform prefetching.
      friend void* LLCDataSVMLayerPrefetch<Dtype>(void* layer_pointer);

     public:
      explicit LLCDataSVMLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            datum_channels_(0),
            datum_height_(0),
            datum_width_(0),
            datum_size_(0),
            thread_(0),
            llc_dim_(0),
            patch_height_(0),
            patch_width_(0),
            patch_size_(0),
            num_patch_(0),
            prefetch_llc_code_(),
            prefetch_llc_patch_pos_(),
            prefetch_label_()
      {
      }
      virtual ~LLCDataSVMLayer();
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      shared_ptr<leveldb::DB> db_;
      shared_ptr<leveldb::Iterator> iter_;
      int datum_channels_;
      int datum_height_;
      int datum_width_;
      int datum_size_;

      int llc_dim_;
      int patch_height_;
      int patch_width_;
      int patch_size_;
      int num_patch_;

      pthread_t thread_;
      shared_ptr<Blob<Dtype> > prefetch_llc_code_;
      shared_ptr<Blob<Dtype> > prefetch_llc_patch_pos_;
      shared_ptr<Blob<Dtype> > prefetch_label_;

      Blob<Dtype> data_mean_;
  };

  template<typename Dtype>
  void* DataUnsupLayerPrefetch(void* layer_pointer);

  template<typename Dtype>
  class DataUnsupLayer : public Layer<Dtype>
  {
      // The function used to perform prefetching.
      friend void* DataUnsupLayerPrefetch<Dtype>(void* layer_pointer);

     public:
      explicit DataUnsupLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual ~DataUnsupLayer();
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      shared_ptr<leveldb::DB> db_;
      shared_ptr<leveldb::Iterator> iter_;
      int datum_channels_;
      int datum_height_;
      int datum_width_;
      int datum_size_;
      pthread_t thread_;
      shared_ptr<Blob<Dtype> > prefetch_data_;
      Blob<Dtype> data_mean_;
  };

  template<typename Dtype>
  class PatchLayer : public Layer<Dtype>
  {
     public:
      explicit PatchLayer(const LayerParameter& param);

      virtual ~PatchLayer()
      {
        if (tmp_)
          free(tmp_);
      }

      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      /*
       virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       vector<Blob<Dtype>*>* top);
       virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
       const bool propagate_down,
       vector<Blob<Dtype>*>* bottom);
       */

      virtual void forward_patch(const cv::Mat& img, float* const top);

      int img_height_;
      int img_width_;
      int img_channels_;
      int img_size_;

      int patch_height_;
      int patch_width_;
      int patch_channels_;
      int patch_size_;

      uint32_t dsift_step_;
      vector<uint32_t> dsift_sizes_;
      uint32_t dsift_std_size_;
      vector<int> dsift_off_;
      vector<uint32_t> dsift_num_patches_;
      int all_num_patches_;
      vector<int> dsift_start_x_;
      vector<int> dsift_start_y_;
      vector<int> dsift_end_x_;
      vector<int> dsift_end_y_;

      cv::Mat tmpimg_;
      float* tmp_;
  };

  template<typename Dtype>
  class LLCCodeLayer : public Layer<Dtype>
  {
     public:
      explicit LLCCodeLayer(const LayerParameter& param);

      virtual ~LLCCodeLayer()
      {
        ;
      }

      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      /*
       virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       vector<Blob<Dtype>*>* top);
       virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
       const bool propagate_down,
       vector<Blob<Dtype>*>* bottom);
       */

      int img_height_;
      int img_width_;
      int img_channels_;
      int img_size_;

      int patch_height_;
      int patch_width_;
      int patch_channels_;
      int patch_size_;

      uint32_t dsift_step_;
      vector<uint32_t> dsift_sizes_;
      uint32_t dsift_std_size_;
      vector<int> dsift_off_;
      vector<uint32_t> dsift_num_patches_;
      int all_num_patches_;
      vector<int> dsift_start_x_;
      vector<int> dsift_start_y_;
      vector<int> dsift_end_x_;
      vector<int> dsift_end_y_;

      EYE::LLC llc_model_;
      EYE::DSift dsift_model_;
      int llc_dim_;

      cv::Mat tmpimg_;
      cv::Mat grayimg_;
  };

  /*
   template<typename Dtype>
   void* LLCDataLayerPrefetch(void* layer_pointer);

   template<typename Dtype>
   class LLCDataLayer : public Layer<Dtype>
   {
   // The function used to perform prefetching.
   friend void* LLCDataLayerPrefetch<Dtype>(void* layer_pointer);

   public:
   explicit LLCDataLayer(const LayerParameter& param)
   : Layer<Dtype>(param),
   datum_channels_(0),
   datum_height_(0),
   datum_width_(0),
   datum_size_(0),
   thread_(0),
   llc_dim_(0),
   patch_height_(0),
   patch_width_(0),
   patch_size_(0),
   num_patch_(0)
   {
   }
   virtual ~LLCDataLayer();
   virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
   vector<Blob<Dtype>*>* top);

   protected:
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
   vector<Blob<Dtype>*>* top);
   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   vector<Blob<Dtype>*>* top);
   virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
   const bool propagate_down,
   vector<Blob<Dtype>*>* bottom);
   virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
   const bool propagate_down,
   vector<Blob<Dtype>*>* bottom);

   shared_ptr<leveldb::DB> db_;
   shared_ptr<leveldb::Iterator> iter_;
   int datum_channels_;
   int datum_height_;
   int datum_width_;
   int datum_size_;

   int llc_dim_;
   int patch_height_;
   int patch_width_;
   int patch_size_;
   int num_patch_;

   pthread_t thread_;
   shared_ptr<Blob<Dtype> > prefetch_data_;
   shared_ptr<Blob<Dtype> > prefetch_llc_codes_;
   shared_ptr<Blob<Dtype> > prefetch_llc_patch_pos_;
   shared_ptr<Blob<Dtype> > prefetch_label_;

   Blob<Dtype> data_mean_;
   };
   */

  template<typename Dtype>
  class ShiftLayer : public Layer<Dtype>
  {
     public:
      explicit ShiftLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
        /*
         scale_ = NULL;
         all_one_ = NULL;
         sum_top_dif_data_ = NULL;
         */
      }
      virtual ~ShiftLayer()
      {
        /*
         if (scale_)
         free(scale_);
         if (all_one_)
         free(all_one_);
         if (sum_top_dif_data_)
         free(sum_top_dif_data_);
         */
        ;
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);

      // sum_multiplier is just used to carry out sum using blas
      Blob<Dtype> sum_multiplier_;
      // scale is an intermediate blob to hold temporary results.
      Blob<Dtype> scale_;
  };

  template<typename Dtype>
  class L1LossLayer : public Layer<Dtype>
  {
     public:
      explicit L1LossLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            difference_()
      {
        output_ = fopen("code.txt", "w");
      }

      virtual ~L1LossLayer()
      {
        fclose(output_);
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      // The loss layer will do nothing during forward - all computation are
      // carried out in the backward pass.
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top)
      {
        return;
      }
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
      Blob<Dtype> difference_;

      FILE* output_;
  };

  template<typename Dtype>
  class LLCAccuracyLayer : public Layer<Dtype>
  {
     public:
      explicit LLCAccuracyLayer(const LayerParameter& param)
          : Layer<Dtype>(param)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);
      // The accuracy layer should not be used to compute backward operations.
      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom)
      {
        NOT_IMPLEMENTED;
        return Dtype(0.);
      }

      Blob<Dtype> difference_;
  };

  template<typename Dtype>
  class SPMLayer : public Layer<Dtype>
  {
     public:
      explicit SPMLayer(const LayerParameter& param)
          : Layer<Dtype>(param),
            num_spm_level_(0),
            feat_dim_(0),
            img_width_(0),
            img_height_(0),
            num_img_(0),
            num_patch_(0),
            spm_dim_(0)
      {
      }
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      /*
       virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       vector<Blob<Dtype>*>* top);
       virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
       const bool propagate_down,
       vector<Blob<Dtype>*>* bottom);
       */

     private:
      //void build_spm();

      int num_img_;
      int num_patch_;

      int num_spm_level_;
      int img_width_;
      int img_height_;

      int feat_dim_;
      int spm_dim_;

      EYE::SPM spm_model_;

      map<uint32_t, vector<uint32_t> > map_cell_blk_;
      map<uint32_t, vector<uint32_t> > map_blk_cell_;
      /*
       int num_patch_;
       bool hor_pool_;

       int num_cell_x_y_;
       int finest_num_blk_;
       int finest_num_cell_;

       vector<int> level_start_idx_;
       vector<int> level_num_blk_;
       map<pair<int, int>, vector<int> > map_cell_blk_start_idx_;

       int num_img_;
       int llc_dim_;
       int pos_dim_;*/

  };

  template<typename Dtype>
  class SVMOutLayer : public Layer<Dtype>
  {
     public:
      explicit SVMOutLayer(const LayerParameter& param);
      virtual ~SVMOutLayer();
      virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                         vector<Blob<Dtype>*>* top);

     protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               vector<Blob<Dtype>*>* top);

      virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const bool propagate_down,
                                 vector<Blob<Dtype>*>* bottom);
      /*
       virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       vector<Blob<Dtype>*>* top);
       virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
       const bool propagate_down,
       vector<Blob<Dtype>*>* bottom);
       */

      int dim_;
      string out_file_;
      FILE* output_;
  };

/*
 template<typename Dtype>
 void* ZJQDataLayerPrefetch(void* layer_pointer);

 template<typename Dtype>
 class ZJQDataLayer : public Layer<Dtype>
 {
 // The function used to perform prefetching.
 friend void* ZJQDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
 explicit ZJQDataLayer(const LayerParameter& param)
 : Layer<Dtype>(param)
 {
 }
 virtual ~ZJQDataLayer();
 virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top);

 protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top);
 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top);
 virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
 const bool propagate_down,
 vector<Blob<Dtype>*>* bottom);
 virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
 const bool propagate_down,
 vector<Blob<Dtype>*>* bottom);

 shared_ptr<leveldb::DB> db_;
 shared_ptr<leveldb::Iterator> iter_;
 int datum_channels_;
 int datum_height_;
 int datum_width_;
 int datum_size_;

 int num_multi_label_;
 int num_context_;

 pthread_t thread_;
 shared_ptr<Blob<Dtype> > prefetch_data_;
 shared_ptr<Blob<Dtype> > prefetch_multi_label_;
 shared_ptr<Blob<Dtype> > prefetch_context_;
 Blob<Dtype> data_mean_;
 };

 template<typename Dtype>
 class ZJQContextLayer : public Layer<Dtype>
 {
 public:
 explicit ZJQContextLayer(const LayerParameter& param)
 : Layer<Dtype>(param),
 num_feat_map_(0),
 height_(0),
 width_(0),
 context_dim_(0)
 {
 }
 virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top);

 protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top);
 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top);

 virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
 const bool propagate_down,
 vector<Blob<Dtype>*>* bottom);
 virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
 const bool propagate_down,
 vector<Blob<Dtype>*>* bottom);

 int num_feat_map_;
 int height_;
 int width_;

 int context_dim_;
 Blob<Dtype> w_multi_context_;
 Blob<Dtype> all_ones_;
 Blob<Dtype> all_ones_sample_;
 Blob<Dtype> tmp_;

 Blob<Dtype> bias_;
 shared_ptr<SyncedMemory> bias_multiplier_;
 };
 */
}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_

