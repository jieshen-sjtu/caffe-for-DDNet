/*
 * llc_data_layer.cpp
 *
 *  Created on: Feb 20, 2014
 *      Author: jieshen
 */

#include <mkl.h>
#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::endl;

namespace caffe
{

  template<typename Dtype>
  void* LLCDataLayerPrefetch(void* layer_pointer)
  {
    CHECK(layer_pointer);
    LLCDataLayer<Dtype>* layer = reinterpret_cast<LLCDataLayer<Dtype>*>(layer_pointer);
    CHECK(layer);
    Datum datum;
    CHECK(layer->prefetch_data_);
    Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
    //Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
    Dtype* top_llc_codes = layer->prefetch_llc_codes_->mutable_cpu_data();

    const Dtype scale = layer->layer_param_.scale();
    const int batchsize = layer->layer_param_.batchsize();
    const int cropsize = layer->layer_param_.cropsize();
    const bool mirror = layer->layer_param_.mirror();

    if (mirror && cropsize == 0)
    {
      LOG(FATAL)<< "Current implementation requires mirror and cropsize to be "
      << "set at the same time.";
    }
    // datum scales
    const int channels = layer->datum_channels_;
    const int height = layer->datum_height_;
    const int width = layer->datum_width_;
    const int size = layer->datum_size_;

    // patch setting
    const int llc_dim = layer->llc_dim_;

    const Dtype* mean = layer->data_mean_.cpu_data();

    for (int itemid = 0; itemid < batchsize; ++itemid)
    {
      // get a blob
      CHECK(layer->iter_);
      CHECK(layer->iter_->Valid());
      datum.ParseFromString(layer->iter_->value().ToString());
      const string& data = datum.data();
      if (cropsize)
      {
        CHECK(data.size()) << "Image cropping only support uint8 data";
        int h_off, w_off;
        // We only do random crop when we do training.
        if (Caffe::phase() == Caffe::TRAIN)
        {
          h_off = rand() % (height - cropsize);
          w_off = rand() % (width - cropsize);
        }
        else
        {
          h_off = (height - cropsize) / 2;
          w_off = (width - cropsize) / 2;
        }
        if (mirror && rand() % 2)
        {
          // Copy mirrored version
          for (int c = 0; c < channels; ++c)
          {
            for (int h = 0; h < cropsize; ++h)
            {
              for (int w = 0; w < cropsize; ++w)
              {
                top_data[((itemid * channels + c) * cropsize + h) * cropsize
                    + cropsize - 1 - w] = (static_cast<Dtype>((uint8_t) data[(c
                    * height + h + h_off) * width + w + w_off])
                    - mean[(c * height + h + h_off) * width + w + w_off])
                    * scale;
              }
            }
          }
        }
        else
        {
          // Normal copy
          for (int c = 0; c < channels; ++c)
          {
            for (int h = 0; h < cropsize; ++h)
            {
              for (int w = 0; w < cropsize; ++w)
              {
                top_data[((itemid * channels + c) * cropsize + h) * cropsize + w] = (static_cast<Dtype>((uint8_t) data[(c
                    * height + h + h_off) * width + w + w_off])
                    - mean[(c * height + h + h_off) * width + w + w_off])
                    * scale;
              }
            }
          }
        }
      }
      else
      {
        // we will prefer to use data() first, and then try float_data()
        if (data.size())
        {
          for (int j = 0; j < size; ++j)
          {
            top_data[itemid * size + j] = (static_cast<Dtype>((uint8_t) data[j])
                - mean[j]) * scale;
          }
        }
        else
        {
          for (int j = 0; j < size; ++j)
          {
            top_data[itemid * size + j] = (datum.float_data(j) - mean[j])
                * scale;
          }
        }
      }

      // fetch label
      //top_label[itemid] = datum.label();

      // fetch patch data
      for (int j = 0; j < llc_dim; ++j)
        top_llc_codes[itemid * llc_dim + j] = static_cast<Dtype>(datum.llc_codes(
            j));

      // normalize llc code to one
      float* cur_llc_code = (float*) top_llc_codes + itemid * llc_dim;
      const float norm1 = cblas_sasum(llc_dim, cur_llc_code, 1);
      cblas_sscal(llc_dim, 1.0 / norm1, cur_llc_code, 1);

      // go to the next iter
      layer->iter_->Next();
      if (!layer->iter_->Valid())
      {
        // We have reached the end. Restart from the first.
        DLOG(INFO)<< "Restarting data prefetching from start.";
        layer->iter_->SeekToFirst();
      }
    }

    return (void*) NULL;
  }

  template<typename Dtype>
  LLCDataLayer<Dtype>::~LLCDataLayer<Dtype>()
  {
    // Finally, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  }

  template<typename Dtype>
  void LLCDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                  vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 0)<< "LLCData Layer takes no input blobs.";
    CHECK_EQ(top->size(), 2) << "LLCData Layer takes two blobs as output.";

    // Initialize the leveldb
    leveldb::DB* db_temp;
    leveldb::Options options;
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
    << this->layer_param_.source() << std::endl << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();

    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.rand_skip())
    {
      unsigned int skip = rand() % this->layer_param_.rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      while (skip-- > 0)
      {
        iter_->Next();
        if (!iter_->Valid())
        {
          iter_->SeekToFirst();
        }
      }
    }

    // Read a data point, and use it to initialize the top blob.
    Datum datum;
    datum.ParseFromString(iter_->value().ToString());

    // datum size
    datum_channels_ = datum.channels();
    datum_height_ = datum.height();
    datum_width_ = datum.width();
    datum_size_ = datum.channels() * datum.height() * datum.width();

    llc_dim_ = datum.llc_dim();

    // image data
    int cropsize = this->layer_param_.cropsize();
    CHECK_GT(datum_height_, cropsize);
    CHECK_GT(datum_width_, cropsize);
    if (cropsize > 0)
    {
      (*top)[0]->Reshape(
          this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize);
      prefetch_data_.reset(new Blob<Dtype>(
              this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize));
    }
    else
    {
      (*top)[0]->Reshape(
          this->layer_param_.batchsize(), datum.channels(), datum.height(),
          datum.width());
      prefetch_data_.reset(new Blob<Dtype>(
              this->layer_param_.batchsize(), datum.channels(), datum.height(),
              datum.width()));
    }
    LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
    << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
    << (*top)[0]->width();

    // label
    /*
     (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
     prefetch_label_.reset(
     new Blob<Dtype>(this->layer_param_.batchsize(), 1, 1, 1));
     LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
     << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
     << (*top)[1]->width();
     */

    // LLC code
    (*top)[1]->Reshape(this->layer_param_.batchsize(), datum.llc_dim(), 1, 1);
    prefetch_llc_codes_.reset(new Blob<Dtype>(this->layer_param_.batchsize(), datum.llc_dim(), 1, 1));
    LOG(INFO) << "output LLC code size: " << (*top)[1]->num() << ","
    << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
    << (*top)[1]->width();

    // check if we want to have mean
    if (this->layer_param_.has_meanfile())
    {
      BlobProto blob_proto;
      LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
      ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
      data_mean_.FromProto(blob_proto);
      CHECK_EQ(data_mean_.num(), 1);
      CHECK_EQ(data_mean_.channels(), datum_channels_);
      CHECK_EQ(data_mean_.height(), datum_height_);
      CHECK_EQ(data_mean_.width(), datum_width_);
    }
    else
    {
      // Simply initialize an all-empty mean.
      data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
    }
    // Now, start the prefetch thread. Before calling prefetch, we make two
    // cpu_data calls so that the prefetch thread does not accidentally make
    // simultaneous cudaMalloc calls when the main thread is running. In some
    // GPUs this seems to cause failures if we do not so.
    prefetch_data_->mutable_cpu_data();
    //prefetch_label_->mutable_cpu_data();
    prefetch_llc_codes_->mutable_cpu_data();
    data_mean_.cpu_data();
    DLOG(INFO) << "Initializing prefetch";
    CHECK(!pthread_create(&thread_, NULL, LLCDataLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
    DLOG(INFO) << "Prefetch initialized.";
  }

  template<typename Dtype>
  void LLCDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        vector<Blob<Dtype>*>* top)
  {
    // First, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
    // Copy the data
    memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
           sizeof(Dtype) * prefetch_data_->count());
    /*memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
     sizeof(Dtype) * prefetch_label_->count());*/
    memcpy((*top)[1]->mutable_cpu_data(), prefetch_llc_codes_->cpu_data(),
           sizeof(Dtype) * prefetch_llc_codes_->count());

    // Start a new prefetch thread
    CHECK(!pthread_create(&thread_, NULL, LLCDataLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  }

  template<typename Dtype>
  void LLCDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        vector<Blob<Dtype>*>* top)
  {
    // First, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
    // Copy the data
    CUDA_CHECK(
        cudaMemcpy((*top)[0]->mutable_gpu_data(), prefetch_data_->cpu_data(),
                   sizeof(Dtype) * prefetch_data_->count(),
                   cudaMemcpyHostToDevice));
    /*CUDA_CHECK(
     cudaMemcpy((*top)[1]->mutable_gpu_data(), prefetch_label_->cpu_data(),
     sizeof(Dtype) * prefetch_label_->count(),
     cudaMemcpyHostToDevice));*/
    CUDA_CHECK(
        cudaMemcpy((*top)[1]->mutable_gpu_data(),
                   prefetch_llc_codes_->cpu_data(),
                   sizeof(Dtype) * prefetch_llc_codes_->count(),
                   cudaMemcpyHostToDevice));
    // Start a new prefetch thread
    CHECK(!pthread_create(&thread_, NULL, LLCDataLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  }

// The backward operations are dummy - they do not carry any computation.
  template<typename Dtype>
  Dtype LLCDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const bool propagate_down,
                                          vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.);
  }

  template<typename Dtype>
  Dtype LLCDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const bool propagate_down,
                                          vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.);
  }

  INSTANTIATE_CLASS(LLCDataLayer);

}  // namespace caffe
