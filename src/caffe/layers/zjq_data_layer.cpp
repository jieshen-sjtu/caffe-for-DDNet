/*
 * zjq_data_layer.cpp
 *
 *  Created on: Mar 19, 2014
 *      Author: jieshen
 */

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe
{

  template<typename Dtype>
  void* ZJQDataLayerPrefetch(void* layer_pointer)
  {
    CHECK(layer_pointer);
    ZJQDataLayer<Dtype>* layer =
        reinterpret_cast<ZJQDataLayer<Dtype>*>(layer_pointer);
    CHECK(layer);
    Datum datum;
    CHECK(layer->prefetch_data_);
    CHECK(layer->prefetch_multi_label_);
    CHECK(layer->prefetch_context_);
    Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
    Dtype* top_multi_label = layer->prefetch_multi_label_->mutable_cpu_data();
    Dtype* top_context = layer->prefetch_context_->mutable_cpu_data();
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
    const Dtype* mean = layer->data_mean_.cpu_data();
    for (int itemid = 0; itemid < batchsize; ++itemid)
    {
      // get a blob
      CHECK(layer->iter_);
      CHECK(layer->iter_->Valid());
      datum.ParseFromString(layer->iter_->value().ToString());

      // get data
      {
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
                      + cropsize - 1 - w] =
                      (static_cast<Dtype>((uint8_t) data[(c * height + h + h_off)
                          * width + w + w_off])
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
                  top_data[((itemid * channels + c) * cropsize + h) * cropsize
                      + w] = (static_cast<Dtype>((uint8_t) data[(c * height + h
                      + h_off) * width + w + w_off])
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
              top_data[itemid * size + j] =
                  (static_cast<Dtype>((uint8_t) data[j]) - mean[j]) * scale;
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
      }

      // get multi-label
      {
        const int num_multi_label = layer->num_multi_label_;
        for (int j = 0; j < num_multi_label; ++j)
          top_multi_label[itemid * num_multi_label + j] = datum.multi_label(j);
      }

      // get context
      {
        const int num_contxt = layer->num_context_;
        for (int j = 0; j < num_contxt; ++j)
          top_context[itemid * num_contxt + j] = datum.context(j);
      }

      // go to the next iter
      layer->iter_->Next();
      if (!layer->iter_->Valid())
      {
        // We have reached the end. Restart from the first.
        DLOG(INFO)<< "Restarting data prefetching from start.";
        layer->iter_->SeekToFirst();
      }
    }  // itemid

    return (void*) NULL;
  }

  template<typename Dtype>
  ZJQDataLayer<Dtype>::~ZJQDataLayer<Dtype>()
  {
    // Finally, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  }

  template<typename Dtype>
  void ZJQDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                  vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 0)<<"Data Layer takes no input blobs.";
    CHECK_EQ(top->size(), 3) << "Data Layer takes three blobs as output: "
    <<"data, multi-label, context";

    // Initialize the leveldb
    leveldb::DB* db_temp;
    leveldb::Options options;
    options.create_if_missing = false;
    options.max_open_files = 100;
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

    // image
    {
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

      // datum size
      datum_channels_ = datum.channels();
      datum_height_ = datum.height();
      datum_width_ = datum.width();
      datum_size_ = datum.channels() * datum.height() * datum.width();

      LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

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
    }

    // multi-label
    {
      num_multi_label_ = datum.num_multi_label();
      (*top)[1]->Reshape(this->layer_param_.batchsize(), num_multi_label_, 1, 1);
      prefetch_multi_label_.reset(
          new Blob<Dtype>(this->layer_param_.batchsize(), num_multi_label_, 1, 1));
      LOG(INFO) << "output multi-label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
    }

    // context
    {
      num_context_ = datum.num_context();
      (*top)[2]->Reshape(this->layer_param_.batchsize(), num_context_, 1, 1);
      prefetch_context_.reset(
          new Blob<Dtype>(this->layer_param_.batchsize(), num_context_, 1, 1));
      LOG(INFO) << "output multi-label size: " << (*top)[2]->num() << ","
      << (*top)[2]->channels() << "," << (*top)[2]->height() << ","
      << (*top)[2]->width();
    }

    // Now, start the prefetch thread. Before calling prefetch, we make two
    // cpu_data calls so that the prefetch thread does not accidentally make
    // simultaneous cudaMalloc calls when the main thread is running. In some
    // GPUs this seems to cause failures if we do not so.
    prefetch_data_->mutable_cpu_data();
    prefetch_multi_label_->mutable_cpu_data();
    prefetch_context_->mutable_cpu_data();
    data_mean_.cpu_data();
    DLOG(INFO) << "Initializing prefetch";
    CHECK(!pthread_create(&thread_, NULL, ZJQDataLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
    DLOG(INFO) << "Prefetch initialized.";
  }

  template<typename Dtype>
  void ZJQDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        vector<Blob<Dtype>*>* top)
  {
    // First, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
    // Copy the data
    memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
           sizeof(Dtype) * prefetch_data_->count());
    memcpy((*top)[1]->mutable_cpu_data(), prefetch_multi_label_->cpu_data(),
           sizeof(Dtype) * prefetch_multi_label_->count());
    memcpy((*top)[2]->mutable_cpu_data(), prefetch_context_->cpu_data(),
           sizeof(Dtype) * prefetch_context_->count());
    // Start a new prefetch thread
    CHECK(!pthread_create(&thread_, NULL, ZJQDataLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  }

  template<typename Dtype>
  void ZJQDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        vector<Blob<Dtype>*>* top)
  {
    // First, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
    // Copy the data
    CUDA_CHECK(
        cudaMemcpy((*top)[0]->mutable_gpu_data(), prefetch_data_->cpu_data(),
                   sizeof(Dtype) * prefetch_data_->count(),
                   cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy((*top)[1]->mutable_gpu_data(),
                   prefetch_multi_label_->cpu_data(),
                   sizeof(Dtype) * prefetch_multi_label_->count(),
                   cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy((*top)[2]->mutable_gpu_data(), prefetch_context_->cpu_data(),
                   sizeof(Dtype) * prefetch_context_->count(),
                   cudaMemcpyHostToDevice));
    // Start a new prefetch thread
    CHECK(!pthread_create(&thread_, NULL, ZJQDataLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  }

// The backward operations are dummy - they do not carry any computation.
  template<typename Dtype>
  Dtype ZJQDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const bool propagate_down,
                                          vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.);
  }

  template<typename Dtype>
  Dtype ZJQDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const bool propagate_down,
                                          vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.);
  }

  INSTANTIATE_CLASS(ZJQDataLayer);

}
// namespace caffe

