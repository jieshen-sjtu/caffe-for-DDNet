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
  void* LLCDataFTLayerPrefetch(void* layer_pointer)
  {
    CHECK(layer_pointer);
    LLCDataFTLayer<Dtype>* layer =
        reinterpret_cast<LLCDataFTLayer<Dtype>*>(layer_pointer);
    CHECK(layer);
    Datum datum;
    CHECK(layer->prefetch_patch_data_);
    CHECK(layer->prefetch_llc_patch_pos_);
    CHECK(layer->prefetch_label_);
    Dtype* top_patch_data = layer->prefetch_patch_data_->mutable_cpu_data();
    Dtype* top_patch_pos = layer->prefetch_llc_patch_pos_->mutable_cpu_data();
    Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();

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
    const int img_height = layer->datum_height_;
    const int img_width = layer->datum_width_;
    const int img_size = layer->datum_size_;

    const int patch_height = layer->patch_height_;
    const int patch_width = layer->patch_width_;
    const int patch_size = layer->patch_size_;
    const int num_patch = layer->num_patch_;

    // patch setting
    const int llc_dim = layer->llc_dim_;

    const int num_img = batchsize;

    const Dtype* mean = layer->data_mean_.cpu_data();

    for (int imgid = 0; imgid < num_img; ++imgid)
    {
      // get a blob
      CHECK(layer->iter_);
      CHECK(layer->iter_->Valid());
      datum.ParseFromString(layer->iter_->value().ToString());
      const int label = datum.label();
      const string& imgname = datum.img_name();

      //LOG(INFO)<< imgname << " " << label;

      // fetch label
      top_label[imgid] = label;

      const int data_start = imgid * num_patch * patch_size;
      const int patch_pos_start = imgid * num_patch * 2;

      for (int pid = 0; pid < num_patch; ++pid)
      {
        Dtype* cur_data = top_patch_data + data_start + pid * patch_size;
        for (int j = 0; j < patch_size; ++j)
        {
          *(cur_data + j) = static_cast<Dtype>(datum.patch_data(
              pid * patch_size + j));
        }

        // fetch patch position
        Dtype* cur_patch_pos = top_patch_pos + patch_pos_start + pid * 2;
        for (int j = 0; j < 2; ++j)
        {
          *(cur_patch_pos + j) = static_cast<Dtype>(datum.llc_pos(pid * 2 + j));
        }
      }

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
  LLCDataFTLayer<Dtype>::~LLCDataFTLayer<Dtype>()
  {
    // Finally, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  }

  template<typename Dtype>
  void LLCDataFTLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                    vector<Blob<Dtype>*>* top)
  {
    CHECK_EQ(bottom.size(), 0)<< "LLCDataFT Layer takes no input blobs.";
    CHECK_EQ(top->size(), 3) << "LLCDataFT Layer takes four blobs as output: "
    << "patch data, pos, label";

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
    patch_height_ = datum.patch_height();
    patch_width_ = datum.patch_width();
    patch_size_ = datum_channels_ * patch_height_ * patch_width_;
    num_patch_ = datum.num_patch();

    CHECK_EQ(datum_size_, patch_size_ * num_patch_);

    const int batchsz = this->layer_param_.batchsize();

    (*top)[0]->Reshape(batchsz * num_patch_, datum_channels_, patch_height_,
        patch_width_);
    prefetch_patch_data_.reset(new Blob<Dtype>(batchsz * num_patch_, datum_channels_,
            patch_height_, patch_width_));
    LOG(INFO) << "output patch data size: " << (*top)[0]->num() << ","
    << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
    << (*top)[0]->width();

    // patch position
    (*top)[1]->Reshape(batchsz * num_patch_, 2, 1, 1);
    prefetch_llc_patch_pos_.reset(new Blob<Dtype>(batchsz * num_patch_, 2, 1, 1));
    LOG(INFO) << "output patch position size: " << (*top)[1]->num() << ","
    << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
    << (*top)[1]->width();

    // label
    (*top)[2]->Reshape(batchsz, 1, 1, 1);
    prefetch_label_.reset(new Blob<Dtype>(batchsz, 1, 1, 1));
    LOG(INFO) << "output label size: " << (*top)[2]->num() << ","
    << (*top)[2]->channels() << "," << (*top)[2]->height() << ","
    << (*top)[2]->width();

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
    prefetch_patch_data_->mutable_cpu_data();
    prefetch_llc_patch_pos_->mutable_cpu_data();
    prefetch_label_->mutable_cpu_data();
    data_mean_.cpu_data();
    DLOG(INFO) << "Initializing prefetch";
    CHECK(!pthread_create(&thread_, NULL, LLCDataFTLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
    DLOG(INFO) << "Prefetch initialized.";
  }

  template<typename Dtype>
  void LLCDataFTLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          vector<Blob<Dtype>*>* top)
  {
    // First, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
    // Copy the data
    memcpy((*top)[0]->mutable_cpu_data(), prefetch_patch_data_->cpu_data(),
           sizeof(Dtype) * prefetch_patch_data_->count());
    memcpy((*top)[1]->mutable_cpu_data(), prefetch_llc_patch_pos_->cpu_data(),
           sizeof(Dtype) * prefetch_llc_patch_pos_->count());
    memcpy((*top)[2]->mutable_cpu_data(), prefetch_label_->cpu_data(),
           sizeof(Dtype) * prefetch_label_->count());

    // Start a new prefetch thread
    CHECK(!pthread_create(&thread_, NULL, LLCDataFTLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  }

  template<typename Dtype>
  void LLCDataFTLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          vector<Blob<Dtype>*>* top)
  {
    // First, join the thread
    CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
    // Copy the data
    CUDA_CHECK(
        cudaMemcpy((*top)[0]->mutable_gpu_data(),
                   prefetch_patch_data_->cpu_data(),
                   sizeof(Dtype) * prefetch_patch_data_->count(),
                   cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy((*top)[1]->mutable_gpu_data(),
                   prefetch_llc_patch_pos_->cpu_data(),
                   sizeof(Dtype) * prefetch_llc_patch_pos_->count(),
                   cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy((*top)[2]->mutable_gpu_data(), prefetch_label_->cpu_data(),
                   sizeof(Dtype) * prefetch_label_->count(),
                   cudaMemcpyHostToDevice));

    // Start a new prefetch thread
    CHECK(!pthread_create(&thread_, NULL, LLCDataFTLayerPrefetch<Dtype>,
            reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  }

// The backward operations are dummy - they do not carry any computation.
  template<typename Dtype>
  Dtype LLCDataFTLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const bool propagate_down,
                                            vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.);
  }

  template<typename Dtype>
  Dtype LLCDataFTLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const bool propagate_down,
                                            vector<Blob<Dtype>*>* bottom)
  {
    return Dtype(0.);
  }

  INSTANTIATE_CLASS(LLCDataFTLayer);

}  // namespace caffe
