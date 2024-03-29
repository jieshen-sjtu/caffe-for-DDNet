// Copyright 2013 Yangqing Jia

#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::map;
using std::vector;
using std::string;

namespace caffe
{

  template<typename Dtype>
  class Net
  {
     public:
      Net(const NetParameter& param);
      Net(const string& param_file);
      virtual ~Net()
      {
      }

      // Initialize a network with the network parameter.
      void Init(const NetParameter& param);

      // Run forward with the input blobs already fed separately. You can get the
      // input blobs using input_blobs().
      virtual const vector<Blob<Dtype>*>& ForwardPrefilled();
      // Run forward using a set of bottom blobs, and return the result.
      virtual const vector<Blob<Dtype>*>& Forward(
          const vector<Blob<Dtype>*> & bottom);
      // Run forward using a serialized BlobProtoVector and return the result
      // as a serialized BlobProtoVector
      virtual string Forward(const string& input_blob_protos);

      // The network backward should take no input and output, since it solely
      // computes the gradient w.r.t the parameters, and the data has already
      // been provided during the forward pass.
      virtual Dtype Backward();

      virtual Dtype ForwardBackward(const vector<Blob<Dtype>*> & bottom)
      {
        Forward(bottom);
        return Backward();
      }

      // Updates the network weights based on the diff values computed.
      virtual void Update();

      // For an already initialized net, CopyTrainedLayersFrom() copies the already
      // trained layers from another net parameter instance.
      void CopyTrainedLayersFrom(const NetParameter& param);
      void CopyTrainedLayersFrom(const string trained_filename);
      // Writes the net to a proto.
      void ToProto(NetParameter* param, bool write_diff = false);

      // returns the network name.
      inline const string& name() const
      {
        return name_;
      }
      // returns the layer names
      inline const vector<string>& layer_names() const
      {
        return layer_names_;
      }
      // returns the blob names
      inline const vector<string>& blob_names() const
      {
        return blob_names_;
      }
      // returns the blobs
      inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const
      {
        return blobs_;
      }
      // returns the layers
      inline const vector<shared_ptr<Layer<Dtype> > >& layers() const
      {
        return layers_;
      }
      // returns the bottom and top vecs for each layer - usually you won't need
      // this unless you do per-layer checks such as gradients.
      inline vector<vector<Blob<Dtype>*> >& bottom_vecs()
      {
        return bottom_vecs_;
      }
      inline vector<vector<Blob<Dtype>*> >& top_vecs()
      {
        return top_vecs_;
      }
      inline vector<vector<int> >& bottom_id_vecs()
      {
        return bottom_id_vecs_;
      }
      inline vector<vector<int> >& top_id_vecs()
      {
        return top_id_vecs_;
      }
      // returns the parameters
      inline vector<shared_ptr<Blob<Dtype> > >& params()
      {
        return params_;
      }
      // returns the parameter learning rate multipliers
      inline vector<float>& params_lr()
      {
        return params_lr_;
      }
      inline vector<float>& params_weight_decay()
      {
        return params_weight_decay_;
      }
      // Input and output blob numbers
      inline int num_inputs()
      {
        return net_input_blobs_.size();
      }
      inline int num_outputs()
      {
        return net_output_blobs_.size();
      }
      inline vector<Blob<Dtype>*>& input_blobs()
      {
        return net_input_blobs_;
      }
      inline vector<Blob<Dtype>*>& output_blobs()
      {
        return net_output_blobs_;
      }

      inline const string& blob_name_for_deep_feat() const
      {
        return blob_name_for_deep_feat_;
      }
      inline int blob_id_for_deep_feat() const
      {
        return blob_id_for_deep_feat_;
      }
      inline const string& blob_name_for_deep_code() const
      {
        return blob_name_for_deep_code_;
      }
      inline int blob_id_for_deep_code() const
      {
        return blob_id_for_deep_code_;
      }

     protected:
      // Function to get misc parameters, e.g. the learning rate multiplier and
      // weight decay.
      void GetLearningRateAndWeightDecay();

      // Individual layers in the net
      vector<shared_ptr<Layer<Dtype> > > layers_;
      vector<string> layer_names_;
      vector<bool> layer_need_backward_;
      // blobs stores the blobs that store intermediate results between the
      // layers.
      vector<shared_ptr<Blob<Dtype> > > blobs_;
      vector<string> blob_names_;
      vector<bool> blob_need_backward_;
      // bottom_vecs stores the vectors containing the input for each layer.
      // They don't actually host the blobs (blobs_ does), so we simply store
      // pointers.
      vector<vector<Blob<Dtype>*> > bottom_vecs_;
      vector<vector<int> > bottom_id_vecs_;
      // top_vecs stores the vectors containing the output for each layer
      vector<vector<Blob<Dtype>*> > top_vecs_;
      vector<vector<int> > top_id_vecs_;
      // blob indices for the input and the output of the net
      vector<int> net_input_blob_indices_;
      vector<Blob<Dtype>*> net_input_blobs_;
      vector<Blob<Dtype>*> net_output_blobs_;
      string name_;
      // The parameters in the network.
      vector<shared_ptr<Blob<Dtype> > > params_;
      // the learning rate multipliers
      vector<float> params_lr_;
      // the weight decay multipliers
      vector<float> params_weight_decay_;

      string blob_name_for_deep_feat_;
      int blob_id_for_deep_feat_;
      string blob_name_for_deep_code_;
      int blob_id_for_deep_code_;

    DISABLE_COPY_AND_ASSIGN(Net);
  };

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
