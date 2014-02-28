/*
 * llc_sgd_solver.cpp
 *
 *  Created on: Feb 28, 2014
 *      Author: jieshen
 */

#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe
{
  template<typename Dtype>
  void LLCSGDSolver<Dtype>::Solve(const char* resume_file)
  {
    Caffe::set_mode(Caffe::Brew(param_.solver_mode()));
    if (param_.solver_mode() && param_.has_device_id())
    {
      Caffe::SetDevice(param_.device_id());
    }
    Caffe::set_phase(Caffe::TRAIN);
    LOG(INFO) << "LLCSGD Solving " << net_->name();
    PreSolve();

    iter_ = 0;
    if (resume_file)
    {
      LOG(INFO) << "Restoring previous solver status from " << resume_file;
      Restore(resume_file);
    }

    // For a network that is trained by the solver, no bottom or top vecs
    // should be given, and we will just provide dummy vecs.
    vector<Blob<Dtype>*> bottom_vec;
    while (iter_++ < param_.max_iter())
    {
      Dtype loss = net_->ForwardBackward(bottom_vec);
      ComputeUpdateValue();
      net_->Update();

      if (param_.display() && iter_ % param_.display() == 0)
      {
        LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
      }
      if (param_.test_interval() && iter_ % param_.test_interval() == 0)
      {
        // We need to set phase to test before running.
        Caffe::set_phase(Caffe::TEST);
        Test();
        Caffe::set_phase(Caffe::TRAIN);
      }
      // Check if we need to do snapshot
      if (param_.snapshot() && iter_ % param_.snapshot() == 0)
      {
        Snapshot();
      }
    }
    // After the optimization is done, always do a snapshot.
    iter_--;
    Snapshot();
    LOG(INFO) << "Optimization Done.";
  }

  INSTANTIATE_CLASS(LLCSGDSolver);
}

