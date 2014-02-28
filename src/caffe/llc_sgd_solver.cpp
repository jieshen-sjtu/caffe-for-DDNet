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
    LOG(INFO)<< "LLC SGD";
    //Solver<Dtype>::Solve(resume_file);

    Caffe::set_mode(Caffe::Brew(this->param_.solver_mode()));
    if (this->param_.solver_mode() && this->param_.has_device_id())
    {
      Caffe::SetDevice(this->param_.device_id());
    }
    Caffe::set_phase(Caffe::TRAIN);
    LOG(INFO) << "Solving " << this->net_->name();
    this->PreSolve();

    this->iter_ = 0;
    if (resume_file)
    {
      LOG(INFO) << "Restoring previous solver status from " << resume_file;
      this->Restore(resume_file);
    }

    // For a network that is trained by the solver, no bottom or top vecs
    // should be given, and we will just provide dummy vecs.
    vector<Blob<Dtype>*> bottom_vec;
    while (this->iter_++ < this->param_.max_iter())
    {
      Dtype loss = this->net_->ForwardBackward(bottom_vec);
      this->ComputeUpdateValue();
      this->net_->Update();

      if (this->param_.display() && this->iter_ % this->param_.display() == 0)
      {
        LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss;
      }
      if (this->param_.test_interval()
          && this->iter_ % this->param_.test_interval() == 0)
      {
        // We need to set phase to test before running.
        Caffe::set_phase(Caffe::TEST);
        this->Test();
        Caffe::set_phase(Caffe::TRAIN);
      }
      // Check if we need to do snapshot
      if (this->param_.snapshot() && this->iter_ % this->param_.snapshot() == 0)
      {
        this->Snapshot();
      }
    }
    // After the optimization is done, always do a snapshot.
    this->iter_--;
    this->Snapshot();
    LOG(INFO) << "Optimization Done.";
  }

  INSTANTIATE_CLASS(LLCSGDSolver);
}
