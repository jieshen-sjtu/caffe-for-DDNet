// Copyright Yangqing Jia 2013

#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <boost/shared_ptr.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/common.hpp>
#include <caffe/blob.hpp>


#include <vector>
#include <string>
using boost::shared_ptr;
using std::vector;
using std::string;

namespace caffe
{
  class SolverParameter;

  template<typename Dtype>
  class Net;

  template<typename Dtype>
  class Solver
  {
  public:
    explicit Solver(const SolverParameter& param);
    // The main entry of the solver function. In default, iter will be zero. Pass
    // in a non-zero iter number to resume training for a pre-trained net.
    virtual void Solve(const char* resume_file = NULL);
    inline void Solve(const string resume_file)
    {
      Solve(resume_file.c_str());
    }
    virtual ~Solver()
    {
    }
    inline Net<Dtype>* net()
    {
      return net_.get();
    }

  protected:
    // PreSolve is run before any solving iteration starts, allowing one to
    // put up some scaffold.
    virtual void PreSolve()
    {
    }
    // Get the update value for the current iteration.
    virtual void ComputeUpdateValue() = 0;
    // The Solver::Snapshot function implements the basic snapshotting utility
    // that stores the learned net. You should implement the SnapshotSolverState()
    // function that produces a SolverState protocol buffer that needs to be
    // written to disk together with the learned net.
    void Snapshot();
    // The test routine
    void Test();
    virtual void SnapshotSolverState(SolverState* state) = 0;
    // The Restore function implements how one should restore the solver to a
    // previously snapshotted state. You should implement the RestoreSolverState()
    // function that restores the state from a SolverState protocol buffer.
    void Restore(const char* resume_file);
    virtual void RestoreSolverState(const SolverState& state) = 0;

    SolverParameter param_;
    int iter_;
    shared_ptr<Net<Dtype> > net_;
    shared_ptr<Net<Dtype> > test_net_;

  DISABLE_COPY_AND_ASSIGN (Solver);
  };

  template<typename Dtype>
  class SGDSolver: public Solver<Dtype>
  {
  public:
    explicit SGDSolver(const SolverParameter& param)
        : Solver<Dtype>(param)
    {
    }

  protected:
    virtual void PreSolve();
    Dtype GetLearningRate();
    virtual void ComputeUpdateValue();
    virtual void SnapshotSolverState(SolverState * state);
    virtual void RestoreSolverState(const SolverState& state);
    // history maintains the historical momentum data.
    vector<shared_ptr<Blob<Dtype> > > history_;

  DISABLE_COPY_AND_ASSIGN (SGDSolver);
  };

  /*
  template<typename Dtype>
  class LLCSGDSolver: public caffe::SGDSolver<Dtype>
  {
  public:
    explicit LLCSGDSolver(const caffe::SolverParameter& param)
        : caffe::SGDSolver<Dtype>(param)
    {
    }

  public:
    virtual void Solve(const char* resume_file = NULL);

  DISABLE_COPY_AND_ASSIGN (LLCSGDSolver);
  };
  */

}  // namspace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
