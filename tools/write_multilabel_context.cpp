/*
 * write_multilabel_context.cpp
 *
 *  Created on: Mar 19, 2014
 *      Author: jieshen
 */

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::pair;
using std::string;
using std::vector;

#define NUMLABEL 5760
#define NUMCONTEXT 100

int main(int argc, char** argv)
{
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 5)
  {
    printf("Convert a set of images to the leveldb format used\n"
           "as input for Caffe.\n"
           "Usage:\n"
           "    convert_imageset ROOTFOLDER/ LABELFILE CONTEXT DB_NAME"
           " RANDOM_SHUFFLE_DATA[0 or 1]\n");
    return 0;
  }

  std::vector<std::pair<string, vector<float> > > lines;
  {
    std::ifstream infile(argv[2]);

    vector<float> label(NUMLABEL, 0);
    while (infile.good())
    {
      string filename;
      infile >> filename;
      if (filename.empty())
        break;

      for (int i = 0; i < NUMLABEL; ++i)
        infile >> label[i];

      lines.push_back(std::make_pair(filename, label));
    }
    infile.close();
    if (argc == 6 && argv[5][0] == '1')
    {
      // randomly shuffle data
      LOG(INFO)<< "Shuffling data";
      std::random_shuffle(lines.begin(), lines.end());
    }
    LOG(INFO)<< "A total of " << lines.size() << " images.";
  }

  std::map<string, vector<float> > map_name_contxt;
  {
    vector<float> contxt(NUMCONTEXT, 0);
    std::ifstream input(argv[3], 0);
    while (input.good())
    {
      string filename;
      input >> filename;
      if (filename.empty())
        break;

      for (int i = 0; i < NUMCONTEXT; ++i)
        input >> contxt[i];

      map_name_contxt.insert(std::make_pair(filename, contxt));
    }
    input.close();
  }

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO)<< "Opening leveldb " << argv[4];
  leveldb::Status status = leveldb::DB::Open(options, argv[4], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[4];

  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  int data_size;
  bool data_size_initialized = false;
  for (int line_id = 0; line_id < lines.size(); ++line_id)
  {
    const std::pair<string, vector<float> >& name_label = lines[line_id];
    const string& name = name_label.first;
    const vector<float>& cur_labels = name_label.second;
    const vector<float>& cur_conxts = map_name_contxt.find(name)->second;

    // set image name
    datum.set_img_name(name);

    // set image data
    {
      const string img_full_name = root_folder + name;
      cv::Mat cv_img = cv::imread(img_full_name, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data)
      {
        LOG(ERROR)<< "Could not open or find file " << img_full_name;
        return false;
      }

      datum.set_channels(3);
      datum.set_height(cv_img.rows);
      datum.set_width(cv_img.cols);
      datum.clear_data();
      datum.clear_float_data();
      string* datum_string = datum.mutable_data();
      for (int c = 0; c < 3; ++c)
      {
        for (int h = 0; h < cv_img.rows; ++h)
        {
          for (int w = 0; w < cv_img.cols; ++w)
          {
            datum_string->push_back(
                static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
          }
        }
      }
    }

    // set multi-label
    {
      datum.set_num_multi_label(NUMLABEL);
      datum.clear_multi_label();
      datum.mutable_multi_label->Reserve(cur_labels.size());
      for (int i = 0; i < cur_labels.size(); ++i)
        datum.add_multi_label(cur_labels[i]);
    }

    // set context
    {
      datum.set_num_context(NUMCONTEXT);
      datum.clear_context();
      datum.mutable_context->Reserve(cur_conxts.size());
      for (int i = 0; i < cur_conxts.size(); ++i)
        datum.add_context(cur_conxts[i]);
    }

    string value;
    // get the value
    datum.SerializeToString(&value);
    batch->Put(name, value);
    if (++count % 1000 == 0)
    {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR)<< "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % 1000 != 0)
  {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR)<< "Processed " << count << " files.";
  }

  delete batch;
  delete db;
  return 0;
}

