//
//	Copyright (c) 2015, Emory University
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification, are
//	permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this list of
//	conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice, this list
// 	of conditions and the following disclaimer in the documentation and/or other materials
//	provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//	OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//	SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//	BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
//	WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//	DAMAGE.
//
//
#ifndef _TRAIN_NODE_H
#define _TRAIN_NODE_H

#include "node.h"
#include "edge.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include <boost/shared_ptr.hpp>


using namespace std;


class TrainNode: public Node
{
public:

			TrainNode(string id, int mode, int batch_size, int device_id, string model_path, 
					  float base_lr, float momentum, float gamma, int iter, string outFilename);

		void *run();
		void init_model();
		void compute_update_value();
		void snapshot();
		int train_step(int first_idx);
		void cross_validate(vector<cv::Mat> batch, vector<int> batch_labels);


protected:
		int _batch_size;
		string _model_path;
		string _outFilename;

		vector<cv::Mat> _data_buffer; 
		vector<int>		_labels_buffer;
		boost::shared_ptr<caffe::Net<float>> _net;
		float _base_lr;
		float _momentum;
		float _gamma;
		vector<boost::shared_ptr<caffe::Blob<float>>> _history;
		vector<boost::shared_ptr<caffe::Blob<float>>> _temp;
		boost::shared_ptr<caffe::Blob<float> > _out_layer;
		boost::shared_ptr<caffe::MemoryDataLayer<float>> _data_layer;
		int _iter;
		int _device_id;
};


#endif
