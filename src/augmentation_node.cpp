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
#define PI 3.1415926535897932
#include <cmath>
#include <random>
#include <chrono>
#include "augmentation_node.h"
#include "edge.h"
#include "utils.h"






AugmentationNode::AugmentationNode(string id, int transferSize, int mode, int aug_factor) : 
Node(id, mode), 
_counter(0), 
_data_buffer(), 
_labels_buffer(), 
_aug_factor(aug_factor),
_transferSize(transferSize)
{
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	_labels_buffer.clear();
}





void *AugmentationNode::run()
{

	double 	loop, start = utils::get_time();

	while(true) {

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);

		if( _data_buffer.size() >= _transferSize ) {

			loop = utils::get_time();
			// Augment data
			augment_images(_data_buffer, _labels_buffer);

			// Clean buffers
			_data_buffer.clear();
			_labels_buffer.clear();

		} else {

			// Check if all input nodes have already finished
			bool is_all_done = true;

			for(vector<int>::size_type i=0; i < _in_edges.size(); i++) {

				if( !_in_edges.at(i)->is_in_node_done() ) {
					is_all_done = false;
					break;
				}
			}			

			// All input nodes have finished
			if( is_all_done ) {

				// TODO - Add check to see if data still in buffer
	
				cout << "******************" << endl 
					 << "AugmentationNode" << endl 
					 << "Total_time_first: " << to_string(utils::get_time() - runtime_total_first) << endl 
					 << "# of elements: " << to_string(_counter) << endl 
					 << "******************" << endl;

				cout << "AugmentationNode runtime: " << utils::get_time() - start << endl;

				// Notify it has finished
				for(vector<int>::size_type i=0; i < _out_edges.size(); i++) {
					_out_edges.at(i)->set_in_node_done();
				}
				break;
			}
		}
	}
	return NULL;
}





void AugmentationNode::augment_images(vector<cv::Mat> imgs, vector<int> labels)
{

	vector< vector<cv::Mat> > 	out_imgs;
	vector< vector<int> > 		out_labels;

	out_imgs.resize(_aug_factor + 1);
	out_labels.resize(_aug_factor + 1);

	for(int k=0; k < imgs.size(); k++) {
		_counter++;

		for(int i=0; i < _aug_factor; i++) {
			_counter++;

			// Define variables
			cv::Mat src(imgs[k]);
			cv::Mat rot_M(2, 3, CV_8U);

			cv::Mat warped_img(src.size(), src.type());
			int label = labels[k];

	  		// construct a trivial random generator engine from a time-based seed:
	  		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	  		default_random_engine generator (seed);

			// Rotation 
			uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
			float theta = 180 * uniform_dist(generator);
			if( theta < -90.0 ) {
				theta = -180.0;
			} else if( theta < 0.0 ) {
				theta = -90.0;
			} else if( theta < 90.0 ) {
				theta = 0.0;
			} else {
				theta = 90.0;
			}

			cv::Point2f warped_img_center(warped_img.cols/2.0F, warped_img.rows/2.0F);
			rot_M = getRotationMatrix2D(warped_img_center, theta, 1.0);
			cv::warpAffine( src, warped_img, rot_M, warped_img.size());

			// Rescaling
			uniform_real_distribution<double> sec_uniform_dist(1.0, 1.5);
			int Sx = sec_uniform_dist(generator);
			int Sy = sec_uniform_dist(generator);

			cv::Mat scaling_M(2, 3, CV_32F);
			scaling_M.at<float>(0,0) = Sx;
			scaling_M.at<float>(0,1) = 0;
			scaling_M.at<float>(0,2) = 0;

			scaling_M.at<float>(1,0) = 0;
			scaling_M.at<float>(1,1) = Sy;
			scaling_M.at<float>(1,2) = 0;

			cv::warpAffine( warped_img, warped_img, scaling_M, warped_img.size());

			// Flip image
			uniform_real_distribution<double> th_uniform_dist(0.0, 1.0);
			if(th_uniform_dist(generator) > 0.5) {
				cv::Mat flipped_img;
				cv::flip(warped_img, flipped_img, 1);
				warped_img = flipped_img;
			}

			// Accumulate
			out_imgs[i].push_back(warped_img);
			out_labels[i].push_back(label);
		}

		out_imgs[_aug_factor].push_back(imgs[k]);
		out_labels[_aug_factor].push_back(labels[k]);

		if( out_imgs[0].size() > _transferSize ) {

			for(int e = 0; e < out_imgs.size(); e++) {
			
				copy_to_edge(out_imgs[e], out_labels[e], e);
				// Clean
				out_imgs[e].clear();
				out_labels[e].clear();
			}
		}
	}


	if( out_imgs[0].size() > 0 ) {
		for(int e = 0; e < out_imgs.size(); e++) {
			copy_to_edge(out_imgs[e], out_labels[e], e);
			// Clean
			out_imgs[e].clear();
			out_labels[e].clear();
		}
	}
}
