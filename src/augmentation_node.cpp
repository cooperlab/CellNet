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
#include <cmath>
#include <random>
#include <chrono>
#include "augmentation_node.h"
#include "edge.h"
#include "utils.h"



#define PI 3.1415926535897932
#define SHIFT 25



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
	double 	start = utils::get_time();

	while(true) {

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);

		if( _data_buffer.size() >= _transferSize ) {

			augment_images(_data_buffer, _labels_buffer);

			// Clear buffers for next block of data
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

				if( _data_buffer.size() > 0 ) {
					augment_images(_data_buffer, _labels_buffer);
					_data_buffer.clear();
					_labels_buffer.clear();
				}

			
				cout << "******************" << endl 
					 << "AugmentationNode" << endl 
					 << "Run time: " << to_string(utils::get_time() - start) << endl 
					 << "# of elements: " << to_string(_counter) << endl 
					 << "******************" << endl;

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



//	TODO
// ???? Will this generalize to color images?
//
void AugmentationNode::augment_images(vector<cv::Mat> imgs, vector<int> labels)
{
	vector< vector<cv::Mat> > 	out_imgs;
	vector< vector<int> > 		out_labels;

	out_imgs.resize(_aug_factor + 1);
	out_labels.resize(_aug_factor + 1);


	for(int k=0; k < imgs.size(); k++){
		_counter++;
		//cv::Mat src;
		
		// Expand source image
		cv::Mat	expImg(imgs[k].rows * 2, imgs[k].cols * 2, imgs[k].type(), 255);
		imgs[k].copyTo(expImg(cv::Rect(SHIFT, SHIFT, imgs[k].cols, imgs[k].rows)));
	
		
		// Get ROI
		float tl_row = expImg.rows/2.0F - SHIFT;
		float tl_col = expImg.cols/2.0F - SHIFT;
		float br_row = expImg.rows/2.0F + SHIFT;
		float br_col = expImg.cols/2.0F + SHIFT;
		cv::Point tl(tl_row, tl_col);
		cv::Point br(br_row, br_col);

		// Setup a rectangle to define region of interest
		cv::Rect cellROI(tl, br);

		for(int i=0; i < _aug_factor; i++) {
			_counter++;

			// Define variables
			int label = labels[k];

			// construct a trivial random generator engine from a time-based seed:
	  		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	  		default_random_engine generator(seed);
			uniform_real_distribution<double> uniform_dist(0.0, 1.0);

			double Flipx = round(uniform_dist(generator)); 				// randomly distributed reflections over x and y
			double Flipy = round(uniform_dist(generator));
			double Phi = 360 * uniform_dist(generator); 			// rotation angle uniformly distributed on [0, 2pi] radians
			double Tx = 20 * (uniform_dist(generator) - 0.5); 			// 10-pixel uniformly distributed translation
			double Ty = 20 * (uniform_dist(generator) - 0.5);
			double Shx = PI/180 * 20 * (uniform_dist(generator)-0.5);  	// shear angle uniformly distributed on [-20, 20] degrees
			double Shy = PI/180 * 20 * (uniform_dist(generator)-0.5);
			double Sx = 1/1.2 + (1.2 - 1/1.2)*uniform_dist(generator); 	// scale uniformly distributed on [1/1.2, 1.2]
			double Sy = 1/1.2 + (1.2 - 1/1.2)*uniform_dist(generator);

			cv::Mat warped_img;

			// Translation
			cv::Mat trans_M(3, 3, CV_64F);
			trans_M.at<double>(0,0) = 1;
			trans_M.at<double>(0,1) = 0;
			trans_M.at<double>(0,2) = Tx;

			trans_M.at<double>(1,0) = 0;
			trans_M.at<double>(1,1) = 1;
			trans_M.at<double>(1,2) = Ty;

			trans_M.at<double>(2,0) = 0;
			trans_M.at<double>(2,1) = 0;
			trans_M.at<double>(2,2) = 0;

			// Scaling
			cv::Mat scaling_M(3, 3, CV_64F);
			scaling_M.at<double>(0,0) = Sx;
			scaling_M.at<double>(0,1) = 0;
			scaling_M.at<double>(0,2) = 0;

			scaling_M.at<double>(1,0) = 0;
			scaling_M.at<double>(1,1) = Sy;
			scaling_M.at<double>(1,2) = 0;

			scaling_M.at<double>(2,0) = 0;
			scaling_M.at<double>(2,1) = 0;
			scaling_M.at<double>(2,2) = 0;

			// Shearing 
			cv::Mat shear_M(3, 3, CV_64F);
			shear_M.at<double>(0,0) = 1;
			shear_M.at<double>(0,1) = Shy;
			shear_M.at<double>(0,2) = 0;

			shear_M.at<double>(1,0) = Shx;
			shear_M.at<double>(1,1) = 1;
			shear_M.at<double>(1,2) = 0;

			shear_M.at<double>(2,0) = 0;
			shear_M.at<double>(2,1) = 0;
			shear_M.at<double>(2,2) = 0;

			// Rotation 
			cv::Mat rot(2, 3, CV_64F);
			cv::Point2f center(expImg.rows/2.0F, expImg.cols/2.0F);
			rot = getRotationMatrix2D(center, Phi, 1.0);	

			cv::Mat rot_M(3, 3, CV_64F, cv::Scalar(0));
			rot(cv::Rect(0,0,3,2)).copyTo(rot_M.colRange(0,3).rowRange(0,2));

			// Accumulate
			cv::Mat acc(3, 3, CV_64F);

			acc = trans_M * scaling_M * shear_M * rot_M;

			cv::Mat acc_M = acc.colRange(0,3).rowRange(0,2);
			cv::warpAffine( expImg, warped_img, acc_M, expImg.size());

			// Crop Image
			double m00 = acc_M.at<double>(0,0);
			double m01 = acc_M.at<double>(0,1);
			double m10 = acc_M.at<double>(1,0);
			double m11 = acc_M.at<double>(1,1);
			double m02 = acc_M.at<double>(0,2);
			double m12 = acc_M.at<double>(1,2); 

			//cout << m00 << "," << m01 << "," << m02 << endl; 
			//cout << m10 << "," << m11 << "," << m12 << endl;

			int new_cx = expImg.rows/2.0F * m00 + expImg.cols/2.0F * m01 + m02;
			int new_cy = expImg.rows/2.0F * m10 + expImg.cols/2.0F * m11 + m12;

			// Get ROI
			double tl_row = new_cx - SHIFT;
			double tl_col = new_cy - SHIFT;
			double br_row = new_cx + SHIFT;
			double br_col = new_cy + SHIFT;
			cv::Point tl(tl_row, tl_col);
			cv::Point br(br_row, br_col);

			//cout << scaling_M << endl;
			//cout << new_cx << " " << new_cy << endl;
			//cout << tl << endl;
			//cout << br << endl;
			//cout << "*****" << endl; 
			

			cv::Rect new_cellROI(tl, br);
			cv::Mat final_img = warped_img(new_cellROI);

			// Flip
			if( Flipx == 1 && Flipy == 1) {
				cv::flip(final_img, final_img, -1);
			}
			else if(Flipx == 0 && Flipy == 1){
				cv::flip(final_img, final_img, 1);
			}
			else if(Flipx == 1 && Flipy == 0){
				cv::flip(final_img, final_img, 0);
			}
			
        	if(!final_img.isContinuous()){
				final_img=final_img.clone();
			}

			// Accumulate
			out_imgs[i].push_back(final_img);
			out_labels[i].push_back(label);
		}

		// Crop Image
//		cv::Mat new_src = src(cellROI);
//		if(!new_src.isContinuous()){
//			new_src=new_src.clone();
//		}

		out_imgs[_aug_factor].push_back(imgs[k]);
		out_labels[_aug_factor].push_back(labels[k]);

		// TODO - For now we are overriding the mode until we get Mode::Alternate 
		// working properly
		if( out_imgs[0].size() > _transferSize ) {
			for(int e = 0; e < out_imgs.size(); e++) {
			
				copy_to_edge(out_imgs[e], out_labels[e], e % _out_edges.size());
				// Clean
				out_imgs[e].clear();
				out_labels[e].clear();
			}
		}
	}

	// Send any leftover data
	if( out_imgs[0].size() > 0 ) {

		// TODO - For now we are overriding the mode until we get Mode::Alternate 
		// working properly
		for(int e = 0; e < out_imgs.size(); e++) {
			copy_to_edge(out_imgs[e], out_labels[e], e % _out_edges.size());
			// Clean
			out_imgs[e].clear();
			out_labels[e].clear();
		}
	}
}


