//
//	Copyright (c) 2015-2016, Emory University
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
#include "laplacian_pyramid_node.h"
#include "utils.h"


using namespace std;




#define KERNEL_SIZE 	3
#define N_LAYERS 		4



LaplacianPyramidNode::LaplacianPyramidNode(string id, int transferSize, int mode) : 
Node(id, mode),
_transferSize(transferSize)
{
}





void *LaplacianPyramidNode::run()
{
	vector<cv::Mat> 	layer0; 

	increment_threads();

	// Start of runtime 
	_runtimeStart = utils::get_time();


	while(true) {

		copy_chunk_from_buffer(layer0, _labels);

		if( layer0.size() >= _transferSize ) {

			cout << "Appllying laplacian to " << layer0.size() << " images" << endl;

			Convert(layer0);
			layer0.clear();
			_labels.clear();

		} else if(_in_edges.at(0)->is_in_node_done() ) {

			if( layer0.size() > 0 ) {
				Convert(layer0);
			}
			break;
		}
	}


	if( check_finished() == true ) {
	
		cout << "******************" <<  endl 
			 << "LaplacianPyramidNode complete" << endl 
			 << "Run time: " << to_string(utils::get_time() - _runtimeStart) << endl 
			 << "# of elements: " << to_string(_counter) << endl 
			 << "******************" << endl;
		
		// Notify it has finished
		for(int i = 0; i < _out_edges.size(); i++ ) {
			_out_edges.at(i)->set_in_node_done();
		}
	}
	return NULL;
}







void LaplacianPyramidNode::Convert(vector<cv::Mat> imgs)
{
	cv::Mat 	gaussian_layer0, v1, v2, f0, g0;
	cv::Mat 	merged_layers, temp;
	vector<cv::Mat> 	pyramidsOut;
	vector<cv::Mat> 	layers;
	

	for(int i = 0; i < imgs.size(); i++) {

		// Initialize
		increment_counter();

		cv::GaussianBlur(imgs[i], gaussian_layer0, cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0);
		cv::resize(gaussian_layer0, v1, cv::Size(), 0.5f, 0.5f, cv::INTER_CUBIC);
		cv::resize(gaussian_layer0, v2, cv::Size(), 0.25f, 0.25f, cv::INTER_CUBIC);

		cv::resize(v1, temp, imgs[i].size(), 0, 0, cv::INTER_CUBIC);
		f0 = imgs[i] - temp;
		temp.release();
		
	
		cv::resize(v2, temp, v1.size(), 0, 0, cv::INTER_CUBIC);
		g0 = v1 - temp;
		temp.release();

		layers.resize(3);

		layers[2] = f0;
		cv::resize(g0, layers[1], f0.size(), 0, 0, cv::INTER_CUBIC);
		cv::resize(v2, layers[0], f0.size(), 0, 0, cv::INTER_CUBIC);


		cv::merge(layers, merged_layers);
		pyramidsOut.push_back(merged_layers);

		// Release memory
		gaussian_layer0.release();
		layers.clear();
		merged_layers.release();
		
	}
	copy_to_buffer(pyramidsOut, _labels);		
}




