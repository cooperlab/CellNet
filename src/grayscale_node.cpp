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
#include <iostream>

#include "grayscale_node.h"
#include "edge.h"
#include "utils.h"



using namespace std;





GrayScaleNode::GrayScaleNode(string id, int transferSize, int mode) : 
Node(id, mode),
_transferSize(transferSize)
{
	runtime_total_first = utils::get_time();
}







void *GrayScaleNode::run()
{
	vector<cv::Mat> out; 
	double	start = utils::get_time();

	increment_threads();

	while( true ) {

		copy_chunk_from_buffer(out, _labels);

		if( out.size() >= _transferSize ) {

			Convert(out);
			out.clear();
			_labels.clear();

		} else if( _in_edges.at(0)->is_in_node_done() ) {

			// Sending node is done, check for remaining data.
			if( out.size() > 0 ) {
				Convert(out);
			} 
			break;
		}
	}

	if( check_finished() == true ) {

		cout << "******************" << endl 
			 << "GrayScaleNode complete" << endl 
			 << "Run time: " << to_string(utils::get_time() - start) << endl 
			 << "# of elements: " << to_string(_counter) << endl 
			 << "******************" << endl;

		// Notify it has finished
		for(vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}
	return NULL;
}





void GrayScaleNode::Convert(vector<cv::Mat> images) 
{
	vector<cv::Mat> 			grayOut; 
	vector<cv::Mat>::iterator	it;

	for(it = images.begin(); it != images.end(); it++) {
		increment_counter();

		// Convert to grayscale and equalize
		cv::Mat gray_img;
		cv::Mat equilized_img;
		cv::cvtColor(*it, gray_img, CV_BGR2GRAY);
		cv::equalizeHist(gray_img, equilized_img);

		grayOut.push_back(equilized_img);
	}
	// Copy to buffer
	copy_to_buffer(grayOut, _labels);
}
