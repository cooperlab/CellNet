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

	double 	loop, start = utils::get_time();

	increment_threads();

	while( true ) {

		vector<cv::Mat> out, gray_out;
 
		copy_chunk_from_buffer(out, _labels);
		int		loopSize;


		if( out.size() >= _transferSize ) {
			loopSize = out.size(); 

			//std::cout << "total labels in gray: " << std::to_string(_labels.size()) << std::endl;
			loop = utils::get_time();

			for(vector<cv::Mat>::size_type i=0; i < out.size(); i++) {
				increment_counter();

				// Convert to grayscale and equalize
				cv::Mat gray_img;
				cv::Mat equilized_img;
				cv::cvtColor(out.at(i), gray_img, CV_BGR2GRAY);
				cv::equalizeHist(gray_img, equilized_img);


				// Push to vector
				gray_out.push_back(equilized_img);

			}
			// Copy to buffer
			copy_to_buffer(gray_out, _labels);

			gray_out.clear();
			cout << "grey loop took: " << utils::get_time() - loop << " size: " << loopSize << endl;

		} else if( _in_edges.at(0)->is_in_node_done() ) {

			// TODO - Add check to see if there is data in the buffer. 
			break;
		}
		out.clear();
		_labels.clear();
	}

	if( check_finished() == true ) {

		cout << "******************" << endl 
			 << "GrayScaleNode complete" << endl 
			 << "Total_time_first: " << to_string(utils::get_time() - runtime_total_first) << endl 
			 << "# of elements: " << to_string(_counter) << endl 
			 << "******************" << endl;

		cout << "GrayScaleNode runtime: " << utils::get_time() - start << endl;

		// Notify it has finished
		for(vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}
	return NULL;
}
