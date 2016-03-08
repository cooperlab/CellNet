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
#include <iostream>

#include "graph_net.h"


using namespace std;



WriteImageNode::WriteImageNode(std::string id, bool split, int transferSize, int mode) : 
Node(id, mode),
_transferSize(transferSize),
_split(split)
{

}




void *WriteImageNode::run(){

	std::vector<cv::Mat> out; 
	double	start = utils::get_time();

	while( true ) {

		copy_chunk_from_buffer(out, _labels);

		if( out.size() >= _transferSize ) {

			SaveImages(out);

			out.clear();
			_labels.clear();

		} else if( _in_edges.at(0)->is_in_node_done() ) {

			// Check for any data leftover
			if( out.size() > 0 ) {

				SaveImages(out);

				out.clear();
				_labels.clear();
			}
			break;
		}
	}

	
	// Notify it has finished
	vector<Edge *>::iterator	it;

	for(it = _out_edges.begin(); it != _out_edges.end(); it++) {
		(*it)->set_in_node_done();
	}

	cout << "******************" << endl 
		 << "WriteImageNode complete" << endl 
		 << "Run time: " << to_string(utils::get_time() - start) << endl 
		 << "# of elements: " << to_string(_counter) << endl 
		 << "******************" << endl;

	return NULL;
}





void WriteImageNode::SaveImages(vector<cv::Mat> images)
{
	string name; 
	vector<cv::Mat>		layers;
	
		
	for(int i = 0; i < images.size(); i++) {

		if( _split ) {
			cv::split(images[i], layers);


			for(int l = 0; l < layers.size(); l++) {

				name = "Test" + to_string(_counter) + "_" + to_string(_labels[i]) 
						+ "_" + to_string(l) + ".jpg";
			
				cv::imwrite(name.c_str(), layers[l]);
			}
		} else {
			name = "Test" + to_string(_counter) + "_" + to_string(_labels[i]) 
				 + ".jpg";

			if( images[i].channels() == 3 ) {
				cv::cvtColor(images[i], images[i], CV_BGR2RGB);
			}
			cv::imwrite(name.c_str(), images[i]);
		}
		increment_counter();
	}
}

