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



DebugNode::DebugNode(std::string id, int transferSize, int mode) : 
Node(id, mode),
_transferSize(transferSize)
{

}




void *DebugNode::run(){

	std::vector<cv::Mat> out; 
	double	start = utils::get_time();

	while( true ) {

		copy_chunk_from_buffer(out, _labels);

		// !!!!! Add debug code here  !!!!!
		out.clear();
		_labels.clear();

		if( _in_edges.at(0)->is_in_node_done() ) {
			break;
		}
	}

	
	// Notify it has finished
	vector<Edge *>::iterator	it;

	for(it = _out_edges.begin(); it != _out_edges.end(); it++) {
		(*it)->set_in_node_done();
	}

	cout << "******************" << endl 
		 << "DebugNode complete" << endl 
		 << "Run time: " << to_string(utils::get_time() - start) << endl 
		 << "# of elements: " << to_string(_counter) << endl 
		 << "******************" << endl;
}
