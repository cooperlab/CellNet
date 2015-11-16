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

#include "graph_net.h"


using namespace std;



DebugNode::DebugNode(std::string id, int mode) : 
Node(id, mode)
{
}




void *DebugNode::run(){
	int counter = 0;

	while(true){

		std::vector<cv::Mat> out; 
		copy_chunk_from_buffer(out, _labels);

		if( !out.empty() ){

			cout << "DebugNode: buffer has " << out.size() << " objects" << endl;
			string name; 

			for(int i = 0; i < out.size(); i++) {
				
				name = "Test" + to_string(counter++) + "_" + to_string(_labels[i]) + ".jpg";
				cv::imwrite(name.c_str(), out[i]);

			}
			//cv::imshow("img " + std::to_string(counter), out);
			//std::cout << std::to_string(counter++) << std::endl;
			// Debugger
			//std::cout << "DebugNode complete" << std::endl;
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			std::cout << "Stopping DebuggerNode" << std::endl;
			//cv::waitKey(0);
			// Some debug
			break;
		}
		out.clear();
	}

	// Notify it has finished
	for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
		_out_edges.at(i)->set_in_node_done();
	}

	return NULL;
}
