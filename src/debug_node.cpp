#include "debug_node.h"
#include "edge.h"
#include <iostream>

DebugNode::DebugNode(std::string id): Node(id){
}

void *DebugNode::run(){
	int counter = 0;

	while(true){

		cv::Mat out; 
		copy_from_buffer(out);
		if(!out.empty()){

			std::cout << "DebugNode start" << std::endl; 
			cv::imshow("img " + std::to_string(counter), out);
			std::cout << std::to_string(counter++) << std::endl;
			// Debugger
			std::cout << "DebugNode complete" << std::endl;
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			std::cout << "Stopping DebuggerNode" << std::endl;
			cv::waitKey(0);
			// Some debug
			break;
		}
		out.release();
	}

	// Notify it has finished
	for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
		_out_edges.at(i)->set_in_node_done();
	}

	return NULL;
}