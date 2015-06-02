#include "debug_node.h"
#include "edge.h"
#include <iostream>

DebugNode::DebugNode(std::string id): Node(id){
}

void *DebugNode::run(){

	//while(true){

		//cv::Mat out = copy_from_buffer();
		//if(!out.empty()){
		//	std::cout << "DebugNode start" << std::endl; 

			// Debugger
		//	std::cout << "DebugNode complete" << std::endl;
		//}
		//else if(_in_edges.at(0)->is_in_node_done()){
			
			// Some debug
			//std::cout << "Generated images " << std::to_string(output.size()) << std::endl;
			//std::cout << "Showing images "  << std::endl;	
			//for(std::vector<cv::Mat>::size_type i =0; i < _output.size(); i++){
	
			//	std::string fname = "debug " + std::to_string(i);
			//	cv::imshow(fname, _output.at(i));
			//}
			//cv::waitKey(0);
			//break;
		//}
		//out.release();
	//}

	// Notify it has finished
	//for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
	//	_out_edges.at(i)->set_in_node_done();
	//}

	return NULL;
}