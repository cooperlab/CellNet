#include "write_node.h"
#include "edge.h"

WriteNode::WriteNode(std::string id, std::string fname): Node(id), _fname(fname), _counter(0){}

void *WriteNode::run(){

	while(true){
		cv::Mat out;
		copy_from_buffer(out);
		if(!out.empty()){
			std::cout << "Writing sample" << std::endl; 
			write_to_disk(out);
			out.release();
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			std::cout << "Stopping Write Node" << std::endl;
			break;
		}
	}
	return NULL;
}

void WriteNode::write_to_disk(cv::Mat &out){

	cv::imwrite(_fname + "/sample_" + std::to_string(_counter++) + ".png", out);	
}