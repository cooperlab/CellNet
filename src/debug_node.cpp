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
				
				name = "Test" + to_string(i) + "_" + to_string(_labels[i]) + ".jpg";
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
