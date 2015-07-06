#include "train_node.h"
#include "edge.h"
#include "utils.h"
#include <iostream>

TrainNode::TrainNode(std::string id, int mode): Node(id, mode){
	runtime_total_first = utils::get_time();
}

void *TrainNode::run(){

	increment_threads();
	while(true){

		std::vector<cv::Mat> out; 
		copy_chunk_from_buffer(out, _labels);
		if(!out.empty()){

			for(std::vector<cv::Mat>::size_type i=0; i < out.size(); i++){
				
				increment_counter();
				  
			}
		}
		else{

			bool is_done = true;
			for(int i =0; i < ){

			}

			 if(_in_edges.at(0)->is_in_node_done())
			}
			break;
		}
		out.clear();
	}

	if(check_finished() == true){

		std::cout << "******************" << std::endl << "TrainNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}