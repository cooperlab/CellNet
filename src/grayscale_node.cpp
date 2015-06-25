#include "grayscale_node.h"
#include "edge.h"
#include "utils.h"
#include <iostream>

GrayScaleNode::GrayScaleNode(std::string id): Node(id){
}

void *GrayScaleNode::run(){

	while(true){

		cv::Mat out; 
		copy_from_buffer(out);

		if(!out.empty()){

			double begin_time = utils::get_time();
			//std::cout << "GrayScaleNode start" << std::endl; 
			
			// Convert to grayscale and equalize
			cv::Mat gray_img;
			cv::Mat equilized_img;

			cv::cvtColor(out, gray_img, CV_BGR2GRAY);
			cv::equalizeHist(gray_img, equilized_img);

			// Push to vector
			std::vector<cv::Mat> gray_out;
			gray_out.push_back(equilized_img);

			// Copy to buffer
			copy_to_buffer(gray_out);
			count++;

			runtime_average_first += float( utils::get_time() - begin_time );
			//std::cout << "Time to convert to grayscale: " << float( utils::get_time() - begin_time )  << std::endl;
			//std::cout << "GrayScaleNode complete" << std::endl;
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			std::cout << "******************" << std::endl;
			std::cout << "GrayScaleNode complete" << std::endl;
			std::cout << "Avg_first: " << std::to_string(runtime_average_first/count) << std::endl;
			std::cout << "Total_time_first: " << std::to_string(runtime_average_first) << std::endl;
			std::cout << "******************" << std::endl;

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