#include "grayscale_node.h"
#include "edge.h"
#include "utils.h"
#include <iostream>

GrayScaleNode::GrayScaleNode(std::string id): Node(id){
	runtime_total_first = utils::get_time();
}

void *GrayScaleNode::run(){

	increment_threads();

	while(true){

		cv::Mat out; 
		copy_from_buffer(out);

		if(!out.empty()){
			increment_counter();

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
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			break;
		}
		out.release();
	}

	if(check_finished() == true){

		std::cout << "******************" << std::endl;
		std::cout << "GrayScaleNode complete" << std::endl;
		std::cout << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl;
		std::cout << "# of elements: " << std::to_string(_counter) << std::endl;
		std::cout << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}