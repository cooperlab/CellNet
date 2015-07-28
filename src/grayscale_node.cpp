#include "grayscale_node.h"
#include "edge.h"
#include "utils.h"
#include <iostream>

GrayScaleNode::GrayScaleNode(std::string id, int mode): Node(id, mode){
	runtime_total_first = utils::get_time();
}

void *GrayScaleNode::run(){

	increment_threads();

	while(true){

		std::vector<cv::Mat> out; 
		copy_chunk_from_buffer(out, _labels);

		if(!out.empty()){

			//std::cout << "total labels in gray: " << std::to_string(_labels.size()) << std::endl;

			for(std::vector<cv::Mat>::size_type i=0; i < out.size(); i++){
				
				increment_counter();
				std::vector<cv::Mat> gray_out;

				// Convert to grayscale and equalize
				cv::Mat gray_img;
				cv::Mat equilized_img;
				cv::cvtColor(out.at(i), gray_img, CV_BGR2GRAY);
				cv::equalizeHist(gray_img, equilized_img);

				// Push to vector
				gray_out.push_back(equilized_img);

				// Copy to buffer
				copy_to_buffer(gray_out, _labels);
			}
		}

		else if(_in_edges.at(0)->is_in_node_done()){
			break;
		}
		out.clear();
	}

	if(check_finished() == true){

		std::cout << "******************" << std::endl << "GrayScaleNode complete" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}