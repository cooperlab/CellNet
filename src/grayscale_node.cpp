#include "grayscale_node.h"
#include "edge.h"
#include <iostream>

GrayScaleNode::GrayScaleNode(std::string id): Node(id){
}

void *GrayScaleNode::run(){

	while(true){

		cv::Mat out; 
		copy_from_buffer(out);

		if(!out.empty()){

			std::cout << "GrayScaleNode start" << std::endl; 
			
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

			std::cout << "GrayScaleNode complete" << std::endl;
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			std::cout << "Stopping GrayScaleNode" << std::endl;
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