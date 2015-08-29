#include "read_node_jpg.h"
#include "edge.h"
#include <iostream>
#include <sstream>
#include "utils.h"
#define SHIFT 25

ReadNodeJPG::ReadNodeJPG(std::string id, std::vector<std::string> image_paths, std::vector<std::vector<int>> input_labels, int mode): Node(id, mode), _image_paths(image_paths), i_ptr(0), _input_labels(input_labels){
	
	 runtime_total_first = utils::get_time();
}

void *ReadNodeJPG::run(){
	
	increment_threads();
	cv::Mat entire_image;

	// Execute
	int i = get_input();
	while(i >= 0){

		increment_counter();
		entire_image = open_image(_image_paths[i]);
		copy_to_buffer(entire_image, _input_labels[i]);

		// Execute
		i = get_input();
	}

	/****************** Debug ******************/
	//show_entire_image(entire_image);

	// Release memory
	entire_image.release();
	
	//show_cropped_cells();
	if( check_finished() == true){

		std::cout << "******************" << std::endl << "ReadNode complete" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}
	return NULL;
}	

int ReadNodeJPG::get_input(){

	boost::mutex::scoped_lock lk(_mutex);
	if(i_ptr < _image_paths.size()){
		return i_ptr++;
	}
	else{
		return -1;
	}
}


// This method opens an image using openslide and removes the alpha channel
cv::Mat ReadNodeJPG::open_image(std::string image_path){

	return cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
}