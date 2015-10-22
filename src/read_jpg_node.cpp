#include "read_jpg_node.h"
#include "edge.h"
#include <iostream>
#include <sstream>
#include "utils.h"
#define SHIFT 25

ReadJPGNode::ReadJPGNode(std::string id, std::vector<std::string> slides_name, std::string path, int mode): Node(id, mode), _slides_name(slides_name), _path(path), i_ptr(0){
	
	 runtime_total_first = utils::get_time();
}

void *ReadJPGNode::run(){
	
	increment_threads();

	// Execute
	int i = get_input();
	while(i >= 0){

		open_images(_slides_name[i]);
		copy_to_buffer(_input_data, _input_labels);
		_input_data.clear();
		_input_labels.clear();

		// Execute
		i = get_input();
	}

	if( check_finished() == true){

		std::cout << "******************" << std::endl << "ReadJPGNode complete" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}
	return NULL;
}	

int ReadJPGNode::get_input(){

	boost::mutex::scoped_lock lk(_mutex);
	if(i_ptr < _slides_name.size()){
		return i_ptr++;
	}
	else{
		return -1;
	}
}


// This method opens an image using openslide and removes the alpha channel
void ReadJPGNode::open_images(std::string image_path){
	
	std::vector<std::string> images_name;
	images_name = utils::get_images_path(_path + image_path);
	for(int k=0; k < images_name.size(); k++){
		int size = images_name[k].size();
		int label = (images_name[k].at(size - 5) - '0');
		_input_labels.push_back(label);
		cv::Mat image = cv::imread(_path  + image_path + "/" + images_name[k], CV_LOAD_IMAGE_COLOR);
		_input_data.push_back(image);
		_counter++;
	}	
}
