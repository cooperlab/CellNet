#include "read_pipe_node.h"
#include "edge.h"
#include "utils.h"

ReadPipeNode::ReadPipeNode(std::string id, std::string pipe_name, int mode): Node(id, mode), _pipe_name(pipe_name), _counter(0), _pipe(-1){
	runtime_total_first = utils::get_time();
	_counter = 0;
	_pipe = open(_pipe_name.c_str(), O_RDONLY);
	if(_pipe == -1){
		std::cout << "Fail to open pipe" << std::endl;
		return;
	}
}

void *ReadPipeNode::run(){

	while(true){

		// Request data
		std::vector<cv::Mat> outs;
		std::vector<int> labels;
		int res = read_from_pipe(outs, labels);
		if(res){

			if(!outs.empty()){
	=
				// Copy to buffer
				copy_to_buffer(outs, labels);

				// Clean vectors
				outs.clear();
				labels.clear();
			}
			else{
				std::cout << "Empty" << std::endl;
			}
		}
		else{

			std::cout << "******************" << std::endl << "ReadPipeNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

			// Notify it has finished
			for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
				_out_edges.at(i)->set_in_node_done();
			}
			break;
		}
	}
	return NULL;
}

int ReadPipeNode::read_from_pipe(std::vector<cv::Mat> &outs, std::vector<int> &labels){
	
	// Create buffer
	std::vector<uint8_t> buffer(4);
	
	// Format <height, width, channels, label, data>
	// Read header
	int res = read(_pipe, &buffer[0], buffer.size());
	while(res == 0){
		res = read(_pipe, &buffer[0], buffer.size());
	}
	
	int height = (int)buffer[0];
	int width = (int)buffer[1];
	int channels = (int)buffer[2];
	int label = (int)buffer[3];

	if((height > 0) && (width > 0) && (channels > 0)){

		// Increment
		_counter++;

		// Read 
		std::vector<uint8_t> buffer_data(height * width * channels);
                res = read(_pipe, &buffer_data[0], buffer_data.size());
		while(res == 0){
                       	res = read(_pipe, &buffer_data[0], buffer_data.size());
               	}


		cv::Mat img(height, width, CV_8UC(channels));
		memcpy(img.data, &buffer_data[0], height * width * channels * sizeof(uint8_t));
		std::vector<cv::Mat> vec_img;
		vec_img.push_back(img);

		outs = vec_img;
		labels.push_back(label);
		vec_img.clear();
		return 1;
	}
	return 0;
}
