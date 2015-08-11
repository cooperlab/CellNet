#include "read_pipe_node.h"
#include "edge.h"
#include "utils.h"
#define MAX_SIZE 10004

ReadPipeNode::ReadPipeNode(std::string id, std::string pipe_name, int mode): Node(id, mode), _pipe_name(pipe_name), _counter(0){
	runtime_total_first = utils::get_time();
	_counter = 0;
}

void *ReadPipeNode::run(){

	while(true){

		// Request data
		std::vector<cv::Mat> outs;
		std::vector<int> labels;
		int res = read_from_pipe(outs, labels);
		if(res){

			if(!outs.empty()){
	
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
	
	// Open a named pipe
	int pipe = open(_pipe_name.c_str(), O_RDONLY);

	if(pipe != 0){

		// Create buffer
		std::vector<uint8_t> buffer(MAX_SIZE);
		
		// Format <height, width, channels, label, data>
		// Read header
		read(pipe, &buffer[0], buffer.size());
		int height = (int)buffer[0];
		int width = (int)buffer[1];
		int channels = (int)buffer[2];

		// DEBUG
		std::cout << "height: " << height << std::endl;
		std::cout << "width: " << width << std::endl;
		std::cout << "channels: " << channels << std::endl;

		if((height > 0) && (width > 0) && (channels > 0)){

			// Increment
			_counter++;

			// Read 
			int label = (int) buffer[3];
			cv::Mat img(height, width, CV_8UC4);
			memcpy(img.data, &buffer[4], height * width * channels * sizeof(uint8_t));
			std::vector<cv::Mat> vec_img;
			vec_img.push_back(img);

			// DEBUG
			cv::Mat channel[4];
		    cv::split(img, channel);
		    for(int k=0; k < 4; k++){
		    	cv::imwrite("./test/image" + std::to_string(_counter) + std::to_string(k) + ".jpg", channel[k]);
		    }
			// DEBUG

			outs = vec_img;
			labels.push_back(label);
			vec_img.clear();

			// close the pipe
			close(pipe);
			return 1;
		}
		else{

			return 0;
		}
	}else{

		std::cout << "Could not open pipe" << std::endl;
	}
	
	return 0;
}