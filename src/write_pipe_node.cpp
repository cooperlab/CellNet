#include "write_pipe_node.h"
#include "edge.h"
#include "utils.h"
#define MAX_SIZE 10004

WritePipeNode::WritePipeNode(std::string id, std::string pipe_name): Node(id, 0), _pipe_name(pipe_name), _counter(0), _data_buffer(), _labels_buffer(){
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	_labels_buffer.clear();
}

void *WritePipeNode::run(){

	while(true){
		
		copy_from_buffer(_data_buffer, _labels_buffer);
		if(!_data_buffer.empty()){
			
			write_to_pipe();
		}
		else{

			// Check if all input nodes have already finished
			bool is_all_done = true;

			for(std::vector<int>::size_type i=0; i < _in_edges.size(); i++){

				if(!_in_edges.at(i)->is_in_node_done()){
					is_all_done = false;
				}
			}			

			// All input nodes have finished
			if(is_all_done){

				send_done_to_pipe();
				std::cout << "******************" << std::endl << "WritePipeNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

				break;
			}
		}
	}
	return NULL;
}

void WritePipeNode::write_to_pipe(){

	//Allocating the vector with the same size of the matrix
	for(int k=0; k < _data_buffer.size(); k++){

		_counter++;

		// Open a named pipe
		int pipe = open(_pipe_name.c_str(), O_WRONLY);

		cv::Mat img = _data_buffer[k];
		int label = _labels_buffer[k];

		//std::cout << "label: " << std::to_string((uint8_t)label) << std::endl;
		//std::cout << "channels: " << std::to_string((uint8_t)img.channels()) << std::endl;
		//std::cout << "cols: " << std::to_string((uint8_t)img.cols) << std::endl;
		//std::cout << "rows: " << std::to_string((uint8_t)img.rows) << std::endl;
		// DEBUG
		//cv::Mat channel[4];
	    // The actual splitting.
	    //cv::split(img, channel);
	    //for(int k=0; k < 4; k++){
	    //	cv::imshow(std::to_string(k), channel[k]);
	    //	cv::waitKey(0);
	    //}

	    // DEBUG

		// Convert Mats to byte stream
		std::vector<uint8_t> byte_stream(img.cols * img.rows * img.channels()); 
		memcpy(byte_stream.data(), img.data, byte_stream.size() * sizeof(uint8_t));

		// Insert header <height, width, channels, label, data>
		byte_stream.insert(byte_stream.begin(), (uint8_t)label );
		byte_stream.insert(byte_stream.begin(), (uint8_t)img.channels() );
		byte_stream.insert(byte_stream.begin(), (uint8_t)img.cols );
		byte_stream.insert(byte_stream.begin(), (uint8_t)img.rows );

		// Actually write out the data and close the pipe
		write(pipe, &byte_stream[0], byte_stream.size());

		// close the pipe
		close(pipe);
	}

	//std::cout << "Bytes:" << std::endl; 
	//for(int i = 0; i < buffer.size(); i++){

	//    printf("%02x ", (unsigned char) buffer[i] & 0xff);
	//}
	
	// Clear data
	_data_buffer.clear();
	_labels_buffer.clear();
}

void WritePipeNode::send_done_to_pipe(){

	// Open a named pipe
	int pipe = open(_pipe_name.c_str(), O_WRONLY);

	// Create buffer
	std::vector<uint8_t> buffer(MAX_SIZE);

	// Actually write out the data and close the pipe
	write(pipe, &buffer[0], buffer.size());

	// Close the pipe
	close(pipe);
}