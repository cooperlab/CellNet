#include "edge.h"

Edge::Edge(std::string in_node_id, std::string out_node_id): _in_node_done(false), _in_node_id(in_node_id), _out_node_id(out_node_id), _buffer(){

	// Initialize mutex for buffer access
	pthread_mutex_destroy(&_mutex);
}
std::string Edge::get_in_node_id(){return _in_node_id;}
std::string Edge::get_out_node_id(){return _out_node_id;}
std::vector<cv::Mat> Edge::get_buffer(){return _buffer;}
void Edge::set_buffer(std::vector<cv::Mat> buffer){_buffer = buffer;}
void Edge::set_in_node_done(){_in_node_done = true;}
bool Edge::is_in_node_done(){return _in_node_done;}
bool Edge::is_empty(){return _buffer.empty();}
void Edge::lock_access(){pthread_mutex_lock(&_mutex);}
void Edge::unlock_access(){pthread_mutex_unlock(&_mutex);}
void Edge::show_image(){
	for(std::vector<cv::Mat>::size_type i =0; i < _buffer.size(); i++){
		std::string fname = "debug " + std::to_string(i);
		cv::imshow(fname, _buffer.at(i));
		cv::waitKey(0);
	}
}