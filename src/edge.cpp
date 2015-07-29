#include "edge.h"

Edge::Edge(std::string id, std::string in_node_id, std::string out_node_id): _id(id), _in_node_done(false), _in_node_id(in_node_id), _out_node_id(out_node_id), _buffer(), _buffer_labels(){}
Edge::~Edge(){
	_buffer.clear();
	_buffer_labels.clear();
}
std::string Edge::get_in_node_id(){return _in_node_id;}
std::string Edge::get_out_node_id(){return _out_node_id;}
std::vector<cv::Mat> *Edge::get_buffer(){return &_buffer;}
std::vector<int> *Edge::get_buffer_labels(){return &_buffer_labels;}
void Edge::set_buffer(std::vector<cv::Mat> buffer, std::vector<int> buffer_labels){
	_buffer = buffer;
	_buffer_labels = buffer_labels;
}
void Edge::set_in_node_done(){_in_node_done = true;}
bool Edge::is_in_node_done(){return _in_node_done;}
bool Edge::is_empty(){return _buffer.empty();}
void Edge::show_image(){
	for(std::vector<cv::Mat>::size_type i =0; i < _buffer.size(); i++){
		std::string fname = "debug " + std::to_string(i);
		cv::imshow(fname, _buffer.at(i));
		cv::waitKey(0);
	}
}