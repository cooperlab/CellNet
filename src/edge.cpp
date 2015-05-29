#include "edge.h"

Edge::Edge(std::string in_node_id, std::string out_node_id): _in_node_id(in_node_id), _out_node_id(out_node_id), _buffer(){}
std::string Edge::get_in_node_id(){return _in_node_id;}
std::string Edge::get_out_node_id(){return _out_node_id;}
std::vector<cv::Mat> Edge::get_buffer(){return _buffer;}
void Edge::set_buffer(std::vector<cv::Mat> buffer){_buffer = buffer;}