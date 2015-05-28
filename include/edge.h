#ifndef _EDGE_H
#define _EDGE_H

#include <vector>
#include <opencv2/core/core.hpp>

class Edge{

	public:
		Edge(std::string in_node_id, std::string out_node_id);
		std::string get_in_node_id(){return _in_node_id};
		std::string get_out_node_id(){return _out_node_id};
		std::vector<cv::Mat> get_buffer(){return_buffer};
		std::vector<cv::Mat> set_buffer(std::vector<cv::Mat> buffer){_buffer = buffer};

	protected:
		std::string _in_node_id;
		std::string _out_node_id;
		std::vector<cv::Mat> _buffer;
};
#endif
