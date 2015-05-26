#ifndef _EDGE_H
#define _EDGE_H

#include <vector>
#include <opencv2/core/core.hpp>

class Edge{

	public:
		Edge(int id, int in_node_id, int out_node_id);
	
	protected:
		int _id;
		int _in_node_id;
		int _out_node_id;
		cv::vector<cv::Mat> _buffer;
};
#endif
