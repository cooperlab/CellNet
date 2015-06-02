#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>
#include <cv.h>
#include "edge.h"

class Node {
	
	public:
		Node(std::string id);
		virtual void *run(){return NULL;};
		std::string get_id();
		void insert_in_edge(Edge *edge);
		void insert_out_edge(Edge *edge);
		
  	protected:
		bool _is_ready;
		bool _is_valid;
		std::string _id;
		std::vector<Edge *> _in_edges;
		std::vector<Edge *> _out_edges;
		void copy_to_buffer(std::vector<cv::Mat> out);
  		void copy_from_buffer(cv::Mat *);
  		std::vector<cv::Mat> _buffer;
};
#endif