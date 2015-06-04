#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>
#include <cv.h>
#include "edge.h"

class Node : private boost::noncopyable{
	
	public:
		Node(std::string id);
		virtual void *run(){return NULL;};
		std::string get_id();
		void insert_in_edge(Edge *edge);
		void insert_out_edge(Edge *edge);
		bool _is_ready;
		bool _is_valid;
		std::string _id;
		std::vector<Edge *> _in_edges;
		std::vector<Edge *> _out_edges;
		
  	protected:
		void copy_to_buffer(std::vector<cv::Mat> out);
  		void copy_from_buffer(cv::Mat &);
};
#endif