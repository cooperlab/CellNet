#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>
#include <cv.h>

class Node {
	
	public:
		Node(std::string id);
		virtual void run(){};
		virtual void set_target(cv::Mat target){};
		virtual bool get_output(std::vector<cv::Mat> &out){return false;};
		std::string get_id();
		void insert_in_edge(int id);
		void insert_out_edge(int id);
		std::vector<int> get_in_edges_ids();
		std::vector<int> get_out_edges_ids();
		
  	protected:
		bool _is_ready;
		bool _is_valid;
		std::string _id;
		std::vector<int> _in_edges_ids;
		std::vector<int> _out_edges_ids;
};
#endif