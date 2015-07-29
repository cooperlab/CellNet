#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>
#include <cv.h>
#include "edge.h"

class Node : private boost::noncopyable{
	
	public:
		Node(std::string id, int mode);
		~Node();
		virtual void *run(){return NULL;};
		std::string get_id();
		void insert_in_edge(Edge *edge);
		void insert_out_edge(Edge *edge);
		std::string _id;
		int _mode;
		std::vector<Edge *> _in_edges;
		std::vector<Edge *> _out_edges;
		long long unsigned int _counter;
		double runtime_total_first;
		boost::mutex _mutex;
		boost::mutex _mutex_counter;
		boost::mutex _mutex_ctrl;
		long long unsigned int _counter_threads;
		int ctrl;
		std::vector<int> _labels;
		
  	protected:
		void copy_to_buffer(std::vector<cv::Mat> out, std::vector<int> &labels);
  		void copy_from_buffer(cv::Mat &, int &labels);
  		void copy_chunk_from_buffer(std::vector<cv::Mat> &out, std::vector<int> &labels);
  		void increment_counter();
  		void increment_threads();
  		bool check_finished();
};
#endif