#ifndef _EDGE_H
#define _EDGE_H

#include <vector>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <pthread.h>

class Edge{

	public:
		Edge(std::string in_node_id, std::string out_node_id);
		std::string get_in_node_id();
		std::string get_out_node_id();
		std::vector<cv::Mat> get_buffer();
		bool is_in_node_done();
		bool is_empty();
		void set_buffer(std::vector<cv::Mat> buffer);
		void set_in_node_done();
		void lock_access();
		void unlock_access();
		void show_image();
		bool _in_node_done;

	protected:
		std::string _in_node_id;
		std::string _out_node_id;
		std::vector<cv::Mat> _buffer;
		pthread_mutex_t _mutex;
};
#endif
