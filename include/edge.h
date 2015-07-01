#ifndef _EDGE_H
#define _EDGE_H

#include <vector>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

class Edge : private boost::noncopyable{

	public:
		Edge(std::string id, std::string in_node_id, std::string out_node_id);
		std::string get_in_node_id();
		std::string get_out_node_id();
		std::vector<cv::Mat> *get_buffer();
		std::vector<double> *get_buffer_labels();
		bool is_in_node_done();
		bool is_empty();
		void set_buffer(std::vector<cv::Mat> buffer, std::vector<double> buffer_labels);
		void set_in_node_done();
		void show_image();
		std::string _id;
		bool _in_node_done;
		boost::mutex _mutex;

	private:
		std::string _in_node_id;
		std::string _out_node_id;
		std::vector<cv::Mat> _buffer;
		std::vector<double> _buffer_labels;
};
#endif
