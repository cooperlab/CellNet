//
//	Copyright (c) 2015-2016, Emory University
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification, are
//	permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this list of
//	conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice, this list
// 	of conditions and the following disclaimer in the documentation and/or other materials
//	provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//	OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//	SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//	BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
//	WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//	DAMAGE.
//
//
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
		~Edge();
		std::string get_in_node_id();
		std::string get_out_node_id();
		std::vector<cv::Mat> *get_buffer();
		std::vector<int> *get_buffer_labels();
		bool is_in_node_done();
		bool is_empty();
		void set_buffer(std::vector<cv::Mat> buffer, std::vector<int> buffer_labels);
		void set_in_node_done();
		void show_image();
		std::string _id;
		bool _in_node_done;
		boost::mutex _mutex;

	private:
		std::string _in_node_id;
		std::string _out_node_id;
		std::vector<cv::Mat> _buffer;
		std::vector<int> _buffer_labels;
};
#endif
