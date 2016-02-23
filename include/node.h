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
#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>
#include <cv.h>
#include "edge.h"


using namespace std;



class Node : private boost::noncopyable{
	
	public:

		enum Mode {Repeat, Alternate, Chunk};

		Node(std::string id, int mode);
		~Node();

		virtual void *run()	{	return NULL; }
		string get_id() {  return _id;	}
		void insert_in_edge(Edge *edge);
		void insert_out_edge(Edge *edge);
		
		vector<Edge *>	get_in_edges(void) { return _in_edges;	}
 		vector<Edge *>	get_out_edges(void) { return _out_edges;  }

  	protected:
		
		int 		_mode;
		string 		_id;
		vector<int> _labels;
		int			_curSendEdge;

		double 		_runtimeStart;
		boost::mutex _mutex;
		boost::mutex _mutex_counter;
		boost::mutex _mutex_ctrl;
		long long unsigned int _counter_threads;
		long long unsigned int _counter;
		
		vector<Edge *> _in_edges;
		vector<Edge *> _out_edges;



		void 	copy_to_buffer(std::vector<cv::Mat> out, std::vector<int> &labels);
  		void 	copy_from_buffer(std::vector<cv::Mat> &, std::vector<int> &labels);
  		void 	copy_chunk_from_buffer(std::vector<cv::Mat> &out, std::vector<int> &labels);
		void 	copy_to_edge(vector<cv::Mat>& out, vector<int>& labels, int edge);
  		void	increment_counter();
  		void 	increment_threads();
  		bool 	check_finished();
};
#endif
