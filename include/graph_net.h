//
//	Copyright (c) 2015, Emory University
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
#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include <map> 
#include <boost/thread.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "node.h"
#include "edge.h"
#include "utils.h"
#include "debug_node.h"
#include "train_node.h"
#include "prediction_node.h"
#include "grayscale_node.h"
#include "augmentation_node.h"
#include "laplacian_pyramid_node.h"
#include "prediction_pipe_node.h"
#include "read_jpg_node.h"
#include "read_node.h"
#include "read_pipe_node.h"
#include "read_write_node.h"
#include "read_hdf5_node.h"
#include "write_hdf5_node.h"
#include "write_pipe_node.h"
#include "write_png_node.h"



using namespace std;




class GraphNet {
	
	public:
		enum Mode {Serial, Parallel};

		GraphNet(int mode);
		void *run();
		void add_node(Node *node);
		void add_edge(Edge *edge);

  	private:
  		void link();
  		void start_parallel();
  		void start_serial();
  		int _mode;
		boost::ptr_deque<Node> _nodes;
		boost::ptr_deque<Edge> _edges;
		std::map<std::string, int> _node_map;
};


#endif
