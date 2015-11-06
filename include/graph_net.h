#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include <map> 
#include <boost/thread.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "node.h"
#include "edge.h"
#include "utils.h"
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
#include "write_hdf5_node.h"
#include "write_pipe_node.h"
#include "write_png_node.h"



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
