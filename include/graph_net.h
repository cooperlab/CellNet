#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include "node.h"
#include "edge.h"
#include <map> 
#include <boost/thread.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

class GraphNet {
	
	public:
		GraphNet();
		void *run();
		void add_node(Node *node);
		void add_edge(Edge *edge);

  	private:
  		void link();
  		void start_parallel();
  		void start_serial();
		boost::ptr_deque<Node> _nodes;
		boost::ptr_deque<Edge> _edges;
		std::map<std::string, int> _node_map;
};
#endif