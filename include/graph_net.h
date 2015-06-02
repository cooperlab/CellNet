#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include "node.h"
#include "edge.h"
#include <map> 
#include <boost/thread.hpp>



class GraphNet {
	
	public:
		GraphNet();
		void run();
		void add_node(Node *node);
		void add_edge(Edge edge);

  	private:
  		void link();
  		void start_parallel();
  		void start_serial();
		std::vector<Node*> _nodes;
		std::vector<Edge> _edges;
		std::map<std::string, int> _node_map;
};
#endif