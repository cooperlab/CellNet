#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include "node.h"
#include "edge.h"

class GraphNet {
	
	public:
		GraphNet();
		void run();
		void add_node(Node node){_nodes.insert(node)};
		void add_edge(Edge edge){_edges.insert(edge)};

  	private:
  		void link();
  		void start_parallel();
  		void start_serial();
  		void Graph::copy_to_buffer(int node_idx);
  		cv::Mat Graph::copy_from_buffer(int edge_idx);
		std::vector<Node> _nodes;
		std::vector<Node> _edges;
		std::map<char, int> _node_map;
};
#endif