#include "graph_net.h"

GraphNet::GraphNet(): _nodes(), _edges(), _node_map(){}

void Graph::run(){
	link();
}

void Graph::link(){
	map<char, int>::iterator it;

	for(int i=0; i < _nodes.size(); i++){

		// Map node positions to their id's
		_node_map.insert(std::pair<char, int>(_nodes.at(i).get_id().c_str(), i));
	}

	for(int i=0; i < _edges.size(); i++){

		// Link edge to nodes
		it = _node_map.find(_edges.at(i).get_in_node_id().c_str());
		if(it != _node_map.end()){
		    _nodes.at(it->second).insert_in_edge(i);
		}

		it = _node_map.find(_edges.at(i).get_out_node_id().c_str());
		if(it != _node_map.end()){
		    _nodes.at(it->second).insert_out_edge(i);
		}
	}
}

void Graph::start_parallel(){

	// Run node in different threads

}

void Graph::start_serial(){

	// Run graph in serial mode

	// Run read node
	_nodes.at(0).run();
	copy_to_buffer(0);

	// Run remaining nodes
	for(int i=1; i < _nodes.size(); i++){
		_nodes.at(i).run();
	}
}

void Graph::copy_to_buffer(int node_idx){

}

cv::Mat Graph::copy_from_buffer(int edge_idx){

}
