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
	/* Run graph in serial mode */
	int edge_idx;

	// Run read node
	_nodes.at(0).run();
	copy_to_buffer(0);

	// Run remaining nodes
	for(int i=1; i < _nodes.size(); i++){
		
		edge_i = _nodes.at(i).get_in_edges_ids().at(0);
		_nodes.at(i).set_target(copy_from_buffer(edge_idx));
		_nodes.at(i).run();
	}
}

void Graph::copy_to_buffer(int node_idx){
	std::vector<cv::Mat> out = _nodes.at(node_idx).get_output();
	std::vector<int> edges_ids = _nodes.at(node_idx).get_out_edges_ids();
	std::vector<cv::Mat> curr_buffer;

	for(int i=0; i < edges_ids.size(); i++){

		// Get current buffer
		curr_buffer = _edges.at(edges_ids.at(i)).get_buffer();
		
		// Concatenate buffers
		std::vector<cv::Mat> new_buffer;
		new_buffer.reserve(curr_buffer.size() + out.size());
		new_buffer.insert( new_buffer.end(), curr_buffer.begin(), curr_buffer.end());
		new_buffer.insert( new_buffer.end(), out.begin(), out.end());

		// Set new buffer
		_edges.at(edges_ids.at(i)).set_buffer(new_buffer);
	}
}

cv::Mat Graph::copy_from_buffer(int edge_idx){
	std::vector<cv::Mat> buffer;
	cv::Mat out;

	// Get buffer
	buffer = _edges.at(edge_idx).get_buffer();

	// Remove first element from buffer
	if(!buffer.empty()){
		out = buffer.at(0);
		buffer.erase(buffer.begin());
	}

	// Set new buffer and return first element
	_edges.at(edge_idx).set_buffer(buffer);
	return out;
}
