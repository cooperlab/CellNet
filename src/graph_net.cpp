#include "graph_net.h"
#include <iostream>

GraphNet::GraphNet(): _nodes(), _edges(), _node_map(){}

void GraphNet::run(){

	std::cout << "Linking graph..." << std::endl;
	link();

	std::cout << "Executing in serial mode..." << std::endl;
	start_serial();
}

void GraphNet::link(){
	std::map<std::string, int>::iterator it;

	for(std::vector<Node*>::size_type i=0; i < _nodes.size(); i++){

		// Map node positions to their id's
		_node_map.insert(std::pair<std::string, int>(_nodes.at(i)->get_id(), i));
	}

	for(std::vector<Edge>::size_type i=0; i < _edges.size(); i++){

		// Link edge to nodes
		it = _node_map.find(_edges.at(i).get_in_node_id());
		if(it != _node_map.end()){
		    _nodes.at(it->second)->insert_out_edge(i);
		}

		it = _node_map.find(_edges.at(i).get_out_node_id());
		if(it != _node_map.end()){
		    _nodes.at(it->second)->insert_in_edge(i);
		}
	}

	/************************* Debug *******************************/
	// Print Graph 
	/*for(std::vector<Node>::size_type i=0; i < _nodes.size(); i++){

		it = _node_map.find(_nodes.at(i).get_id());
		if(it != _node_map.end()){
		    std::cout << _nodes.at(i).get_id() << " " << it ->second<< std::endl;
		}
	}*/
	
	// Print all connections
	/*for(std::vector<Node>::size_type i=0; i < _nodes.size(); i++){

		std::cout << _nodes.at(i).get_id() << std::endl;

		if(!_nodes.at(i).get_in_edges_ids().empty()){
			std::cout << "in: "<< std::to_string(_nodes.at(i).get_in_edges_ids().at(0)) << std::endl;
		}
		else{
			std::cout << "in: none" << std::endl;
		}
		if(!_nodes.at(i).get_out_edges_ids().empty()){
			std::cout << "out: 	"<< std::to_string(_nodes.at(i).get_out_edges_ids().at(0)) << std::endl;
		}
		else{
			std::cout << "out: none" << std::endl;
		}
	}*/
}

void GraphNet::start_parallel(){

	// Run node in different threads

}

void GraphNet::start_serial(){
	/* Run graph in serial mode */
	int edge_idx;
	std::cout << " ==> Executing Stage "<< std::to_string(0) << std::endl;
	
	// Run read node
	_nodes.at(0)->run();
	copy_to_buffer(0);

	// Run remaining nodes
	for(std::vector<Node*>::size_type i=1; i < _nodes.size(); i++){

		std::cout << " ==> Executing Stage "<< std::to_string(i) << std::endl;
		edge_idx = _nodes.at(i)->get_in_edges_ids().at(0);

		_nodes.at(i)->set_target(copy_from_buffer(edge_idx));
		_nodes.at(i)->run();
		copy_to_buffer(i);
	}
	std::cout << "*Execution complete*" << std::endl;
}

void GraphNet::copy_to_buffer(int node_idx){
	
	std::vector<cv::Mat> out;
	_nodes.at(node_idx)->get_output(out);

	std::vector<int> edges_ids = _nodes.at(node_idx)->get_out_edges_ids();
	std::vector<cv::Mat> curr_buffer;

	for(std::vector<int>::size_type i=0; i < edges_ids.size(); i++){

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

cv::Mat GraphNet::copy_from_buffer(int edge_idx){
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

void GraphNet::add_node(Node *node){
	_nodes.push_back(node);
}

void GraphNet::add_edge(Edge edge){
	_edges.push_back(edge);	
}