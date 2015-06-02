#include "graph_net.h"
#include <iostream>
#define NUM_THREADS 1

GraphNet::GraphNet(): _nodes(), _edges(), _node_map(){}

void GraphNet::run(){

	std::cout << "Linking graph..." << std::endl;
	link();

	std::cout << "Executing in parallel mode..." << std::endl;
	
	start_parallel();
	//start_serial();
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
		    _nodes.at(it->second)->insert_out_edge(&_edges.at(i));
		}

		it = _node_map.find(_edges.at(i).get_out_node_id());
		if(it != _node_map.end()){
		    _nodes.at(it->second)->insert_in_edge(&_edges.at(i));
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
	/* Run graph in parallel mode */
	boost::thread *threads_handle[_nodes.size()][NUM_THREADS];

	// For each node
	boost::thread thr(boost::bind(&Node::run, _nodes.at(0)));
	threads_handle[0][0] = &thr;

	for(std::vector<Node*>::size_type i=1; i < _nodes.size(); i++){

		// Release fixed number of threads for each node
		for(int j=0; j < NUM_THREADS; j++){
			boost::thread thr(boost::bind(&Node::run, _nodes.at(i)));
			threads_handle[i][j] = &thr; 
		}		
	}

	// Wait for every thread complete
	for(std::vector<Node*>::size_type i=0; i < _nodes.size(); i++){

		// Join each node at this point
		for(int j=0; j < NUM_THREADS; j++){

			threads_handle[i][j]->join();
		}		
	}	
}

void GraphNet::start_serial(){

	/* Run graph in serial mode */
	std::cout << " ==> Executing Stage "<< std::to_string(0) << std::endl;
	
	// Run read node
	_nodes.at(0)->run();

	// Run remaining nodes
	for(std::vector<Node*>::size_type i=1; i < _nodes.size(); i++){

		std::cout << " ==> Executing Stage "<< std::to_string(i) << std::endl;
		_nodes.at(i)->run();
	}
	std::cout << "*Execution complete*" << std::endl;
}

void GraphNet::add_node(Node *node){
	_nodes.push_back(node);
}

void GraphNet::add_edge(Edge edge){
	_edges.push_back(edge);	
}