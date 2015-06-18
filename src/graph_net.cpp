#include "graph_net.h"
#include <iostream>
/* Define the number of threads running for each node
NOTE: At some nodes (e.g writing nodes), the graph considers only one thread for each node.
To include more threads working at the same node, we would have to control the access at some nodes (e.g writing nodes),
Also, changing the number of threads per node does not guarantee the order of the outputs. */
#define NUM_THREADS 1

GraphNet::GraphNet(): _nodes(), _edges(), _node_map(){}

void *GraphNet::run(){

	std::cout << "Linking graph..." << std::endl;
	link();

	std::cout << "Executing in parallel mode..." << std::endl;
	
	start_parallel();
	//start_serial();
}

void GraphNet::link(){
	std::map<std::string, int>::iterator it;

	for(boost::ptr_deque<Node>::size_type i=0; i < _nodes.size(); i++){

		// Map node positions to their id's
		_node_map.insert(std::pair<std::string, int>(_nodes.at(i)._id, i));
	}

	for(boost::ptr_deque<Edge>::size_type i=0; i < _edges.size(); i++){

		// Link edge to nodes
		it = _node_map.find(_edges.at(i).get_in_node_id());
		if(it != _node_map.end()){
		    _nodes.at(it->second).insert_out_edge(&_edges.at(i));
		}	

		it = _node_map.find(_edges.at(i).get_out_node_id());
		if(it != _node_map.end()){
		    _nodes.at(it->second).insert_in_edge(&_edges.at(i));
		}
	}

	/************************* Debug *******************************/

}

void GraphNet::start_parallel(){
	/* Run graph in parallel mode */
	boost::thread_group threads;

	// For each node
	for(std::vector<Node*>::size_type i=0; i < _nodes.size(); i++){

		// Release fixed number of threads for each node
		for(int j=0; j < NUM_THREADS; j++){
			threads.create_thread(boost::bind(&Node::run, boost::ref(_nodes.at(i))));
		}		
	}
	threads.join_all();
}

void GraphNet::start_serial(){

	/* Run graph in serial mode */
	std::cout << " ==> Executing Stage "<< std::to_string(0) << std::endl;
	
	// Run read node
	_nodes.at(0).run();

	// Run remaining nodes
	for(std::vector<Node*>::size_type i=1; i < _nodes.size(); i++){

		std::cout << " ==> Executing Stage "<< std::to_string(i) << std::endl;
		_nodes.at(i).run();
	}
	std::cout << "*Execution complete*" << std::endl;
}

void GraphNet::add_node(Node *node){
	_nodes.push_back(node);
}

void GraphNet::add_edge(Edge *edge){
	_edges.push_back(edge);	
}