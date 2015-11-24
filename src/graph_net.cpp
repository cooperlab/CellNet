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
#include <iostream>

#include "graph_net.h"
#include "base_config.h"




/* Define the number of threads running for each node
NOTE: At some nodes (e.g writing nodes), the graph considers only one thread for each node.
To include more threads working at the same node, we would have to control the access at some nodes (e.g writing nodes),
Also, changing the number of threads per node does not guarantee the order of the outputs. */
#define NUM_THREADS 1




GraphNet::GraphNet(int mode) : 
_mode(mode), 
_nodes(), 
_edges(), 
_node_map()
{

}





void *GraphNet::run()
{

	cout << "Linking graph..." << endl;
	link();
	
	if( _mode == 0 ) {
		cout << "Executing in serial mode..." << endl;
		start_serial();
	} else {
		cout << "Executing in parallel mode..." << endl;
		start_parallel();
	}
}





void GraphNet::link()
{
	map<string, int>::iterator it;

	for( boost::ptr_deque<Node>::size_type i=0; i < _nodes.size(); i++ ) {

		// Map node positions to their id's
		_node_map.insert(pair<std::string, int>(_nodes.at(i).get_id(), i));
	}

	for( boost::ptr_deque<Edge>::size_type i=0; i < _edges.size(); i++ ){

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
#if 1
	// Print Graph 
	for(vector<Node>::size_type i=0; i < _nodes.size(); i++) {

		it = _node_map.find(_nodes.at(i).get_id());
		if(it != _node_map.end()){
		    cout << _nodes.at(i).get_id() << " " << it->second << endl;
		}
	}
		
	// Print all connections
	for( vector<Node>::size_type i=0; i < _nodes.size(); i++ ) {

		cout << _nodes.at(i).get_id() << endl;

		if( !_nodes.at(i)._in_edges.empty() ) {
			
			for(int k = 0; k < _nodes.at(i)._in_edges.size(); k++){
				cout << "in: "<< _nodes.at(i)._in_edges.at(k)->_id << endl;
			}
		} else {
			cout << "in: none" << endl;
		}

		if( !_nodes.at(i)._out_edges.empty() ) {
			for(int k = 0; k < _nodes.at(i)._out_edges.size(); k++) {
				cout << "out: "<< _nodes.at(i)._out_edges.at(k)->_id << endl;
			}
		} else {
			cout << "out: none" << endl;
		}
	}
#endif
}






void GraphNet::start_parallel()
{
	/* Run graph in parallel mode */
	boost::thread_group threads;

	// For each node
	for( vector<Node*>::size_type i=0; i < _nodes.size(); i++ ) {

		// Release fixed number of threads for each node
		for(int j=0; j < NUM_THREADS; j++){
			threads.create_thread(boost::bind(&Node::run, boost::ref(_nodes.at(i))));
		}		
	}
	cout << "All nodes started" << endl;
	threads.join_all();
}





void GraphNet::start_serial()
{

	/* Run graph in serial mode */
	cout << " ==> Executing Stage "<< std::to_string(0) << endl;
	
	// Run read node
	_nodes.at(0).run();

	// Run remaining nodes
	for(vector<Node*>::size_type i=1; i < _nodes.size(); i++){

		cout << " ==> Executing Stage "<< std::to_string(i) << endl;
		_nodes.at(i).run();
	}
	cout << "*Execution complete*" << endl;
}





void GraphNet::add_node(Node *node)
{
	_nodes.push_back(node);
}





void GraphNet::add_edge(Edge *edge)
{
	cout << "Adding edge: " << edge->get_in_node_id() << " -> " << edge->get_out_node_id() << endl;

	_edges.push_back(edge);	
}
