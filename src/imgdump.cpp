//
//	Copyright (c) 2015-2016, Emory University
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
#include <string>
#include <ctime>
#include <boost/thread.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "graph_net.h"
#include "base_config.h"
#include "img-dump-cmd.h"


using namespace std;




int main (int argc, char *argv[])
{

	gengetopt_args_info		args;

	if( cmdline_parser(argc, argv, &args) != 0 ) {
		exit(-1);
	}

	// Start clock
	double begin_time = utils::get_time();	
	GraphNet *test_graph = new GraphNet(GraphNet::Parallel);
	int		channels = args.channels_arg;

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	cout << "Channels: " << channels << endl;

	// Currently we only pass 1 file on the command line. Ww will add
	// the option to pass a directory so we can glob the files and process
	// all of them.
	//
	vector<string>	files;
	files.push_back(args.dataset_arg);
	test_graph->add_node(new ReadHDF5Node("read_node", files, Node::Repeat, channels, args.tag_labels_flag == 1));
	if( args.grayscale_flag ) {
		test_graph->add_node(new GrayScaleNode("grayscale_node", 1000, Node::Repeat));
	}
	test_graph->add_node(new WriteImageNode("write_node", channels != 0, 1000, Node::Chunk));

		
	std::cout << "Defining edges" << std::endl;
	// Add edges
	int n_edges = 0;

	if( args.grayscale_flag ) {
		test_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_node", "grayscale_node"));
		test_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node", "write_node"));
	} else {
		test_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_node", "write_node"));
	}
	std::cout << "Graph defined*" << std::endl;
	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs
	double start = utils::get_time();
	test_graph->run();
	cout << "Total runtime: " << utils::get_time() - start << endl;


	/*********************************************    Clean   ***************************************************/
	
	// Release memory
	delete test_graph;

	return 0;
}
