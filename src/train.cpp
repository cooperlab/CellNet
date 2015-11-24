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
#include <string>
#include <iostream>
#include <ctime>
#include <dirent.h>

#include "graph_net.h"


#define GPU_ID 0


using namespace std;



int main (int argc, char * argv[])
{

	if( argc != 4 ) {
		cerr << "Usage: " << argv[0] << " <training set> <cnn proto> <out file>" << endl;
		exit(-1);
	}
	FLAGS_alsologtostderr = 1;
 	caffe::GlobalInit(&argc, &argv);

	string	fname = argv[1],
			netModel = argv[2],
			outFilename = argv[3];

	// Start clock
	double begin_time = utils::get_time();

	// Declare Variables
	GraphNet *train_graph = new GraphNet(GraphNet::Parallel);
	std::string train_dataset_name = "data";

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// The ReadHDF5Node object takes a list of files. For training we only
	// use one file, so we just put in into a vector.
	//
	vector<string>	filenames;
	filenames.push_back(fname);

	// Add some Train Nodes
	train_graph->add_node(new ReadHDF5Node("read_node", 
										  filenames, 
										  Node::Repeat, 
										  true) );

	train_graph->add_node(new DebugNode("debug_node", Node::Repeat));

	// Define grayscale nodes
//	train_graph->add_node(new GrayScaleNode("grayscale_node", Node::Alternate));
	

//	int batch_size = 16;
//	float momentum = 0.9;
//	float gamma = 0.0005;
//	float base_lr = 0.0001;
//	train_graph->add_node(new TrainNode("train_node", 
//										Node::Repeat, 
//										batch_size, 
//										GPU_ID, 
//										netModel, 
//										base_lr, 
//										momentum, 
//										gamma, 
//										100,
//										outFilename));

	// Add train edges
	int n_edges = 0;

	train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), 
								   "read_node", 
								   "debug_node"));

//	train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), 
//								   "grayscale_node", 
//								   "train_node"));

	std::cout << "*Graph defined*" << std::endl;

	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs in parallel
	boost::thread_group threads;
	threads.create_thread(boost::bind(&GraphNet::run, boost::ref(train_graph)));
	threads.join_all();

	/*********************************************    Clean   ***************************************************/
	
	// Stop clock
	std::cout << "Elapsed Time: " << double( utils::get_time() - begin_time )  << std::endl;
	
	// Release memory
	delete train_graph;
	
	return 0;
}
