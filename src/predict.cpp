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
#include <ctime>
#include <boost/thread.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "graph_net.h"
#include "base_config.h"



using namespace std;





int main (int argc, char * argv[])
{

	if( argc != 5 ) {
		cerr << "Usage: " << argv[0] << " <dataset.h5> <trained model> <net model> <out file>" << endl;
		exit(-1);
	}
	FLAGS_alsologtostderr = 1;
 	caffe::GlobalInit(&argc, &argv);


	string trainedFilename = argv[2];
	string netModelFilename = argv[3];
	string outFilename = argv[4];


	int batch_size = 10, gpu_id = 0;


	// Start clock
	double begin_time = utils::get_time();	
	GraphNet *prediction_graph = new GraphNet(GraphNet::Parallel);

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;


	// Currently we only pass 1 file on the command line. Ww will add
	// the option to pass a directory so we can glob the files and process
	// all of them.
	//
	vector<string>	files;
	files.push_back(argv[1]);
	prediction_graph->add_node(new ReadHDF5Node("read_node", files, Node::Repeat, false));
	prediction_graph->add_node(new GrayScaleNode("grayscale_node", Node::Repeat));
	prediction_graph->add_node(new AugmentationNode("augmentation_node", Node::Chunk, 3));

	size_t 	pos;
	string 	baseName, extension;

	pos = outFilename.find_last_of(".");
	if( pos != string::npos ) {
		baseName = outFilename.substr(0, pos);
		extension = outFilename.substr(pos + 1);
	} else {
		baseName = outFilename;
		extension = ".txt";
	}

	for(int i = 0; i < 4; i++ ) {
		prediction_graph->add_node(new PredictionNode("prediction_node" + to_string(i), 
													  Node::Repeat, 
													  batch_size, 
													  netModelFilename, 
													  trainedFilename, i, 
													  baseName + "_" + to_string(i) + extension));
	}
		
	std::cout << "Defining edges" << std::endl;
	// Add edges
	int n_edges = 0;

	prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_node", "grayscale_node"));
	prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node", "augmentation_node"));

	for(int i = 0; i < 4; i++) {
		prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), 
											"augmentation_node", 
											"prediction_node" + to_string(i)));
	}
		

	std::cout << "Preproccess Graph defined*" << std::endl;
	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs
	double start = utils::get_time();
	prediction_graph->run();
	cout << "Total runtime: " << utils::get_time() - start << endl;


	/*********************************************    Clean   ***************************************************/
	
	// Release memory
	delete prediction_graph;

	return 0;
}
