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



int main (int argc, char * argv[])
{

	if( argc != 6 ) {
		std::cerr << "Usage: " << argv[0] << " <dataset.h5> <trained model> <net model> <image path> <out file>" << std::endl;
		exit(-1);
	}
	FLAGS_alsologtostderr = 1;
 	caffe::GlobalInit(&argc, &argv);

	std::vector<int> prediction_slides;
	std::string fname = argv[1];
	std::string trainedFilename = argv[2];
	std::string netModelFilename = argv[3];
	std::string imagePath = argv[4];
	std::string outFilename = argv[5];

	int batch_size = 10, gpu_id = 0;

	std::vector<std::string> slideNames;
	utils::get_data(fname, "slides", slideNames);
	for(int i = 0; i < slideNames.size(); i++) {
		prediction_slides.push_back(i);
	}

	// Start clock
	double begin_time = utils::get_time();	

	/**************************************** Get Input Data  ***************************************/

	// Declare input data
	std::vector<std::string> slides;

	// Get input data from HDF5
	utils::get_data(fname, "slides", slides);

	std::cout << "Time to read HDF5: " << float( utils::get_time() - begin_time )  << std::endl;
	/************************************* Create prediction_Dataset ************************************/

	// Declare Variables
	GraphNet *prediction_graph = new GraphNet(GraphNet::Parallel);

	/********************************    Remove Slides   ********************************************/
	utils::remove_slides(slides, prediction_slides);

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some prediction_Nodes
	prediction_graph->add_node(new ReadJPGNode("read_node", slides, Node::Chunk, imagePath));

	// Define grayscale nodes
	prediction_graph->add_node(new GrayScaleNode("grayscale_node", Node::Repeat));

	// Prediction Node
	prediction_graph->add_node(new PredictionNode("prediction_node", Node::Repeat, batch_size, netModelFilename, trainedFilename, gpu_id, outFilename));

		
	std::cout << "Defining edges" << std::endl;
	// Add edges
	int n_edges = 0;
			
	prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_node", "grayscale_node"));
	prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node", "prediction_node"));

	std::cout << "Preproccess Graph defined*" << std::endl;
	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs
	prediction_graph->run();

	/*********************************************    Clean   ***************************************************/
	
	// Release memory
	delete prediction_graph;

	return 0;
}
