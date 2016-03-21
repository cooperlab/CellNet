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
#include <iostream>
#include <ctime>
#include <dirent.h>

#include "graph_net.h"
#include "train-cmd.h"



using namespace std;



int main (int argc, char * argv[])
{
	gengetopt_args_info		args;

	if( cmdline_parser(argc, argv, &args) != 0 ) {
		exit(-1);
	}

	FLAGS_alsologtostderr = 1;
	// Need to fake no args to caffe
	int caffeArgc = 1;
 	caffe::GlobalInit(&caffeArgc, &argv);

	// Start clock
	double begin_time = utils::get_time();

	// Declare Variables
	GraphNet	*train_graph = new GraphNet(GraphNet::Parallel);


	// Define Graphs
	cout << "Defining graph nodes..." << endl;

	if( args.deconv_img_arg > 0 ) {
		cout << "Using " << args.deconv_img_arg << " channel deconvoluted images" << endl;
	}

	
	int	transferSize = 1000;

	// The ReadHDF5Node object takes a list of files. For training we only
	// use one file, so we just put in into a vector.
	//
	vector<string>	files;
	files.push_back(args.training_set_arg);

	train_graph->add_node(new ReadHDF5Node("read_node", files, Node::Repeat, args.deconv_img_arg, true));
	if( args.grayscale_flag ) {
		train_graph->add_node(new GrayScaleNode("grayscale_node", transferSize, Node::Repeat));
	}
	train_graph->add_node(new AugmentationNode("augmentation_node", transferSize, Node::Repeat, 
												args.aug_factor_arg));

	int batch_size = args.batch_size_arg;
	float momentum = 0.9;
	float gamma = 0.0005;
	float base_lr = 0.0001;
	train_graph->add_node(new TrainNode("train_node", 
										Node::Repeat, 
										batch_size, 
										args.gpu_dev_arg, 
										args.params_arg, 
										base_lr, 
										momentum, 
										gamma, 
										100,
										args.output_arg));	

	std::cout << "Defining edges" << std::endl;
	// Add edges
	int n_edges = 0;

	if( args.grayscale_flag ) {
		train_graph->add_edge(new Edge("edge" + to_string(n_edges++), "read_node", "grayscale_node"));
		train_graph->add_edge(new Edge("edge" + to_string(n_edges++), "grayscale_node", "augmentation_node"));
	} else {
		train_graph->add_edge(new Edge("edge" + to_string(n_edges++), "read_node", "augmentation_node"));
	}
	train_graph->add_edge(new Edge("edge" + to_string(n_edges++), "augmentation_node", "train_node"));

	std::cout << "*Graph defined*" << std::endl;

	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs in parallel
	boost::thread_group threads;
	threads.create_thread(boost::bind(&GraphNet::run, boost::ref(train_graph)));
	threads.join_all();

	/*********************************************    Clean   ***************************************************/
	
	// Stop clock
	cout << "Elapsed Time: " << double( utils::get_time() - begin_time ) << endl;
	
	// Release memory
	delete train_graph;
	
	return 0;
}
