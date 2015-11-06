#include <string>
#include <iostream>
#include <ctime>
#include <dirent.h>

#include "graph_net.h"


#define GPU_ID 0


using namespace std;



int main (int argc, char * argv[])
{

	if( argc != 5 ) {
		cerr << "Usage: " << argv[0] << " <dataset> <cnn proto> <image data> <out file>" << endl;
		exit(-1);
	}
	FLAGS_alsologtostderr = 1;
 	caffe::GlobalInit(&argc, &argv);

	
	vector<int> train_slides;
	string	fname = argv[1],
			netModel = argv[2],
			imageDir = argv[3],
			outFilename = argv[4];

	vector<std::string> slideNames;
	utils::get_data(fname, "slides", slideNames);
	for(int i = 0; i < slideNames.size(); i++) {
		train_slides.push_back(i);
	}


	// Start clock
	double begin_time = utils::get_time();

	/**************************************** Get Input Data  ***************************************/
	// Declare input data
	std::vector<std::string> slides;

	// Get input data from HDF5
	utils::get_data(fname, "slides", slides);

	std::cout << "Time to read images: " << float( utils::get_time() - begin_time )  << std::endl;
	/************************************ Create Train Dataset ************************************/
	
	// Declare Variables
	GraphNet *train_graph = new GraphNet(GraphNet::Parallel);
	std::string train_dataset_name = "data";

	/********************************    Remove Slides   ********************************************/

	utils::remove_slides(slides, train_slides);

	/************************************************************************************************/

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Train Nodes
	train_graph->add_node(new ReadJPGNode("read_jpg_node", 
										  slides, 
										  Node::Alternate, 
										  imageDir));

	// Define grayscale nodes
	train_graph->add_node(new GrayScaleNode("grayscale_node", Node::Alternate));
	

	int batch_size = 16;
	float momentum = 0.9;
	float gamma = 0.0005;
	float base_lr = 0.0001;
	train_graph->add_node(new TrainNode("train_node", 
										Node::Repeat, 
										batch_size, 
										GPU_ID, 
										netModel, 
										base_lr, 
										momentum, 
										gamma, 
										100,
										outFilename));

	// Add train edges
	int n_edges = 0;

	train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), 
								   "read_jpg_node", 
								   "grayscale_node"));

	train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), 
								   "grayscale_node", 
								   "train_node"));

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
