#include "graph_net.h"
//#include "read_node.h"
#include "read_jpg_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "prediction_node.h"
#include "augmentation_node.h"
#include "edge.h"
#include "hdf5.h"
#include "utils.h"
#include <tuple>
#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <boost/thread.hpp>
#include <boost/ptr_container/ptr_deque.hpp>
#define REPEAT_MODE  0
#define ALTERNATE_MODE  1
#define CHUNK_MODE 2
#define NUMB_GRAYSCALE_NODE 1
#define NUMB_LAPLACIAN_NODE 0
#define NUMB_WRITE_PIPE_NODE 1
#define SERIAL 0
#define PARALLEL 1


int main (int argc, char * argv[])
{

	if( argc != 6 ) {
		std::cerr << "Usage: " << argv[0] << " <dataset.h5> <trained model> <net model> <image path> <out file>" << std::endl;
		exit(-1);
	}

 
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
	GraphNet *prediction_graph = new GraphNet(PARALLEL);

	/********************************    Remove Slides   ********************************************/
	utils::remove_slides(slides, prediction_slides);

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some prediction_Nodes
	prediction_graph->add_node(new ReadJPGNode("read_node", slides, CHUNK_MODE, imagePath));

	// Define grayscale nodes
	prediction_graph->add_node(new GrayScaleNode("grayscale_node", REPEAT_MODE));

	// Prediction Node
	prediction_graph->add_node(new PredictionNode("prediction_node", REPEAT_MODE, batch_size, netModelFilename, trainedFilename, gpu_id, outFilename));

		
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
