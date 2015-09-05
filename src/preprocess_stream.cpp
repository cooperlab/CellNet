#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "read_jpg_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "prediction_node.h"
#include "write_pipe_node.h"
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
#define NUMB_LAPLACIAN_NODE 1
#define NUMB_WRITE_PIPE_NODE 1
#define SERIAL 0
#define PARALLEL 1

const static std::string IMAGE_PATH = "/home/lcoop22/Images/LGG";
//const static std::string LOCAL_HOME = "/home/nnauata";
//const static std::string fname = "/home/nnauata/LGG-test/LGG-features-2.h5";
//const static std::string IMAGE_PATH = "/home/nnauata/LGG-test";
const static std::string LOCAL_HOME = "/home/nnauata";
const static std::string fname = "/home/mnalisn/testsets/LGG-Endothelial-combined-fixed.h5";

int main (int argc, char * argv[])
{

	std::vector<int> prediction_slides;
	std::string slides_str(argv[1]);
	std::string word;
    std::stringstream stream(slides_str);

    // Define slides to use for training
    while( getline(stream, word, ',') ){
    	prediction_slides.push_back(atoi(word.c_str()));
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
	std::string prediction_dataset_name = "data";
	/********************************    Remove Slides   ********************************************/

	utils::remove_slides(slides, prediction_slides);

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some prediction_Nodes
	prediction_graph->add_node(new ReadJPGNode("read_node", slides, CHUNK_MODE));

	// Define grayscale nodes
	for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
		prediction_graph->add_node(new GrayScaleNode("grayscale_node" + std::to_string(i), REPEAT_MODE));
	}

	// Define laplacian nodes
	for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
		for(int j = 0; j < NUMB_LAPLACIAN_NODE; j++){

			prediction_graph->add_node(new LaplacianPyramidNode("laplacian_node" + std::to_string(i)+std::to_string(j), REPEAT_MODE));
		}
	}

	// Define prediction nodes
	std::string prediction_d_model_path = LOCAL_HOME + "/CellNet/app/cell_net.caffemodel";
	std::string test_model_path = LOCAL_HOME + "/CellNet/online_caffe_model/cnn_test.prototxt";
	std::string model_path = LOCAL_HOME + "/CellNet/online_caffe_model/cnn_prediction_val.prototxt";
	int num_pipe = 0;
	
	for(int i=0; i < NUMB_WRITE_PIPE_NODE; i++){

		prediction_graph->add_node(new WritePipeNode("write_pipe_node" + std::to_string(i), LOCAL_HOME + "/CellNet/app/pipe" + std::to_string(num_pipe++)));
	}
	
	std::cout << "Defining edges" << std::endl;
	// Add edges
	int n_edges = 0;
	int n_w =0;
	for(int k=0; k < 1; k++){

		for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
			
			prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_node", "grayscale_node" + std::to_string(i)));
			for(int j=0; j < NUMB_LAPLACIAN_NODE; j++){

				prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node" + std::to_string(i), "laplacian_node" + std::to_string(i)+std::to_string(j)));
				prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "laplacian_node" + std::to_string(i)+std::to_string(j), "write_pipe_node" + std::to_string(n_w++)));
			}
		}
	}
	std::cout << "*Graph defined*" << std::endl;
	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs
	prediction_graph->run();

	/*********************************************    Clean   ***************************************************/
	
	// Release memory
	delete prediction_graph;

	return 0;
}
