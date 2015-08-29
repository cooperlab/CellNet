#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "train_node.h"
#include "read_jpg_node.h"
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
#include <dirent.h>
#define REPEAT_MODE  0
#define ALTERNATE_MODE  1
#define CHUNK_MODE 2
#define NUMB_GRAYSCALE_NODE 1
#define NUMB_LAPLACIAN_NODE 2
#define NUMB_READ_NODE 1
#define NUMB_AUGMENTATION_NODE 0
#define GPU_ID 0
#define SERIAL 0
#define PARALLEL 1

//const static std::string LOCAL_HOME = "/home/nnauata";
//const static std::string fname = "/home/nnauata/LGG-test/LGG-Endothelial-Test-67-536.h5";
const static std::string IMAGE_PATH = "/home/nelson/LGG-test";
const static std::string LOCAL_HOME = "/home/nelson";
const static std::string fname = "/home/nelson/LGG-test/LGG-Endothelial-small.h5";

int main (int argc, char * argv[])
{
	std::vector<int> train_slides;
	std::string slides_str(argv[1]);
	std::string word;
    std::stringstream stream(slides_str);

    // Define slides to use for training
    while( getline(stream, word, ',') ){
    	train_slides.push_back(atoi(word.c_str()));
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
	GraphNet *train_graph = new GraphNet(PARALLEL);
	std::vector<std::string> train_file_paths;
	std::string train_dataset_name = "data";

	/********************************    Setup Graphs     *******************************************/
	
	//Define paths
	for(int k = 0; k < slides.size(); k++){

		std::string img_name = utils::get_image_name(slides[k], IMAGE_PATH);
		train_file_paths.push_back(IMAGE_PATH + "/" + img_name);
	}

	/********************************    Remove Slides   ********************************************/

	utils::remove_slides(train_file_paths, train_slides);

	/************************************************************************************************/

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Train Nodes
	for(int i=0; i < NUMB_READ_NODE; i++){

		train_graph->add_node(new ReadJPGNode("read_jpg_node" + std::to_string(i), slides, ALTERNATE_MODE));
	}

	// Define grayscale nodes
	for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
		train_graph->add_node(new GrayScaleNode("grayscale_node" + std::to_string(i), ALTERNATE_MODE));
	}
	
	// Define laplacian nodes
	for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
		for(int j = 0; j < NUMB_LAPLACIAN_NODE; j++){
			train_graph->add_node(new LaplacianPyramidNode("laplacian_node" + std::to_string(i)+std::to_string(j), CHUNK_MODE));
		}
	}

	// Define augmentation nodes
	for(int i=0; i < NUMB_AUGMENTATION_NODE; i++){
		train_graph->add_node(new AugmentationNode("augmentation_node" + std::to_string(i), REPEAT_MODE, 4));
	}

	// Define train node
	std::string trained_model_path = LOCAL_HOME + "/CellNet/app/cell_net.caffemodel";
	std::string test_model_path = LOCAL_HOME + "/CellNet/online_caffe_model/cnn_test.prototxt";
	std::string model_path = LOCAL_HOME + "/CellNet/online_caffe_model/cnn_train_val.prototxt";
	int batch_size = 8;
	float momentum = 0.9;
	float gamma = 0.0005;
	float base_lr = 0.0001;
	train_graph->add_node(new TrainNode("train_node", REPEAT_MODE, batch_size, GPU_ID, model_path, base_lr, momentum, gamma, 1));

	// Add train edges
	int n_edges = 0;
	for(int k=0; k < 1; k++){

		for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
			
			for(int l = 0; l < NUMB_READ_NODE; l++){

				train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_jpg_node" + std::to_string(l), "grayscale_node" + std::to_string(i)));
				//train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node"+ std::to_string(i), "augmentation_node" + std::to_string(i)));
				//train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "augmentation_node" + std::to_string(l), "train_node"));

			}
			for(int j=0; j < NUMB_LAPLACIAN_NODE; j++){

				train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node" + std::to_string(i), "laplacian_node" + std::to_string(i)+std::to_string(j)));
				train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "laplacian_node" + std::to_string(i)+std::to_string(j), "train_node"));
			}
		}
	}
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
