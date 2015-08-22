#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "train_node.h"
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

const static std::string IMAGE_PATH = "/home/lcoop22/Images/LGG";
const static std::string LOCAL_HOME = "/home/nnauata";
const static std::string fname = "/home/nnauata/LGG-test/LGG-Endothelial-Test-67-536.h5";
//const static std::string IMAGE_PATH = "/home/nelson/LGG-test";
//const static std::string LOCAL_HOME = "/home/nelson";
//const static std::string fname = "/home/nelson/LGG-test/LGG-Endothelial-small.h5";

int main (int argc, char * argv[])
{

	// Start clock
	double begin_time = utils::get_time();

	/**************************************** Get Input Data  ***************************************/

	// Define slides to use for training
	std::vector<int> train_slides = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46};


	// Declare input data
	std::vector<float> x_centroid;
	std::vector<float> y_centroid;
	std::vector<float> slide_idx;
	std::vector<int> labels;
	std::vector<std::string> slides;

	// Get input data from HDF5
	utils::get_data(fname, "x_centroid", x_centroid);
	utils::get_data(fname, "y_centroid", y_centroid);
	utils::get_data(fname, "slideIdx", slide_idx);
	utils::get_data(fname, "labels", labels);
	utils::get_data(fname, "slides", slides);

	std::cout << "Time to read images: " << float( utils::get_time() - begin_time )  << std::endl;
	/************************************ Create Train Dataset ************************************/
	
	// Declare Variables
	long long unsigned int num_elems = x_centroid.size();
	GraphNet *train_graph = new GraphNet(PARALLEL);
	std::vector<std::string> train_file_paths;
	std::vector<std::vector<std::tuple<float, float>>> train_cells_coordinates_set;
	std::string train_dataset_name = "data";
	std::vector<std::vector<int>> train_labels;

	// Create input
	for(int k = 0; k < slides.size(); k++){

		// Create data for each slide
		std::vector<std::tuple<float, float>> train_slide;
		std::vector<int> label_slide;

		// Append
		train_cells_coordinates_set.push_back(train_slide);
		train_labels.push_back(label_slide);
	}
	

	/******************************** Shuffle & Split Data ******************************************/
	
	float begin_time_2 = utils::get_time();
	utils::fill_data(num_elems, num_elems, train_cells_coordinates_set, train_labels, x_centroid, y_centroid, labels, slide_idx);

	std::cout << "Time to fill data: " << float( utils::get_time() - begin_time_2)  << std::endl;
	std::cout << "train_size: " << num_elems << std::endl;

	
	/********************************    Setup Graphs     *******************************************/
	
	//Define paths
	for(int k = 0; k < slides.size(); k++){

		std::string img_name = utils::get_image_name(slides[k], IMAGE_PATH);
		std::cout << img_name << std::endl;
		train_file_paths.push_back(IMAGE_PATH + "/" + img_name);
		std::cout << IMAGE_PATH << "/" << img_name << std::endl;
	}

	/********************************    Remove Slides   ********************************************/

	utils::remove_slides(train_file_paths, train_cells_coordinates_set, train_labels, train_slides);

	/************************************************************************************************/

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Train Nodes
	for(int i=0; i < NUMB_READ_NODE; i++){

		train_graph->add_node(new ReadNode("read_node" + std::to_string(i), train_file_paths, train_cells_coordinates_set, train_labels, ALTERNATE_MODE));
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

				train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_node" + std::to_string(l), "grayscale_node" + std::to_string(i)));
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
