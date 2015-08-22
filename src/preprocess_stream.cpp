#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "prediction_node.h"
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
#define NUMB_LAPLACIAN_NODE 0
#define NUMB_WRITE_PIPE_NODE 1
#define NUMB_AUGMENTATION_NODE 1
#define SERIAL 0
#define PARALLEL 1

//const static std::string IMAGE_PATH = "/home/lcoop22/Images/LGG";
//const static std::string LOCAL_HOME = "/home/nnauata";
//const static std::string fname = "/home/nnauata/LGG-test/LGG-features-2.h5";
const static std::string IMAGE_PATH = "/home/nelson/LGG-test";
const static std::string LOCAL_HOME = "/home/nelson";
const static std::string fname = "/home/nelson/LGG-test/LGG-Endothelial-small.h5";

int main (int argc, char * argv[])
{
	// Start clock
	double begin_time = utils::get_time();	

	// Define slides to use for prediction
	std::vector<int> prediction_slides;
	prediction_slides.push_back(0);

	/**************************************** Get Input Data  ***************************************/

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
	//utils::get_data(fname, "labels", labels);
	// Create fake labels
	for(int k = 0; k < x_centroid.size(); k++){

		labels.push_back(0);
	}

	utils::get_data(fname, "slides", slides);

	std::cout << "Time to read HDF5: " << float( utils::get_time() - begin_time )  << std::endl;
	/************************************* Create prediction_Dataset ************************************/

	// Declare Variables
	
	long long unsigned int num_elems = x_centroid.size();
	GraphNet *prediction_graph = new GraphNet(PARALLEL);
	std::vector<std::string> prediction_file_paths;
	std::vector<std::vector<std::tuple<float, float>>> prediction_cells_coordinates_set;
	std::string prediction_dataset_name = "data";
	std::vector<std::vector<int>> prediction_labels;

	// Create input
	for(int k = 0; k < slides.size(); k++){

		// Create data for each slide
		std::vector<std::tuple<float, float>> prediction_slide;
		std::vector<int> label_slide;

		// Append
		prediction_cells_coordinates_set.push_back(prediction_slide);
		prediction_labels.push_back(label_slide);
	}

	/******************************** Shuffle & Split Data ******************************************/

	float begin_time_2 = utils::get_time();
	utils::fill_data(num_elems, num_elems, prediction_cells_coordinates_set, prediction_labels, x_centroid, y_centroid, labels, slide_idx);

	std::cout << "Time to fill data: " << float( utils::get_time() - begin_time_2)  << std::endl;
	std::cout << "prediction_size: " << num_elems << std::endl;

	/********************************    Setup Graphs     *******************************************/

	//Define paths
	for(int k = 0; k < slides.size(); k++){

		std::string img_name = utils::get_image_name(slides[k], IMAGE_PATH);
		std::cout << img_name << std::endl;
		prediction_file_paths.push_back(IMAGE_PATH + "/" + img_name);
		std::cout << IMAGE_PATH << "/" << img_name << std::endl;
	}

	/********************************    Remove Slides   ********************************************/

	utils::remove_slides(prediction_file_paths, prediction_cells_coordinates_set, prediction_labels, prediction_slides);

	/************************************************************************************************/

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some prediction_Nodes
	prediction_graph->add_node(new ReadNode("read_node", prediction_file_paths, prediction_cells_coordinates_set, prediction_labels, CHUNK_MODE));

	// Define grayscale nodes
	for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
		prediction_graph->add_node(new GrayScaleNode("grayscale_node" + std::to_string(i), REPEAT_MODE));
	}

	// Define augmentation nodes
	for(int i=0; i < NUMB_AUGMENTATION_NODE; i++){
		prediction_graph->add_node(new AugmentationNode("augmentation_node" + std::to_string(i), REPEAT_MODE, 1));
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
			prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node" + std::to_string(i), "augmentation_node" + std::to_string(i)));
			prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "augmentation_node"+ std::to_string(i), "write_pipe_node" + std::to_string(i)));

			//for(int j=0; j < NUMB_LAPLACIAN_NODE; j++){

			//	prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node" + std::to_string(i), "laplacian_node" + std::to_string(i)+std::to_string(j)));
			//	prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "laplacian_node" + std::to_string(i)+std::to_string(j), "write_pipe_node" + std::to_string(n_w++)));
			//}
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
