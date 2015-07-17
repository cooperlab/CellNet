#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "train_node.h"
#include "prediction_node.h"
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
#define NUMB_LAPLACIAN_NODE 2

const static std::string LOCAL_HOME = "/home/nelson";

void fill_data(int N, int num_elem, std::vector<std::vector<std::tuple<float, float>>> &cells_coordinates_set, std::vector<std::vector<int>> &shuffled_labels, std::vector<float> &x_centroid, std::vector<float> &y_centroid, std::vector<int> &labels, std::vector<float> &slide_idx){
	
	// Fill train dataset
	srand (time(NULL));
	int k = 0;
	for(int i=0; i < N; i++){

		// Get random sample
		k = rand() % num_elem;
		
		// Adjust label value to 0 or 1
		if(labels[k] == -1){
			labels[k] = 0;
		}

		// Append 
		cells_coordinates_set[slide_idx[k]].push_back(std::make_tuple(x_centroid[k], y_centroid[k]));
		shuffled_labels[slide_idx[k]].push_back(labels[k]);

		// Erase selected element
		x_centroid.erase(x_centroid.begin() + k);
		y_centroid.erase(y_centroid.begin() + k);
		slide_idx.erase(slide_idx.begin() + k);
		labels.erase(labels.begin() + k);
		num_elem--;
	}
}


int main (int argc, char * argv[])
{

	// Start clock
	double begin_time = utils::get_time();
	/**************************************** Get Input Data  ***************************************/

	// Declare input data
	std::vector<float> x_centroid;
	std::vector<float> y_centroid;
	std::vector<float> slide_idx;
	std::vector<int> labels;
	std::vector<std::string> slides;

	// Get input data from HDF5
	std::string fname = LOCAL_HOME + "/LGG-test/LGG-Endothelial-2-test.h5";
	utils::get_data(fname, "x_centroid", x_centroid);
	utils::get_data(fname, "y_centroid", y_centroid);
	utils::get_data(fname, "slideIdx", slide_idx);
	utils::get_data(fname, "labels", labels);
	utils::get_data(fname, "slides", slides);

	std::cout << "Time to read HDF5: " << float( utils::get_time() - begin_time )  << std::endl;

	/************************************* Create Train Dataset ************************************/

	// Declare Variables
	long long unsigned int num_elems = x_centroid.size();
	GraphNet *train_graph = new GraphNet();
	std::vector<std::string> train_file_paths;
	std::vector<std::vector<std::tuple<float, float>>> train_cells_coordinates_set;
	std::string train_dataset_name = "data";
	std::vector<std::vector<int>> train_labels;

	// Create input
	for(int k = 0; k < slides.size(); k++){

		std::vector<std::tuple<float, float>> train_slide;
		train_cells_coordinates_set.push_back(train_slide);
	}

	/******************************** Shuffle & Split Data ******************************************/

	float begin_time_2 = utils::get_time();
	fill_data(num_elems, num_elems, train_cells_coordinates_set, train_labels, x_centroid, y_centroid, labels, slide_idx);

	std::cout << "Time to fill data: " << float( utils::get_time() - begin_time_2)  << std::endl;
	std::cout << "train_size: " << train_cells_coordinates_set[0].size() << std::endl;
	std::cout << "x_centroid_size: " << x_centroid.size() << std::endl;
	std::cout << "y_centroid_size: " << y_centroid.size() << std::endl;

	/********************************    Setup Graphs     *******************************************/

	//Define paths
	for(int k = 0; k < slides.size(); k++){

		train_file_paths.push_back(LOCAL_HOME + "/" + slides[k] + "*.svs");
	}

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Train Nodes
	train_graph->add_node(new ReadNode("read_node", train_file_paths, train_cells_coordinates_set, train_labels, CHUNK_MODE));

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

	// Define train node
	std::string trained_model_path = LOCAL_HOME + "/CellNet/app/cell_net.caffemodel";
	std::string test_model_path = LOCAL_HOME + "/CellNet/online_caffe_model/cnn_test.prototxt";
	std::string model_path = LOCAL_HOME + "/CellNet/online_caffe_model/cnn_train_val.prototxt";
	int batch_size = 8;
	train_graph->add_node(new PredictionNode("prediction_node", REPEAT_MODE, batch_size, test_model_path, trained_model_path));

	// Add train edges
	int n_edges = 0;
	for(int k=0; k < 1; k++){

		for(int i=0; i < NUMB_GRAYSCALE_NODE; i++){
			
			train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_node", "grayscale_node" + std::to_string(i)));
			for(int j=0; j < NUMB_LAPLACIAN_NODE; j++){

				train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "grayscale_node" + std::to_string(i), "laplacian_node" + std::to_string(i)+std::to_string(j)));
				train_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "laplacian_node" + std::to_string(i)+std::to_string(j), "prediction_node"));
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
	return 0;
}
