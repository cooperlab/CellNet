#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "train_node.h"
#include "prediction_pipe_node.h"
#include "read_pipe_node.h"
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
#define SERIAL 0
#define PARALLEL 1

//const static std::string LOCAL_HOME = "/home/nnauata";
const static std::string LOCAL_HOME = "/home/nelson";

int main (int argc, char * argv[3])
{

	// Store parameters
	int gpu_id;
	int batch_size;

	sscanf(argv[1],"%d",&gpu_id);
	sscanf(argv[2],"%d",&batch_size);

	std::cout << "Running on device " << std::to_string(gpu_id) << std::endl;
	std::cout << "Batch size: " << std::to_string(batch_size) << std::endl;

	/********************************    Setup Graphs     *******************************************/
	// Define Graphs
	GraphNet *prediction_graph = new GraphNet(SERIAL);

	// Define prediction nodes
	std::string pipe_name = LOCAL_HOME + "/CellNet/app/pipe"+std::to_string(gpu_id);
	std::string trained_model_path = LOCAL_HOME + "/CellNet/app/cell_net.caffemodel";
	std::string test_model_path = LOCAL_HOME + "/CellNet/online_caffe_model/cnn_test.prototxt";
	std::string out_file = LOCAL_HOME + "/CellNet/app/predictions.txt";

	prediction_graph->add_node(new PredictionPipeNode("prediction_node", REPEAT_MODE, batch_size, test_model_path, trained_model_path, gpu_id, pipe_name, out_file));
	
	// Add edges
	int n_edges = 0;
	prediction_graph->add_edge(new Edge("edge" + std::to_string(n_edges++), "read_pipe_node", "prediction_node"));
	/********************************************* Run Graphs ***************************************************/
	
	// Run graph
	prediction_graph->run();

	/*********************************************    Clean   ***************************************************/
	
	// Release memory
	delete prediction_graph;

	return 0;
}
