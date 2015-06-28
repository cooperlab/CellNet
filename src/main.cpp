#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
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

const static std::string LOCAL_HOME = "/home/nnauata";

void fill_data(int N, int num_elem, std::vector<std::vector<std::tuple<double, double>>> &cells_coordinates_set, std::vector<double> &x_centroid, std::vector<double> &y_centroid, std::vector<double> &slide_idx){

	// Fill train dataset
	srand (time(NULL));
	int k = 0;
	for(int i=0; i < N; i++){

		k = rand() % num_elem;
		if(!slide_idx[k]){
			cells_coordinates_set[0].push_back(std::make_tuple(x_centroid[k], y_centroid[k]));
		}
		else{
			cells_coordinates_set[1].push_back(std::make_tuple(x_centroid[k], y_centroid[k]));
		}
		x_centroid.erase(x_centroid.begin() + k);
		y_centroid.erase(y_centroid.begin() + k);
		slide_idx.erase(slide_idx.begin() + k);
		num_elem--;
	}
}


int main (int argc, char * argv[])
{

	// Start clock
	double begin_time = utils::get_time();
	/**************************************** Get Input Data  ***************************************/

	// Declare input data
	std::vector<double> x_centroid;
	std::vector<double> y_centroid;
	std::vector<double> slide_idx;
	std::vector<double> labels;
	
	// Get input data from HDF5
	utils::get_data(LOCAL_HOME + "/LGG-test/LGG-Endothelial-2-test.h5", "x_centroid", x_centroid);
	utils::get_data(LOCAL_HOME + "/LGG-test/LGG-Endothelial-2-test.h5", "y_centroid", y_centroid);
	utils::get_data(LOCAL_HOME + "/LGG-test/LGG-Endothelial-2-test.h5", "slideIdx", slide_idx);

	std::cout << "Time to read HDF5: " << float( utils::get_time() - begin_time )  << std::endl;

	/************************************* Create Train Dataset ************************************/

	// Declare Variables
	long long unsigned int num_elems = x_centroid.size();
	GraphNet *train_graph = new GraphNet();
	std::vector<std::string> train_file_paths;
	std::vector<std::vector<std::tuple<double, double>>> train_cells_coordinates_set;
	std::vector<hsize_t> train_dim1 = {num_elems, 3, 50, 50};
	std::vector<hsize_t> train_dim2 = {num_elems, 1, 50, 50};
	std::vector<hsize_t> train_dim3 = {num_elems, 1, 4, 50, 50};
	std::string train_dataset_name = "data";
	std::vector<double> train_labels;

	// Create input
	std::vector<std::tuple<double, double>> train_slide1;
	std::vector<std::tuple<double, double>> train_slide2;
	train_cells_coordinates_set.push_back(train_slide1);
	train_cells_coordinates_set.push_back(train_slide2);

	/******************************** Shuffle & Split Data ******************************************/

	double begin_time_2 = utils::get_time();
	fill_data(num_elems, num_elems, train_cells_coordinates_set, x_centroid, y_centroid, slide_idx);
	std::cout << "Time to fill data: " << float( utils::get_time() - begin_time_2)  << std::endl;

	std::cout << "train_size: " << train_cells_coordinates_set[0].size() << std::endl;
	std::cout << "x_centroid_size: " << x_centroid.size() << std::endl;
	std::cout << "y_centroid_size: " << y_centroid.size() << std::endl;

	/********************************    Setup Graphs     *******************************************/

	//Define paths
	train_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-EZ-7264-01Z-00-DX1.80a61d74-77d9-4998-bb55-213767a588ff.svs");
	train_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-HT-7474-01Z-00-DX1.B3E88862-6C35-4E30-B374-A7BC80231B8C.svs");

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Train Nodes
	ReadNode *train_read_node = new ReadNode("read_node", train_file_paths, train_cells_coordinates_set, ALTERNATE_MODE);

	GrayScaleNode *train_grayscale_node1 = new GrayScaleNode("grayscale_node1", ALTERNATE_MODE);
	GrayScaleNode *train_grayscale_node2 = new GrayScaleNode("grayscale_node2", ALTERNATE_MODE);

	LaplacianPyramidNode *train_laplacian_node1 = new LaplacianPyramidNode("laplacian_node1", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node2 = new LaplacianPyramidNode("laplacian_node2", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node3 = new LaplacianPyramidNode("laplacian_node3", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node4 = new LaplacianPyramidNode("laplacian_node4", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node5 = new LaplacianPyramidNode("laplacian_node5", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node6 = new LaplacianPyramidNode("laplacian_node6", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node7 = new LaplacianPyramidNode("laplacian_node7", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node8 = new LaplacianPyramidNode("laplacian_node8", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node9 = new LaplacianPyramidNode("laplacian_node9", REPEAT_MODE);
	LaplacianPyramidNode *train_laplacian_node10 = new LaplacianPyramidNode("laplacian_node10", REPEAT_MODE);

	//WritePNGNode *train_write_png_node1 = new WritePNGNode("write_png_node1", LOCAL_HOME + "/CellNet/train/data_gray/");
	//WritePNGNode *train_write_png_node2 = new WritePNGNode("write_png_node2", LOCAL_HOME + "/CellNet/train/data_rgb/");
	//WriteHDF5Node *train_write_hdf5_node1 = new WriteHDF5Node("write_hdf5_node1", LOCAL_HOME + "/CellNet/train/data_hdf5/rgb_train", train_dim1, train_dataset_name, train_labels);
	//WriteHDF5Node *train_write_hdf5_node2 = new WriteHDF5Node("write_hdf5_node2", LOCAL_HOME + "/CellNet/train/data_hdf5/gray_train", train_dim2, train_dataset_name, train_labels);
	//WriteHDF5Node *train_write_hdf5_node3 = new WriteHDF5Node("write_hdf5_node3", LOCAL_HOME + "/CellNet/train/data_hdf5/lap_train", train_dim3, train_dataset_name, train_labels);

	train_graph->add_node(train_read_node);
	
	train_graph->add_node(train_grayscale_node1);
	train_graph->add_node(train_grayscale_node2);

	train_graph->add_node(train_laplacian_node1);
	train_graph->add_node(train_laplacian_node2);
	train_graph->add_node(train_laplacian_node3);
	train_graph->add_node(train_laplacian_node4);
	train_graph->add_node(train_laplacian_node5);
	train_graph->add_node(train_laplacian_node6);
	train_graph->add_node(train_laplacian_node7);
	train_graph->add_node(train_laplacian_node8);
	train_graph->add_node(train_laplacian_node9);
	train_graph->add_node(train_laplacian_node10);
	
	// Add train edges 
	Edge *train_edge1 = new Edge("edge1", "read_node", "grayscale_node1");
	Edge *train_edge2 = new Edge("edge2", "read_node", "grayscale_node2");

	Edge *train_edge3 = new Edge("edge3", "grayscale_node1", "laplacian_node1");
	Edge *train_edge4 = new Edge("edge4", "grayscale_node1", "laplacian_node2");
	Edge *train_edge5 = new Edge("edge5", "grayscale_node1", "laplacian_node3");
	Edge *train_edge6 = new Edge("edge6", "grayscale_node1", "laplacian_node4");
	Edge *train_edge7 = new Edge("edge7", "grayscale_node1", "laplacian_node5");
	Edge *train_edge8 = new Edge("edge8", "grayscale_node1", "laplacian_node6");
	Edge *train_edge9 = new Edge("edge9", "grayscale_node1", "laplacian_node7");
	Edge *train_edge10 = new Edge("edge10", "grayscale_node1", "laplacian_node8");

	Edge *train_edge11 = new Edge("edge11", "grayscale_node2", "laplacian_node9");
	Edge *train_edge12 = new Edge("edge12", "grayscale_node2", "laplacian_node10");

	train_graph->add_edge(train_edge1);
	train_graph->add_edge(train_edge2);
	train_graph->add_edge(train_edge3);
	train_graph->add_edge(train_edge4);
	train_graph->add_edge(train_edge5);
	train_graph->add_edge(train_edge6);
	train_graph->add_edge(train_edge7);
	train_graph->add_edge(train_edge8);
	train_graph->add_edge(train_edge9);
	train_graph->add_edge(train_edge10);
	train_graph->add_edge(train_edge11);
	train_graph->add_edge(train_edge12);

	std::cout << "*Graph defined*" << std::endl;
	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs in parallel
	boost::thread_group threads;
	threads.create_thread(boost::bind(&GraphNet::run, boost::ref(train_graph)));
	threads.join_all();

	/*********************************************    Clean   ***************************************************/
	
	// Stop clock
	std::cout << "Elapsed Time: " << float( utils::get_time() - begin_time )  << std::endl;
	return 0;
}
