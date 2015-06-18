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

const static std::string LOCAL_HOME = "/home/nelson";

void fill_data(int N, int num_elem, std::vector<std::vector<std::tuple<double, double>>> &cells_coordinates_set, std::vector<double> &shuffled_labels, std::vector<double> &x_centroid, std::vector<double> &y_centroid, std::vector<double> &labels, std::vector<double> &slide_idx){

	std::vector<double> labels1;
	std::vector<double> labels2;

	// Fill train dataset
	srand (time(NULL));
	labels1.clear();
	labels2.clear();
	int k = 0;
	for(int i=0; i < N; i++){

		k = rand() % num_elem;
		if(!slide_idx[k]){
			//cells_coordinates_set[0].push_back(std::make_tuple(x_centroid[k], y_centroid[k]));
			//labels1.push_back(labels[k]);
		}
		else{
			std::cout << std::to_string(x_centroid[k]) << " " << std::to_string(y_centroid[k]) << " " << std::to_string(labels[k]) << std::endl;
			cells_coordinates_set[0].push_back(std::make_tuple(x_centroid[k], y_centroid[k]));
			labels2.push_back(labels[k]);
		}

		x_centroid.erase(x_centroid.begin() + k);
		y_centroid.erase(y_centroid.begin() + k);
		labels.erase(labels.begin() + k);
		num_elem--;
	}

	shuffled_labels.clear();
	shuffled_labels.reserve(labels1.size() + labels2.size());
	shuffled_labels.insert( shuffled_labels.end(), labels1.begin(), labels1.end());
	shuffled_labels.insert( shuffled_labels.end(), labels2.begin(), labels2.end());
}


int main (int argc, char * argv[])
{

	// Start clock
	const clock_t begin_time = clock();
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
	utils::get_data(LOCAL_HOME + "/LGG-test/LGG-Endothelial-2-test.h5", "labels", labels);

	/************************************* Create Train Dataset ************************************/

	// Declare Variables
	GraphNet *train_graph = new GraphNet();
	std::vector<std::string> train_file_paths;
	std::vector<std::vector<std::tuple<double, double>>> train_cells_coordinates_set;
	std::vector<hsize_t> train_dim = {108, 4, 1, 50, 50};
	std::string train_dataset_name = "train_data";
	std::vector<double> train_labels;

	// Create input
	std::vector<std::tuple<double, double>> train_slide1;
	std::vector<std::tuple<double, double>> train_slide2;
	train_cells_coordinates_set.push_back(train_slide1);
	train_cells_coordinates_set.push_back(train_slide2);

	/************************************* Create Test Dataset ************************************/

	// Declare Variables
	GraphNet *test_graph = new GraphNet();
	std::vector<std::string> test_file_paths;
	std::vector<std::vector<std::tuple<double, double>>> test_cells_coordinates_set;
	std::vector<hsize_t> test_dim = {20, 4, 1, 50, 50};
	std::string test_dataset_name = "test_data";
	std::vector<double> test_labels;

	// Create input
	std::vector<std::tuple<double, double>> test_slide1;
	std::vector<std::tuple<double, double>> test_slide2;
	test_cells_coordinates_set.push_back(test_slide1);
	test_cells_coordinates_set.push_back(test_slide2);


	/******************************** Create Validation Dataset ***********************************/

	// Declare Variables
	GraphNet *valid_graph = new GraphNet();
	std::vector<std::string> valid_file_paths;
	std::vector<std::vector<std::tuple<double, double>>> valid_cells_coordinates_set;
	std::vector<hsize_t> valid_dim = {20, 4, 1, 50, 50};
	std::string valid_dataset_name = "valid_data";
	std::vector<double> valid_labels;

	// Create input
	std::vector<std::tuple<double, double>> valid_slide1;
	std::vector<std::tuple<double, double>> valid_slide2;
	valid_cells_coordinates_set.push_back(valid_slide1);
	valid_cells_coordinates_set.push_back(valid_slide2);

	/******************************** Shuffle & Split Data ******************************************/

	//fill_data(108, 148, train_cells_coordinates_set, train_labels, x_centroid, y_centroid, labels, slide_idx);
	//fill_data(20, 40, test_cells_coordinates_set, test_labels, x_centroid, y_centroid, labels, slide_idx);
	//fill_data(20, 20, valid_cells_coordinates_set, valid_labels, x_centroid, y_centroid, labels, slide_idx);
	
	fill_data(10, 148, train_cells_coordinates_set, train_labels, x_centroid, y_centroid, labels, slide_idx);
	fill_data(10, 138, test_cells_coordinates_set, test_labels, x_centroid, y_centroid, labels, slide_idx);
	fill_data(10, 128, valid_cells_coordinates_set, valid_labels, x_centroid, y_centroid, labels, slide_idx);
	//std::cout << "labels_size: " << std::to_string(train_labels.size()) << std::endl;

	/********************************    Setup Graphs     *******************************************/

	//Define paths
	//train_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-EZ-7264-01Z-00-DX1.80a61d74-77d9-4998-bb55-213767a588ff.svs");
	train_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-HT-7474-01Z-00-DX1.B3E88862-6C35-4E30-B374-A7BC80231B8C.svs");
	//test_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-EZ-7264-01Z-00-DX1.80a61d74-77d9-4998-bb55-213767a588ff.svs");
	test_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-HT-7474-01Z-00-DX1.B3E88862-6C35-4E30-B374-A7BC80231B8C.svs");
	//valid_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-EZ-7264-01Z-00-DX1.80a61d74-77d9-4998-bb55-213767a588ff.svs");
	valid_file_paths.push_back(LOCAL_HOME + "/LGG-test/TCGA-HT-7474-01Z-00-DX1.B3E88862-6C35-4E30-B374-A7BC80231B8C.svs");

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Train Nodes
	ReadNode *train_read_node = new ReadNode("read_node", train_file_paths, train_cells_coordinates_set);
	LaplacianPyramidNode *train_laplacian_node = new LaplacianPyramidNode("laplacian_node");
	WritePNGNode *train_write_png_node = new WritePNGNode("write_png_node", LOCAL_HOME + "/CellNet/train/data_gray/");
	WritePNGNode *train_write_png_node2 = new WritePNGNode("write_png_node2", LOCAL_HOME + "/CellNet/train/data_rgb/");
	DebugNode *train_debug_node = new DebugNode("debug_node");
	GrayScaleNode *train_grayscale_node = new GrayScaleNode("grayscale_node");
	WriteHDF5Node *train_write_hdf5_node = new WriteHDF5Node("write_hdf5_node", LOCAL_HOME + "/CellNet/train/data_hdf5/train", train_dim, train_dataset_name, train_labels);

	train_graph->add_node(train_read_node);
	train_graph->add_node(train_laplacian_node);
	train_graph->add_node(train_debug_node);
	train_graph->add_node(train_write_png_node);
	train_graph->add_node(train_write_png_node2);
	train_graph->add_node(train_grayscale_node);
	train_graph->add_node(train_write_hdf5_node);

	// Add some Test Nodes
	ReadNode *test_read_node = new ReadNode("read_node", test_file_paths, test_cells_coordinates_set);
	LaplacianPyramidNode *test_laplacian_node = new LaplacianPyramidNode("laplacian_node");
	WritePNGNode *test_write_png_node = new WritePNGNode("write_png_node", LOCAL_HOME + "/CellNet/test/data_gray/");
	WritePNGNode *test_write_png_node2 = new WritePNGNode("write_png_node2", LOCAL_HOME + "/CellNet/test/data_rgb/");
	DebugNode *test_debug_node = new DebugNode("debug_node");
	GrayScaleNode *test_grayscale_node = new GrayScaleNode("grayscale_node");
	WriteHDF5Node *test_write_hdf5_node = new WriteHDF5Node("write_hdf5_node", LOCAL_HOME + "/CellNet/test/data_hdf5/test", test_dim, test_dataset_name, test_labels);

	test_graph->add_node(test_read_node);
	test_graph->add_node(test_laplacian_node);
	test_graph->add_node(test_debug_node);
	test_graph->add_node(test_write_png_node);
	test_graph->add_node(test_write_png_node2);
	test_graph->add_node(test_grayscale_node);
	test_graph->add_node(test_write_hdf5_node);

	// Add some Valid Nodes
	ReadNode *valid_read_node = new ReadNode("read_node", valid_file_paths, valid_cells_coordinates_set);
	LaplacianPyramidNode *valid_laplacian_node = new LaplacianPyramidNode("laplacian_node");
	WritePNGNode *valid_write_png_node = new WritePNGNode("write_png_node", LOCAL_HOME + "/CellNet/valid/data_gray/");
	WritePNGNode *valid_write_png_node2 = new WritePNGNode("write_png_node2", LOCAL_HOME + "/CellNet/valid/data_rgb/");
	DebugNode *valid_debug_node = new DebugNode("debug_node");
	GrayScaleNode *valid_grayscale_node = new GrayScaleNode("grayscale_node");
	WriteHDF5Node *valid_write_hdf5_node = new WriteHDF5Node("write_hdf5_node", LOCAL_HOME + "/CellNet/valid/data_hdf5/valid", valid_dim, valid_dataset_name, valid_labels);

	valid_graph->add_node(valid_read_node);
	valid_graph->add_node(valid_laplacian_node);
	valid_graph->add_node(valid_debug_node);
	valid_graph->add_node(valid_write_png_node);
	valid_graph->add_node(valid_write_png_node2);
	valid_graph->add_node(valid_grayscale_node);
	valid_graph->add_node(valid_write_hdf5_node);

	std::cout << "*Nodes defined*" << std::endl;
	std::cout << "Defining graph edges..." << std::endl;

	// Add train edges 
	Edge *train_edge1 = new Edge("edge1", "read_node", "laplacian_node");
	Edge *train_edge2 = new Edge("edge2", "laplacian_node", "debug_node");
	Edge *train_edge3 = new Edge("edge3", "laplacian_node", "grayscale_node");
	Edge *train_edge4 = new Edge("edge4", "grayscale_node", "write_png_node");
	Edge *train_edge5 = new Edge("edge5", "read_node", "write_png_node2");
	Edge *train_edge6 = new Edge("edge6", "grayscale_node", "write_hdf5_node");

	train_graph->add_edge(train_edge1);
	train_graph->add_edge(train_edge2);
	train_graph->add_edge(train_edge3);
	train_graph->add_edge(train_edge4);
	train_graph->add_edge(train_edge5);
	train_graph->add_edge(train_edge6);

	// Add test edges
	Edge *test_edge1 = new Edge("edge1", "read_node", "laplacian_node");
	Edge *test_edge2 = new Edge("edge2", "laplacian_node", "debug_node");
	Edge *test_edge3 = new Edge("edge3", "laplacian_node", "grayscale_node");
	Edge *test_edge4 = new Edge("edge4", "grayscale_node", "write_png_node");
	Edge *test_edge5 = new Edge("edge5", "read_node", "write_png_node2");
	Edge *test_edge6 = new Edge("edge6", "grayscale_node", "write_hdf5_node");

	test_graph->add_edge(test_edge1);
	test_graph->add_edge(test_edge2);
	test_graph->add_edge(test_edge3);
	test_graph->add_edge(test_edge4);
	test_graph->add_edge(test_edge5);
	test_graph->add_edge(test_edge6);

	// Add valid edges
	Edge *valid_edge1 = new Edge("edge1", "read_node", "laplacian_node");
	Edge *valid_edge2 = new Edge("edge2", "laplacian_node", "debug_node");
	Edge *valid_edge3 = new Edge("edge3", "laplacian_node", "grayscale_node");
	Edge *valid_edge4 = new Edge("edge4", "grayscale_node", "write_png_node");
	Edge *valid_edge5 = new Edge("edge5", "read_node", "write_png_node2");
	Edge *valid_edge6 = new Edge("edge6", "grayscale_node", "write_hdf5_node");

	valid_graph->add_edge(valid_edge1);
	valid_graph->add_edge(valid_edge2);
	valid_graph->add_edge(valid_edge3);
	valid_graph->add_edge(valid_edge4);
	valid_graph->add_edge(valid_edge5);
	valid_graph->add_edge(valid_edge6);

	std::cout << "*Graph defined*" << std::endl;
	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs in parallel
	boost::thread_group threads;
	threads.create_thread(boost::bind(&GraphNet::run, boost::ref(train_graph)));
	threads.create_thread(boost::bind(&GraphNet::run, boost::ref(test_graph)));
	threads.create_thread(boost::bind(&GraphNet::run, boost::ref(valid_graph)));
	threads.join_all();

	/*********************************************    Clean   ***************************************************/
	
	// Stop clock
	std::cout << "Elapsed Time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
	return 0;
}