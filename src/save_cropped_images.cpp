#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_png_node.h" 
#include "grayscale_node.h"
#include "write_hdf5_node.h"
#include "read_write_node.h"
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
#define NUMB_LAPLACIAN_NODE 2
#define NUMB_READ_NODE 1
#define NUMB_AUGMENTATION_NODE 0
#define GPU_ID 0
#define SERIAL 0
#define PARALLEL 1

const static std::string IMAGE_PATH = "/home/lcoop22/Images/LGG";
const static std::string LOCAL_HOME = "/home/nnauata";
const static std::string fname = "/home/mnalisn/testsets/LGG-Endothelial-combined-fixed.h5";

int main (int argc, char * argv[])
{

	// Start clock
	double begin_time = utils::get_time();

	/**************************************** Get Input Data  ***************************************/

	// Define slides to use for converting
	std::vector<int> convert_slides = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87};


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
	/************************************ Create convert Dataset ************************************/

	// Declare Variables
	long long unsigned int num_elems = x_centroid.size();
	GraphNet *convert_graph = new GraphNet(PARALLEL);
	std::vector<std::string> convert_file_paths;
	std::vector<std::vector<std::tuple<float, float>>> convert_cells_coordinates_set;
	std::string convert_dataset_name = "data";
	std::vector<std::vector<int>> convert_labels;

	// Create input
	for(int k = 0; k < slides.size(); k++){

		// Create data for each slide
		std::vector<std::tuple<float, float>> convert_slide;
		std::vector<int> label_slide;

		// Append
		convert_cells_coordinates_set.push_back(convert_slide);
		convert_labels.push_back(label_slide);
	}
	

	/******************************** Shuffle & Split Data ******************************************/

	float begin_time_2 = utils::get_time();
	utils::fill_data(num_elems, num_elems, convert_cells_coordinates_set, convert_labels, x_centroid, y_centroid, labels, slide_idx);

	std::cout << "Time to fill data: " << float( utils::get_time() - begin_time_2)  << std::endl;
	
	/********************************    Setup Graphs     *******************************************/
	
	//Define paths
	for(int k = 0; k < slides.size(); k++){

		std::string img_name = utils::get_image_name(slides[k], IMAGE_PATH);
		convert_file_paths.push_back(IMAGE_PATH + "/" + img_name);
	}

	/********************************    Remove Slides   ********************************************/

	utils::remove_slides(convert_file_paths, convert_cells_coordinates_set, convert_labels, convert_slides);
	int total = 0;
	for(int k=0; k < convert_cells_coordinates_set.size(); k++){

		std::cout << "Slide name: " << convert_file_paths[k] << std::endl;
		std::cout << "Slide #: "  << convert_slides[k] << std::endl;
		std::cout << "# of samples: " << convert_cells_coordinates_set[k].size() << std::endl;
		total += convert_cells_coordinates_set[k].size();
	}
	std::cout << "Total # of samples" << total << std::endl;
	/************************************************************************************************/

	// Define Graphs
	std::cout << "Defining graph nodes..." << std::endl;

	// Add some convert Nodes
	convert_graph->add_node(new ReadWriteNode("read_write_node", slides, convert_file_paths, convert_cells_coordinates_set, convert_labels, REPEAT_MODE));

	std::cout << "*Graph defined*" << std::endl;
	
	/********************************************* Run Graphs ***************************************************/
	
	// Run graphs in parallel
	boost::thread_group threads;
	threads.create_thread(boost::bind(&GraphNet::run, boost::ref(convert_graph)));
	threads.join_all();

	/*********************************************    Clean   ***************************************************/
	
	// Stop clock
	std::cout << "Elapsed Time: " << double( utils::get_time() - begin_time )  << std::endl;
	
	// Release memory
	delete convert_graph;
	
	return 0;
}

