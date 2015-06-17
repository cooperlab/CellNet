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

int main (int argc, char * argv[])
{

	const clock_t begin_time = clock();
	/***************************** Declare nodes variables  **********************/
 	GraphNet *graph = new GraphNet();
	std::vector<std::string> file_paths;
	std::vector<std::vector<std::tuple<double, double>>> cells_coordinates_set;

	std::cout << "Defining nodes arguments..." << std::endl;
	std::vector<hsize_t> dim = {50, 4, 1, 50, 50};
	std::string dataset_name = "training_data";

	// Input data
	std::vector<double> x_centroid;
	std::vector<double> y_centroid;
	std::vector<double> slide_idx;
	

	// Get data from HDF5
	utils::get_data("/home/nelson/LGG-test/LGG-Endothelial-2-test.h5", "x_centroid", x_centroid);
	utils::get_data("/home/nelson/LGG-test/LGG-Endothelial-2-test.h5", "y_centroid", y_centroid);
	utils::get_data("/home/nelson/LGG-test/LGG-Endothelial-2-test.h5", "slideIdx", slide_idx);

	// Create input
	std::vector<std::tuple<double, double>> slide1;
	std::vector<std::tuple<double, double>> slide2;

	cells_coordinates_set.push_back(slide1);
	cells_coordinates_set.push_back(slide2);

	//file_paths.push_back("/home/nelson/LGG-test/TCGA-EZ-7264-01Z-00-DX1.80a61d74-77d9-4998-bb55-213767a588ff.svs");
	file_paths.push_back("/home/nelson/LGG-test/TCGA-HT-7474-01Z-00-DX1.B3E88862-6C35-4E30-B374-A7BC80231B8C.svs");

	//std::cout << "size: " << std::to_string(x_centroid.size()) << std::endl;
	for(int i=0; i < 5; i++){

		//std::cout << "x: " << std::to_string(x_centroid[i]) << ", " << "y: " << std::to_string(y_centroid[i]) << std::endl;
		if((int)slide_idx[i] == 0){
			cells_coordinates_set[0].push_back(std::make_tuple(x_centroid[i], y_centroid[i]));
		}
		else{
			//cells_coordinates_set[1].push_back(std::make_tuple(x_centroid[i], y_centroid[i]));
		}
	}
	/***********************************************************************/

	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Nodes
	ReadNode *read_node = new ReadNode("read_node", file_paths, cells_coordinates_set);
	LaplacianPyramidNode *laplacian_node = new LaplacianPyramidNode("laplacian_node");
	WritePNGNode *write_png_node = new WritePNGNode("write_png_node", "/home/nelson/CellNet/data_gray/");
	WritePNGNode *write_png_node2 = new WritePNGNode("write_png_node2", "/home/nelson/CellNet/data_rgb/");
	DebugNode *debug_node = new DebugNode("debug_node");
	GrayScaleNode *grayscale_node = new GrayScaleNode("grayscale_node");
	WriteHDF5Node *write_hdf5_node = new WriteHDF5Node("write_hdf5_node", "/home/nelson/CellNet/data_hdf5/train", dim, dataset_name);

	graph->add_node(read_node);
	graph->add_node(laplacian_node);
	graph->add_node(debug_node);
	graph->add_node(write_png_node);
	graph->add_node(write_png_node2);
	graph->add_node(grayscale_node);
	graph->add_node(write_hdf5_node);

	std::cout << "*Nodes defined*" << std::endl;
	std::cout << "Defining graph edges..." << std::endl;

	// Add edges
	Edge *edge1 = new Edge("edge1", "read_node", "laplacian_node");
	Edge *edge2 = new Edge("edge2", "laplacian_node", "debug_node");
	Edge *edge3 = new Edge("edge3", "laplacian_node", "grayscale_node");
	Edge *edge4 = new Edge("edge4", "grayscale_node", "write_png_node");
	Edge *edge5 = new Edge("edge5", "read_node", "write_png_node2");
	Edge *edge6 = new Edge("edge6", "grayscale_node", "write_hdf5_node");

	graph->add_edge(edge1);
	graph->add_edge(edge2);
	graph->add_edge(edge3);
	graph->add_edge(edge4);
	graph->add_edge(edge5);
	graph->add_edge(edge6);

	std::cout << "*Graph defined*" << std::endl;
	
	// Start serial execution
	graph->run();

	std::cout << "Elapsed Time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
	return 0;
}