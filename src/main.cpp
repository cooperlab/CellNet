#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
#include "write_node.h" 
#include "grayscale_node.h"
#include "edge.h"
#include <tuple>
#include <vector>
#include <string>
#include <iostream>

int main (int argc, char * argv[])
{

	// Declare variables
	GraphNet *graph = new GraphNet();
	std::vector<std::string> file_paths;
	std::vector<std::tuple<int, int>> coords;

	std::cout << "Defining nodes arguments..." << std::endl;
	coords.push_back(std::make_tuple(7000,7000));
	coords.push_back(std::make_tuple(8000,8000));
	coords.push_back(std::make_tuple(10000,10000));

	file_paths.push_back("/home/nelson/CellNet/resource/14276.svs");

	std::cout << "Defining graph nodes..." << std::endl;

	// Add some Nodes
	ReadNode *read_node = new ReadNode("read_node", file_paths, coords);
	LaplacianPyramidNode *laplacian_node = new LaplacianPyramidNode("laplacian_node");
	WriteNode *write_node = new WriteNode("write_node", "/home/nelson/CellNet/data/");
	WriteNode *write_node2 = new WriteNode("write_node2", "/home/nelson/CellNet/data2/");
	DebugNode *debug_node = new DebugNode("debug_node");
	GrayScaleNode *grayscale_node = new GrayScaleNode("grayscale_node");

	graph->add_node(read_node);
	graph->add_node(laplacian_node);
	graph->add_node(debug_node);
	graph->add_node(write_node);
	graph->add_node(write_node2);
	graph->add_node(grayscale_node);

	std::cout << "*Nodes defined*" << std::endl;

	std::cout << "Defining graph edges..." << std::endl;
	// Add edges
	Edge *edge1 = new Edge("edge1", "read_node", "laplacian_node");
	Edge *edge2 = new Edge("edge2", "laplacian_node", "debug_node");
	Edge *edge3 = new Edge("edge3", "laplacian_node", "grayscale_node");
	Edge *edge4 = new Edge("edge4", "grayscale_node", "write_node");
	Edge *edge5 = new Edge("edge5", "read_node", "write_node2");

	graph->add_edge(edge1);
	graph->add_edge(edge2);
	graph->add_edge(edge3);
	graph->add_edge(edge4);
	graph->add_edge(edge5);

	std::cout << "*Graph defined*" << std::endl;

	// Start serial execution
	graph->run();

	return 0;
}