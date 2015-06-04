#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "debug_node.h"
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

	std::cout << "Defining cropping coordinates..." << std::endl;
	coords.push_back(std::make_tuple(200,200));
	coords.push_back(std::make_tuple(500,500));
	coords.push_back(std::make_tuple(2000,2000));

	std::cout << "Defining graph nodes..." << std::endl;

	// Add some nodes
	file_paths.push_back("/home/nelson/CellNet/resource/14276.svs");
	ReadNode *read_node = new ReadNode("read_node", file_paths, coords);
	LaplacianPyramidNode *laplacian_node = new LaplacianPyramidNode("laplacian_node");
	DebugNode *debug_node = new DebugNode("debug_node");

	graph->add_node(read_node);
	graph->add_node(laplacian_node);
	graph->add_node(debug_node);
	std::cout << "*Nodes defined*" << std::endl;

	std::cout << "Defining graph edges..." << std::endl;
	// Add edges
	Edge *edge1 = new Edge("edge1", "read_node", "laplacian_node");
	Edge *edge2 = new Edge("edge2", "laplacian_node", "debug_node");

	graph->add_edge(edge1);
	graph->add_edge(edge2);

	std::cout << "*Graph defined*" << std::endl;

	// Start serial execution
	graph->run();

	// Free memory
	//delete edge1;
	//delete edge2;

	//delete graph;
	return 0;
}