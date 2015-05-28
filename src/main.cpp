#include "graph_net.h"
#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "edge.h"

int main (int argc, char * argv[])
{
	// Declare variables
	GraphNet graph = GraphNet();
	std::string fname = "/Users/nelson/Desktop/WorkSpaces/GSOCSpace/resource/65351.svs";
	std::vector<std::tuple<int, int>> coords;
	coords.push_back({200,200});
	coords.push_back({500,500});
	coords.push_back({2000,2000});

	// Add some nodes
	graph.add_node(ReadNode("read_node", fname, coords));
	graph.add_node(LaplacianPyramidNode("laplacian_node", 4));
	graph.add_node(DebugNode("debug_node"));

	// Add edges
	graph.add_edge(Edge("read_node", "laplacian_node"));
	graph.add_edge(Edge("laplacian_node", "debug_node"));

	return 0;
}