#include "laplacian_pyramid_node.h"
#include "read_node.h"
#include "edge.h"

int main (int argc, char * argv[])
{
	boost::uuids::uuid uuid = boost::uuids::random_generator()();
	ReadNode read_node = ReadNode("read_node", uuid, "/Users/nelson/Desktop/WorkSpaces/GSOCSpace/resource/65351.svs");
	read_node.run();
	read_node.show_entire_image();

	std::vector<std::tuple<int, int>> coords;
	coords.push_back({200,200});
	coords.push_back({500,500});
	coords.push_back({2000,2000});
	read_node.crop_cells(coords);
	read_node.show_cropped_cells();
	//uuid = boost::uuids::random_generator()();
	//LaplacianPyramidNode laplacian_node = LaplacianPyramidNode("laplacian_node", uuid, img, 4);
	//laplacian_node.print_pyramid();	
	return 0;
}