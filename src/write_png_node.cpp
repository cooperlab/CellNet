#include "write_png_node.h"
#include "edge.h"
#include "utils.h"

WritePNGNode::WritePNGNode(std::string id, std::string fname): Node(id), _fname(fname), _counter(0){}

void *WritePNGNode::run(){

	while(true){
		cv::Mat out;
		copy_from_buffer(out);
		if(!out.empty()){

			double begin_time = utils::get_time();

			//std::cout << "Writing sample" << std::endl; 
			write_to_disk(out);
			out.release();

			runtime_average_first += float( utils::get_time() - begin_time );
			//std::cout << "Time to write png: " << float( utils::get_time() - begin_time )  << std::endl;
		}
		else if(_in_edges.at(0)->is_in_node_done()){
				std::cout << "******************" << std::endl;
				std::cout << "WritePNGNode complete" << std::endl;
				std::cout << "Avg_first: " << std::to_string(runtime_average_first/count) << std::endl;
				std::cout << "******************" << std::endl;
			break;
		}
	}
	return NULL;
}

void WritePNGNode::write_to_disk(cv::Mat &out){

	cv::imwrite(_fname + "/sample_" + std::to_string(_counter++) + ".png", out);	
}