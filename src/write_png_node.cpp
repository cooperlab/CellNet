#include "write_png_node.h"
#include "edge.h"
#include "utils.h"

WritePNGNode::WritePNGNode(std::string id, std::string fname, int mode): Node(id, mode), _fname(fname), _counter(0){
	runtime_total_first = utils::get_time();
}

void *WritePNGNode::run(){

	while(true){
		cv::Mat out;
		copy_from_buffer(out);
		if(!out.empty()){

			//std::cout << "Writing sample" << std::endl; 
			write_to_disk(out);
			out.release();

			//std::cout << "Time to write png: " << float( utils::get_time() - begin_time )  << std::endl;
		}
		else if(_in_edges.at(0)->is_in_node_done()){
				std::cout << "******************" << std::endl;
				std::cout << "WritePNGNode complete" << std::endl;
				std::cout << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl;
				std::cout << "******************" << std::endl;
			break;
		}
	}
	return NULL;
}

void WritePNGNode::write_to_disk(cv::Mat &out){

	cv::imwrite(_fname + "/sample_" + std::to_string(_counter++) + ".png", out);	
}