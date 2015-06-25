#include "laplacian_pyramid_node.h"
#include "utils.h"
#define KERNELL_SIZE 3

LaplacianPyramidNode::LaplacianPyramidNode(std::string id): Node(id) {
}

void *LaplacianPyramidNode::run(){

	while(true){
		cv::Mat _layer0; 

		//std::cout << "Copying from buffer" << std::endl;
		copy_from_buffer(_layer0);
		//std::cout << "Copy finished" << std::endl;
		
		if(!_layer0.empty()){

			double begin_time = utils::get_time();
			//std::cout << "LaplacianPyramidNode start" << std::endl;

			cv::Mat _gaussian_layer0;
			std::vector<cv::Mat> layers;
		
			// Build Laplacian pyramid
			cv::GaussianBlur(_layer0, _gaussian_layer0	, cv::Size(KERNELL_SIZE, KERNELL_SIZE), 0, 0);
			
			//std::cout << "Start building pyramid" << std::endl;
			gen_next_level(_gaussian_layer0, _layer0, &layers, 0);
			//std::cout << "Pyramid built" << std::endl;
			resize_all(layers, _layer0.size());

			copy_to_buffer(layers);
			//std::cout << "Number of layers: " << std::to_string(layers.size()) << std::endl;
			
			// Release memory
			_gaussian_layer0.release();
			layers.clear();

			count++;
			runtime_average_first += float( utils::get_time() - begin_time );
			//std::cout << "LaplacianPyramidNode complete" << std::endl;
			//std::cout << "Time to take laplacian pyramid: " << float( utils::get_time() - begin_time )  << std::endl;
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			//std::cout << "Stopping LaplacianPyramidNode" << std::endl;
				std::cout << "******************" << std::endl;
				std::cout << "LaplacianPyramidNode complete" << std::endl;
				std::cout << "Avg_first: " << std::to_string(runtime_average_first/count) << std::endl;
				std::cout << "Total_time_first: " << std::to_string(runtime_average_first) << std::endl;
				std::cout << "******************" << std::endl;
			break;
		}
		_layer0.release();
	}

	// Notify it has finished
	for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
		_out_edges.at(i)->set_in_node_done();
	}
	/****************** Debug ******************/
	//print_pyramid();
	return NULL;
}

void LaplacianPyramidNode::resize_all(std::vector<cv::Mat> &layers, cv::Size size){

	for(std::vector<cv::Mat>::size_type i =0; i < layers.size(); i++){
		cv::resize(layers.at(i), layers.at(i), size, 0, 0, CV_INTER_CUBIC);
	}
}

/* This method builds a Laplacian Pyramid with '_n_layers' layers indexed from 0 to '_n_layers'-1   */
/* In order to obtain '_n_layers' laplacian layers we build a pyramid with '_n_layers' + 1 layers,  */
/* where the last layer corresponds to a gaussian layer and the first '_n_layers' layers correspond */
/* to laplacian layers. 																			*/

void LaplacianPyramidNode::gen_next_level(cv::Mat _current_gaussian_layer, cv::Mat _current_layer, std::vector<cv::Mat> *layers, int rec_level){
	
	cv::Mat _next_layer;
	cv::Mat _next_gaussian_layer;
	cv::Mat _current_laplacian;
	cv::Mat _upsampled_gaussian_layer;
	
	// Downsample image
	if(rec_level < N_LAYERS-1){

		// Downsample current layer using cubic interpolation
		cv::resize(_current_layer, _next_layer, cv::Size(), 0.5, 0.5, CV_INTER_CUBIC);
		
		// Apply Gaussian smooth 
		cv::GaussianBlur( _next_layer, _next_gaussian_layer, cv::Size( KERNELL_SIZE, KERNELL_SIZE ), 0, 0 );
		
		// Upsample
		cv::resize(_next_gaussian_layer, _upsampled_gaussian_layer, _current_layer.size(), 0, 0, CV_INTER_CUBIC);
		
		// Compute L_i = G_i - G_{i+1}
		_current_laplacian =  _current_gaussian_layer - _upsampled_gaussian_layer;
		
		// Stack Laplacian pyramid
		layers->push_back(_current_laplacian);
		
		//Recursive call
		gen_next_level(_next_gaussian_layer, _next_layer, layers, rec_level+1);
	}
	else{
		// Last layer
		layers->push_back(_current_gaussian_layer);
	}		
}

void LaplacianPyramidNode::print_pyramid(std::vector<cv::Mat> layers){
	int i;
	cv::Mat tmp_layer;
	std::string w_name; 
	
		for(i = 0; i < _n_layers; i++){
			
			// Print each layer
			w_name = "Layer: " + std::to_string(i);
			cv::normalize(layers.at(i), tmp_layer, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::imshow(w_name, tmp_layer);
		}
		cv::waitKey(0);
	}
