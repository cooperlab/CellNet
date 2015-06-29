#include "laplacian_pyramid_node.h"
#include "utils.h"
#define KERNELL_SIZE 3

LaplacianPyramidNode::LaplacianPyramidNode(std::string id, int mode): Node(id, mode) {
	runtime_total_first = utils::get_time();
}

void *LaplacianPyramidNode::run(){
	increment_threads();

	while(true){

		std::vector<cv::Mat> _layer0; 
		copy_chunk_from_buffer(_layer0);
		if(!_layer0.empty()){

			std::vector<cv::Mat> lap_out;
			for(std::vector<cv::Mat>::size_type i=0; i < _layer0.size(); i++){

				// Initialize
				increment_counter();
				cv::Mat _gaussian_layer0;
				std::vector<cv::Mat> layers;
				std::vector<cv::Mat> merged_layers;
				std::vector<cv::Mat> new_lap_out;

				// Compute Laplacian
				cv::GaussianBlur(_layer0.at(i), _gaussian_layer0	, cv::Size(KERNELL_SIZE, KERNELL_SIZE), 0, 0);
				gen_next_level(_gaussian_layer0, _layer0.at(i), &layers, 0);
				resize_all(layers, _layer0.at(i).size());

				// Merge layers
				merged_layers = merge_all(layers);

				// Accumulate
				new_lap_out.reserve(lap_out.size() + merged_layers.size());
				new_lap_out.insert( new_lap_out.end(), lap_out.begin(), lap_out.end());
				new_lap_out.insert( new_lap_out.end(), merged_layers.begin(), merged_layers.end());
				lap_out = new_lap_out;

				// Release memory
				_gaussian_layer0.release();
				layers.clear();
				merged_layers.clear();
				new_lap_out.clear();
			}

			copy_to_buffer(lap_out);
			lap_out.clear();
		}
		else if(_in_edges.at(0)->is_in_node_done()){
			break;
		}
		_layer0.clear();
	}

	if(check_finished() == true){
	
		std::cout << "******************" <<  std::endl << "LaplacianPyramidNode complete" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;
		
		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}
	/****************** Debug ******************/
	//print_pyramid();
	return NULL;
}

std::vector<cv::Mat> LaplacianPyramidNode::merge_all(std::vector<cv::Mat> &layers){

	std::vector<cv::Mat> merged_layers_vector;
	cv::Mat merged_layers;

	cv::merge(layers, merged_layers);
	merged_layers_vector.push_back(merged_layers);

	return merged_layers_vector;
}

void LaplacianPyramidNode::resize_all(std::vector<cv::Mat> &layers, cv::Size size){

	for(std::vector<cv::Mat>::size_type i = 0; i < layers.size(); i++){
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
