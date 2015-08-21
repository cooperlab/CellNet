#define PI 3.1415926535897932
#include <cmath>
#include "augmentation_node.h"
#include "edge.h"
#include "utils.h"
#include <random>
#include <chrono>

AugmentationNode::AugmentationNode(std::string id, int aug_factor): Node(id, 0), _counter(0), _data_buffer(), _labels_buffer(), _aug_factor(aug_factor){
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	_labels_buffer.clear();
}

void *AugmentationNode::run(){

	while(true){

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);
		if(!_data_buffer.empty()){
		
			// Augment data
			augment_images(_data_buffer, _labels_buffer);

			// Clean buffers
			_data_buffer.clear();
			_labels_buffer.clear();
		}
		else{

			// Check if all input nodes have already finished
			bool is_all_done = true;

			for(std::vector<int>::size_type i=0; i < _in_edges.size(); i++){

				if(!_in_edges.at(i)->is_in_node_done()){
					is_all_done = false;
				}
			}			

			// All input nodes have finished
			if(is_all_done){
				std::cout << "******************" << std::endl << "AugmentationNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;
				
				// Notify it has finished
				for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
					_out_edges.at(i)->set_in_node_done();
				}
				break;
			}
		}
	}
	return NULL;
}

void AugmentationNode::augment_images(std::vector<cv::Mat> imgs, std::vector<int> labels){

	std::vector<cv::Mat> out_imgs;
	std::vector<int> out_labels;
	for(int k=0; k < imgs.size(); k++){
		_counter+=2;

		// Define variables
		cv::Mat src(imgs[k]);
		cv::Mat rot_M(2, 3, CV_32F);
		cv::Size double_size(src.rows * 2, src.cols * 2);

		cv::Mat warped_img(double_size, src.type());
		int label = labels[k];

  		// construct a trivial random generator engine from a time-based seed:
  		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  		std::default_random_engine generator (seed);

  		// Rescale for correction
		float Sx = 1.0;
		float Sy = Sx;

		std::cout << Sx << std::endl; 

		cv::Mat double_M(2, 3, CV_32F);
		double_M.at<float>(0,0) = Sx;
		double_M.at<float>(0,1) = 0;
		double_M.at<float>(0,2) = 0;

		double_M.at<float>(1,0) = 0;
		double_M.at<float>(1,1) = Sy;
		double_M.at<float>(1,2) = 0;

		cv::warpAffine( src, warped_img, double_M, warped_img.size());

		// Rotation 
		std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
		float theta = 180 * uniform_dist(generator);

		cv::Point2f src_center(src.cols/2.0F, src.rows/2.0F);
		rot_M = getRotationMatrix2D(src_center, theta, 1.0);
		cv::warpAffine( src, warped_img, rot_M, warped_img.size());

		// Translation
		float Tx = 10*uniform_dist(generator);
		float Ty = 10*uniform_dist(generator);

		cv::Mat trans_M(2, 3, CV_32F);
		trans_M.at<float>(0,0) = 1;
		trans_M.at<float>(0,1) = 0;
		trans_M.at<float>(0,2) = Tx;

		trans_M.at<float>(1,0) = 0;
		trans_M.at<float>(1,1) = 1;
		trans_M.at<float>(1,2) = Ty;

		cv::warpAffine( warped_img, warped_img, trans_M, warped_img.size());

		// Rescaling
		std::uniform_real_distribution<double> sec_uniform_dist(1.0/1.6, 1.6);
		Sx = sec_uniform_dist(generator);
		Sy = Sx;

		std::cout << Sx << std::endl; 

		cv::Mat scaling_M(2, 3, CV_32F);
		scaling_M.at<float>(0,0) = Sx;
		scaling_M.at<float>(0,1) = 0;
		scaling_M.at<float>(0,2) = 0;

		scaling_M.at<float>(1,0) = 0;
		scaling_M.at<float>(1,1) = Sy;
		scaling_M.at<float>(1,2) = 0;


		cv::warpAffine( warped_img, warped_img, scaling_M, warped_img.size());
		
		// Get ROI
		float tl_row = warped_img.rows/2.0F - src.rows/2.0F;
		float tl_col = warped_img.cols/2.0F - src.cols/2.0F;
		float br_row = warped_img.rows/2.0F - src.rows/2.0F;
		float br_col = warped_img.cols/2.0F - src.cols/2.0F;
		cv::Mat final_img = warped_img( cv::Rect(tl_row, tl_col, br_row, br_col) );
 
		// Accumulate
		out_imgs.push_back(src);
		out_imgs.push_back(final_img);
		out_labels.push_back(label);
		out_labels.push_back(label);

		// Copy to buffer
		copy_to_buffer(out_imgs, out_labels);

		// Clean
		out_imgs.clear();
		out_labels.clear();
	}
}