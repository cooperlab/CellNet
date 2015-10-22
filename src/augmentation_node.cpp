#define PI 3.1415926535897932
#include <cmath>
#include "augmentation_node.h"
#include "edge.h"
#include "utils.h"
#include <random>
#include <chrono>
#define SHIFT 25
AugmentationNode::AugmentationNode(std::string id, int mode, int aug_factor): Node(id, mode), _counter(0), _data_buffer(), _labels_buffer(), _aug_factor(aug_factor){
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
		_counter++;
		cv::Mat src(imgs[k]);
		
		// Get ROI
		float tl_row = src.rows/2.0F - SHIFT;
		float tl_col = src.cols/2.0F - SHIFT;
		float br_row = src.rows/2.0F + SHIFT;
		float br_col = src.cols/2.0F + SHIFT;
		cv::Point tl(tl_row, tl_col);
		cv::Point br(br_row, br_col);

		// Setup a rectangle to define region of interest
		cv::Rect cellROI(tl, br);

		for(int i=0; i < _aug_factor; i++){
			_counter++;

			// Define variables
			int label = labels[k];

			// construct a trivial random generator engine from a time-based seed:
	  		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	  		std::default_random_engine generator (seed);
			std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

			double Flipx = round(uniform_dist(generator)); 				// randomly distributed reflections over x and y
			double Flipy = round(uniform_dist(generator));
			double Phi = 360 * uniform_dist(generator); 			// rotation angle uniformly distributed on [0, 2pi] radians
			double Tx = 20 * (uniform_dist(generator) - 0.5); 			// 10-pixel uniformly distributed translation
			double Ty = 20 * (uniform_dist(generator) - 0.5);
			double Shx = PI/180 * 20 * (uniform_dist(generator)-0.5);  	// shear angle uniformly distributed on [-20, 20] degrees
			double Shy = PI/180 * 20 * (uniform_dist(generator)-0.5);
			double Sx = 1/1.2 + (1.2 - 1/1.2)*uniform_dist(generator); 	// scale uniformly distributed on [1/1.2, 1.2]
			double Sy = 1/1.2 + (1.2 - 1/1.2)*uniform_dist(generator);

			cv::Mat warped_img;

			// Translation
			cv::Mat trans_M(3, 3, CV_64F);
			trans_M.at<double>(0,0) = 1;
			trans_M.at<double>(0,1) = 0;
			trans_M.at<double>(0,2) = Tx;

			trans_M.at<double>(1,0) = 0;
			trans_M.at<double>(1,1) = 1;
			trans_M.at<double>(1,2) = Ty;

			trans_M.at<double>(2,0) = 0;
			trans_M.at<double>(2,1) = 0;
			trans_M.at<double>(2,2) = 0;

			// Scaling
			cv::Mat scaling_M(3, 3, CV_64F);
			scaling_M.at<double>(0,0) = Sx;
			scaling_M.at<double>(0,1) = 0;
			scaling_M.at<double>(0,2) = 0;

			scaling_M.at<double>(1,0) = 0;
			scaling_M.at<double>(1,1) = Sy;
			scaling_M.at<double>(1,2) = 0;

			scaling_M.at<double>(2,0) = 0;
			scaling_M.at<double>(2,1) = 0;
			scaling_M.at<double>(2,2) = 0;

			// Shearing 
			cv::Mat shear_M(3, 3, CV_64F);
			shear_M.at<double>(0,0) = 1;
			shear_M.at<double>(0,1) = Shy;
			shear_M.at<double>(0,2) = 0;

			shear_M.at<double>(1,0) = Shx;
			shear_M.at<double>(1,1) = 1;
			shear_M.at<double>(1,2) = 0;

			shear_M.at<double>(2,0) = 0;
			shear_M.at<double>(2,1) = 0;
			shear_M.at<double>(2,2) = 0;

			// Rotation 
			cv::Mat rot(2, 3, CV_64F);
			cv::Point2f center(src.rows/2.0F, src.cols/2.0F);
			rot = getRotationMatrix2D(center, Phi, 1.0);	

			cv::Mat rot_M(3, 3, CV_64F, cv::Scalar(0));
			rot(cv::Rect(0,0,3,2)).copyTo(rot_M.colRange(0,3).rowRange(0,2));

			// Accumulate
			cv::Mat acc(3, 3, CV_64F);

			acc = trans_M * scaling_M * shear_M * rot_M;

			cv::Mat acc_M = acc.colRange(0,3).rowRange(0,2);
			cv::warpAffine( src, warped_img, acc_M, src.size());

			// Crop Image
			double m00 = acc_M.at<double>(0,0);
			double m01 = acc_M.at<double>(0,1);
			double m10 = acc_M.at<double>(1,0);
			double m11 = acc_M.at<double>(1,1);
			double m02 = acc_M.at<double>(0,2);
			double m12 = acc_M.at<double>(1,2); 

			//std::cout << m00 << "," << m01 << "," << m02 << std::endl; 
			//std::cout << m10 << "," << m11 << "," << m12 << std::endl;
			int new_cx = src.rows/2.0F * m00 + src.cols/2.0F * m01 + m02;
			int new_cy = src.rows/2.0F * m10 + src.cols/2.0F * m11 + m12;

			// Get ROI
			double tl_row = new_cx - SHIFT;
			double tl_col = new_cy - SHIFT;
			double br_row = new_cx + SHIFT;
			double br_col = new_cy + SHIFT;
			cv::Point tl(tl_row, tl_col);
			cv::Point br(br_row, br_col);

			//std::cout << scaling_M << std::endl;
			//std::cout << new_cx << " " << new_cy << std::endl;
			//std::cout << tl << std::endl;
			//std::cout << br << std::endl;
			//std::cout << "*****" << std::endl; 
			// Setup a rectangle to define region of interest
			cv::Rect new_cellROI(tl, br);
			cv::Mat final_img = warped_img(new_cellROI);
			
			// Flip
			if(Flipx == 1 && Flipy == 1){
				cv::flip(final_img, final_img, -1);
			}
			else if(Flipx == 0 && Flipy == 1){
				cv::flip(final_img, final_img, 1);
			}
			else if(Flipx == 1 && Flipy == 0){
				cv::flip(final_img, final_img, 0);
			}
			
        	        if(!final_img.isContinuous()){
	                        final_img=final_img.clone();
                	}

			// Accumulate
			out_imgs.push_back(final_img);
			out_labels.push_back(label);
		}

		// Crop Image
		cv::Mat new_src = src(cellROI);
                if(!new_src.isContinuous()){
			new_src=new_src.clone();
		}
		out_imgs.push_back(new_src);
		out_labels.push_back(labels[k]);
		copy_to_buffer(out_imgs, out_labels);

		// Clean
		out_imgs.clear();
		out_labels.clear();
	}
}
 
