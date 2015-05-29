#include "read_node.h"
#include "edge.h"
#define SHIFT 25
#include <iostream>

ReadNode::ReadNode(std::string id, std::string image_path, std::vector<std::tuple<int, int>> cells_coordinates): Node(id), _extracted_images(), _image_path(image_path), _entire_image(), _cells_coordinates(cells_coordinates){
}

void ReadNode::run(){

	open_image();
	crop_cells();

	/****************** Debug ******************/
	show_entire_image();
	show_cropped_cells();
}	

// This method opens an image using openslide and removes the alpha channel
void ReadNode::open_image(){

	std::cout << _image_path << std::endl;
	openslide_t *oslide = openslide_open(_image_path.c_str());
	if(oslide != NULL){
		
		// Declare variables
		int64_t w, h;
		openslide_get_level_dimensions(oslide, 2, &w, &h);
		uint32_t *buf = g_new(uint32_t, w * h);
		uint32_t *out = g_new(uint32_t, w * h);

		std::vector<cv::Mat> channels;
		cv::Mat r = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
		cv::Mat g = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
		cv::Mat b = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);

		// Read region
		openslide_read_region(oslide, buf, 2, 0, 0, w, h);

		// Convert to RGBX
		for (int64_t i = 0; i < w * h; i++) {
			//cout << i;

			uint32_t pixel = buf[i];
			uint8_t a = pixel >> 24;
			if (a == 255) {
	     	   // Common case.  Compiles to a shift and a BSWAP.
				out[i] = GUINT32_TO_BE(pixel << 8);
			} else if (a == 0) {
	        	// Less common case.  Hardcode white pixel; could also
	        	// use value from openslide.background-color property
	        	// if it exists
				out[i] = GUINT32_TO_BE(0xffffff00u);
			} else {
	        	// Unusual case.
				uint8_t r = 255 * ((pixel >> 16) & 0xff) / a;
				uint8_t g = 255 * ((pixel >> 8) & 0xff) / a;
				uint8_t b = 255 * (pixel & 0xff) / a;
				out[i] = GUINT32_TO_BE(r << 24 | g << 16 | b << 8);
			}

			// Convert to opencv
			int k = i/w;
			int l = i%w;

			r.at<uchar>(k, l) = 255 * ((pixel >> 16) & 0xff);
			g.at<uchar>(k, l) = 255 * ((pixel >> 8) & 0xff);
			b.at<uchar>(k, l) = 255 * (pixel & 0xff);
		}

		// Show image
		channels.push_back(r);
		channels.push_back(g);
		channels.push_back(b);

		merge(channels, _entire_image);

		// Close openslide object
		openslide_close(oslide);
	}
}

void ReadNode::crop_cells(){
	if(!_entire_image.empty()){
		cv::Size s = _entire_image.size();
		int _entire_image_height = s.height;
		int _entire_image_width = s.width;

		for (std::vector<cv::Mat>::size_type i= 0; i != _cells_coordinates.size(); i++) {

			int x = std::get<0>(_cells_coordinates[i]);
			int y = std::get<1>(_cells_coordinates[i]);
			

			if((x-SHIFT >=0) && (y-SHIFT >=0) && (x+SHIFT < _entire_image_width) && (y+SHIFT < _entire_image_height)){
				cv::Point tl(x-SHIFT, y-SHIFT);
				cv::Point br(x+SHIFT, y+SHIFT);

				// Setup a rectangle to define region of interest
				cv::Rect cellROI(tl, br);

				// Crop image
				cv::Mat croppedImage = _entire_image(cellROI);

				// Store image
				_extracted_images.push_back(croppedImage);
			}
			else{
				std::cout << "Crop out of bound" << std::endl;
			}
		}
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
}

void ReadNode::show_entire_image(){
	if(!_entire_image.empty()){
		std::cout << "Showing Entire Image" << std::endl;
		imshow("img", _entire_image);
		cv::waitKey(0);
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
}

void ReadNode::show_cropped_cells(){
	if(!_extracted_images.empty()){
		for(std::vector<cv::Mat>::size_type i =0; i < _extracted_images.size(); i++){

			std::cout << "Showing Cropped Images" << std::endl;
			imshow("img" + std::to_string(i), _extracted_images.at(i));
		}
		cv::waitKey(0);
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
}

bool ReadNode::get_output(std::vector<cv::Mat> &out){
	out = _extracted_images;
	return true;
}