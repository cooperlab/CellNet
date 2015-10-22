#include "read_write_node.h"
#include "edge.h"
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include "utils.h"
#include <boost/filesystem.hpp>
#define SHIFT 100

ReadWriteNode::ReadWriteNode(std::string id, std::vector<std::string> image_names, std::vector<std::string> image_paths, std::vector<std::vector<std::tuple<float, float>>> cells_coordinates_set, std::vector<std::vector<int>> input_labels, int mode): Node(id, mode), _image_names(image_names), _image_paths(image_paths), _cells_coordinates_set(cells_coordinates_set), i_ptr(0), _input_labels(input_labels){
	
	 runtime_total_first = utils::get_time();
}


void *ReadWriteNode::run(){
	
	std::vector<cv::Mat> extracted_images;
	cv::Mat entire_image;

	// Execute
	int i = get_input();
	while(i >= 0){
		
		// Declare variable
		int64_t min_x, min_y;

		// Read and crop images
		entire_image = open_image(_image_paths.at(i), i, min_x, min_y);
		extracted_images = crop_cells(entire_image, _cells_coordinates_set[i], min_x, min_y);
		save_images(extracted_images, _input_labels[i], _image_names[i]);

		// Release memory
		extracted_images.clear();
		entire_image.release();

		// Execute
 		i = get_input();
	}

	// Notify it has finished
	for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
		_out_edges.at(i)->set_in_node_done();
	}	
	std::cout << "ReadWriteNode complete" << std::endl;
	
	return NULL;
}	

void ReadWriteNode::save_images(std::vector<cv::Mat> extracted_images, std::vector<int> labels, std::string image_name){

	// Create folder
	boost::filesystem::create_directory("../../dataset_large/" + image_name);

	// Save cropped images
	for(int k=0; k < extracted_images.size(); k++){

		cv::imwrite("../../dataset_large/" + image_name + "/sample_" + std::to_string(_counter++) + "_" + std::to_string(labels[k]) + ".jpg", extracted_images[k]);
	}
}

int ReadWriteNode::get_input(){

	boost::mutex::scoped_lock lk(_mutex);
	if(i_ptr < _image_paths.size()){
		return i_ptr++;
	}
	else{
		return -1;
	}
}

// This method gets the 20 X magnification layer 
int ReadWriteNode::get_layer(openslide_t *oslide){

	/*get # of levels*/
	int levels = openslide_get_level_count(oslide);
	
	/*loop through levels, get dimensions/downsample factor of each*/
	std::vector<double> magnification;
	int objective;
	std::stringstream str_objective;

	//std::cout << "num_levels: " << std::to_string(levels) << std::endl;
	for(int i = 0; i < levels; i++) {

		/*get level info*/
		str_objective << openslide_get_property_value(oslide, OPENSLIDE_PROPERTY_NAME_OBJECTIVE_POWER);
		str_objective >> objective; 
		magnification.push_back(objective / (double) openslide_get_level_downsample(oslide, (int32_t)i));
		//std::cout << "Magnification: " << magnification[i] << std::endl;
	}

	// Return the closest magnification to 20.0
	for(std::vector<double>::size_type i = 0; i < magnification.size(); i++){
		
		if(magnification[i] >= 20.0){
			std::cout << "Picked mag: " << magnification[i] << std::endl;
			return i;
		}
	}

	// Return the second highest value
	std::cout << "Magnification: " << magnification[magnification.size()-1] << std::endl;
	return magnification.size()-1;
}

// Read the target portion of an slide
cv::Mat ReadWriteNode::open_image_region(openslide_t *oslide, int layer_i, float w, float h, float x, float y){

	// Declare buff
	// Read region
	uint32_t *buf = g_new(uint32_t, w * h);

	// Read region
	openslide_read_region(oslide, buf, x, y, layer_i, w, h);

	// Convert to RGBX
	for (int64_t i = 0; i < w * h; i++) {

		uint32_t pixel = buf[i];
		uint8_t a = pixel >> 24;

		if (a == 255) {
     	   // Common case.  Compiles to a shift and a BSWAP.
			buf[i] = GUINT32_TO_BE(pixel << 8);
		} else if (a == 0) {
        	// Less common case.  Hardcode white pixel; could also
        	// use value from openslide.background-color property
        	// if it exists
			buf[i] = GUINT32_TO_BE(0xffffff00u);
		} else {
        	// Unusual case.
			uint8_t r = 255 * ((pixel >> 16) & 0xff) / a;
			uint8_t g = 255 * ((pixel >> 8) & 0xff) / a;
			uint8_t b = 255 * (pixel & 0xff) / a;
			buf[i] = GUINT32_TO_BE(r << 24 | g << 16 | b << 8);
		}
	}
	
	// Convert to opencv
	// Get XBGR channels
	cv::Mat int_XBGR = cv::Mat(h, w, CV_8UC4, buf);
	cv::Mat entire_image;
	std::vector<cv::Mat> XBRG_channels;
	cv::split(int_XBGR, XBRG_channels);

	// Pop X channels
	XBRG_channels.pop_back();

	// Merge channels BGR back to image
	cv::merge(XBRG_channels, entire_image);

	// Close openslide object
	free(buf);

	return entire_image;
}

// This method opens an image using openslide and removes the alpha channel
cv::Mat ReadWriteNode::open_image(std::string image_path, int k, int64_t &min_x, int64_t &min_y){

	std::cout << image_path << std::endl;
	cv::Mat entire_image;

	openslide_t *oslide = openslide_open(image_path.c_str());
	if(oslide != NULL){

		int layer_i = get_layer(oslide);
		std::cout << "layer: " << std::to_string(layer_i) << std::endl;
		
		// Declare variables
		int64_t w, h, w0, h0;
		openslide_get_level_dimensions(oslide, layer_i, &w0, &h0);

		get_bb(k, w, h, min_x, min_y);
		
		float area = (w * h);
		float area0 = (w0*h0);

		std::cout << "Area compression: " << std::to_string((1-(area/area0))*100) << '%' << std::endl; 

		entire_image = open_image_region(oslide, layer_i, w, h, min_x, min_y);
	}

	openslide_close(oslide);
	return entire_image;
}

void ReadWriteNode::get_bb(int64_t k, int64_t &w, int64_t &h, int64_t &x, int64_t &y){

	int64_t min_x = std::get<0>(_cells_coordinates_set[k][0]);
	int64_t max_x = std::get<0>(_cells_coordinates_set[k][0]);
	int64_t min_y = std::get<1>(_cells_coordinates_set[k][0]);
	int64_t max_y = std::get<1>(_cells_coordinates_set[k][0]);

	for(int i=0; i < _cells_coordinates_set[k].size(); i++){

		int64_t x_i = std::get<0>(_cells_coordinates_set[k][i]);
		int64_t y_i = std::get<1>(_cells_coordinates_set[k][i]);

		if(x_i < min_x){
			min_x = x_i;
		}
		if(x_i > max_x){
			max_x = x_i;
		}
		if(y_i < min_y){
			min_y = y_i;
		}
		if(y_i > max_y){
			max_y = y_i;
		}
	}

	x = min_x - SHIFT;
	y = min_y - SHIFT;
	w = max_x - min_x + 2*SHIFT + 1;
	h = max_y - min_y + 2*SHIFT + 1;

	//std::cout << "min_x: " << min_x << " " << "min_y: " << min_y << std::endl; 
	//std::cout << "max_x: " << max_x << " " << "max_y: " << max_y << std::endl; 

}



std::vector<cv::Mat> ReadWriteNode::crop_cells(cv::Mat entire_image, std::vector<std::tuple<float, float>> _cells_coordinates, int64_t offset_x, int64_t offset_y){

	std::vector<cv::Mat> extracted_images;

	if(!entire_image.empty()){

		cv::Size s = entire_image.size();
		int _entire_image_height = s.height;
		int _entire_image_width = s.width;
		//std::cout << "height: " << _entire_image_height << " " << "width: " << _entire_image_width << std::endl; 
		for (std::vector<cv::Mat>::size_type i= 0; i != _cells_coordinates.size(); i++) {

			int x = std::get<0>(_cells_coordinates[i]);
			int y = std::get<1>(_cells_coordinates[i]);
			
			//std::cout << "x: " << std::to_string(x+SHIFT-offset_x) << " " << "y: " << std::to_string(y+SHIFT-offset_y) << std::endl;
			if((x-SHIFT-offset_x >=0) && (y-SHIFT-offset_y >=0) && (x+SHIFT-offset_x < _entire_image_width) && (y+SHIFT-offset_y < _entire_image_height)){
				
				cv::Point tl(x-SHIFT-offset_x, y-SHIFT-offset_y);
				cv::Point br(x+SHIFT-offset_x, y+SHIFT-offset_y);

				// Setup a rectangle to define region of interest
				cv::Rect cellROI(tl, br);

				// Crop image
				cv::Mat croppedImage = entire_image(cellROI);

				// Store image
				extracted_images.push_back(croppedImage);
			}
			else{
				std::cout << "Crop out of bound" << std::endl;
			}
		}
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
	return extracted_images;
}

void ReadWriteNode::show_entire_image(cv::Mat entire_image){
	if(!entire_image.empty()){
		//std::cout << "Showing Entire Image" << std::endl;
		//cv::imshow("img", entire_image);
		//cv::waitKey(0);

		cv::imwrite("/home/nelson/CellNet/entire_image.png", entire_image);
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
}

void ReadWriteNode::show_cropped_cells(std::vector<cv::Mat> extracted_images){
	if(!extracted_images.empty()){
		for(std::vector<cv::Mat>::size_type i =0; i < extracted_images.size(); i++){

			std::cout << "Showing Cropped Images" << std::endl;
			imshow("img" + std::to_string(i), extracted_images.at(i));
		}
		cv::waitKey(0);
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
}
